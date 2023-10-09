#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from time import time
import multiprocessing as mp
import random
import wandb
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sampling import mnist_iid, mnist_noniid
from sampling import cifar_iid, shard_noniid, equal_class_size_noniid_dirichlet, femnist_iid, \
    unequal_class_size_noniid_dirichlet, celeba_split
import os
import math
from femnist_dataset import FEMNIST
import numpy as np
from imagenet32 import Imagenet32

def get_dataset(args):
    """
    Returns train, validation and test datasets
    """
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data_dir = '../data/'

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
            validation_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform_test)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform_test)

        if args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
            validation_dataset = datasets.CIFAR100(data_dir, train=True, download=False, transform=transform_test)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=False, transform=transform_test)

    elif args.dataset == 'celeba':
        # transforms, mean and std values from https://github.com/bozliu/CelebA-Challenge/blob/main/main.py
        # size of these images is 178x218
        # labels are boolean 0, 1 for each of the 40 attributes
        data_dir = '../data/'

        transform_train = transforms.Compose([
            transforms.RandomGrayscale(p=0.5),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.CelebA(root=data_dir, split='train', target_type='attr', download=False, transform=transform_train)
        validation_dataset = datasets.CelebA(root=data_dir, split='train', target_type='attr', download=False, transform=transform_test)
        test_dataset = datasets.CelebA(root=data_dir, split='test', target_type='attr', download=False, transform=transform_test)

    elif args.dataset == 'femnist':
        data_dir = '../data/femnist/'
        train_dataset = FEMNIST(root=data_dir, train=True, download=True)
        mean = train_dataset.train_data.float().mean()
        std = train_dataset.train_data.float().std()

        apply_transform = transforms.Compose([
            transforms.RandomCrop(24, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_dataset = FEMNIST(data_dir, train=True, download=False, transform=apply_transform)
        validation_dataset = FEMNIST(data_dir, train=True, download=False, transform=test_transform)
        test_dataset = FEMNIST(data_dir, train=False, download=False, transform=test_transform)

    elif args.dataset == 'imagenet32':
        data_dir = '../data/imagenet32/imagenet32_train'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        ds = Imagenet32(root=data_dir)

    elif args.dataset == 'mnist' or 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)
            validation_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=apply_transform)
        else:
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=apply_transform)
            validation_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False, transform=apply_transform)

    return train_dataset, validation_dataset, test_dataset

def split_dataset(train_dataset, args):
    ''' Splits the dataset between args.num_clients clients and further partitions each clients subset into training
        and validation sets
    Args:
        train_dataset: the complete training dataset
        args: the user specified options for the run
    Returns:
        user_groups: a nested dict where keys are the user index and values are another dict with keys
        'train' and 'validation' which contain the corresponding sample indices for training and validation subsets of
        each clients partition
    '''
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        # sample training data amongst clients
        if args.iid:
            # Sample IID user data
            user_groups = cifar_iid(train_dataset, args.num_clients)
        else:
            # Sample Non-IID user data
            if args.dirichlet:
                print(f'Creating non iid client datasets using the dirichlet distribution')
                # non iid data distributions for clients created by dirichlet distribution
                user_groups = equal_class_size_noniid_dirichlet(train_dataset, args.alpha, args.num_clients, args.num_classes)
            else:
                # non iid data distributions for clients created by dirichlet distributionvia shard method in Mcmahan(2016)
                print(f'Creating non iid client datasets using shards')
                user_groups = shard_noniid(train_dataset, args.num_clients)
    elif args.dataset == 'celeba':
        user_groups = celeba_split(train_dataset, args.num_clients)
    elif args.dataset == 'femnist':
        if args.dirichlet:
            user_groups = unequal_class_size_noniid_dirichlet(train_dataset, args.alpha, args.num_clients, args.num_classes)
        else:
            user_groups = femnist_iid(train_dataset, args.num_clients)
    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        # sample training data amongst clients
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_clients)
        else:
            user_groups = mnist_noniid(train_dataset, args.num_clients)
    return user_groups

def average_weights(w):
    """
    Returns the average of the local weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def ll_modify(global_w, modified_ll, num_classes, gamma):
    # determine how many clients have each class label
    label_count = [0 for _ in range(num_classes)]
    avg_weights = modified_ll[0][0]
    avg_bias = modified_ll[0][1]
    for i in range(len(modified_ll)):
        if i != 0:
            avg_weights += modified_ll[i][0]
            avg_bias += modified_ll[i][1]
        for j in range(num_classes):
            if modified_ll[i][1][j] == 0:
                if modified_ll[i][0][j, :].all() == 0:
                    pass
                else:
                    label_count[j] += 1
            else:
                label_count[j] += 1
    for i in range(len(label_count)):
        if label_count[i] > 0:
            avg_weights[i, :] /= label_count[i]
            avg_bias[i] /= label_count[i]
    for i in range(num_classes):
        if label_count[i] != 0:
            global_w['fc.bias'][i] = ((1 - gamma) * global_w['fc.bias'][i]) + (gamma * avg_bias[i])
            global_w['fc.weight'][i, :] = ((1 - gamma) * global_w['fc.weight'][i, :]) + (gamma * avg_weights[i, :])
    return global_w

def average_select_weights(w, client_labels, num_classes):
    """
    depreciated
    """
    contributions = torch.zeros(num_classes)
    for i in range(len(w)):
        for c in range(num_classes):
            if c in client_labels[i]:
                contributions[c] += 1
    no_update_indices = (contributions == 0).nonzero()
    no_update_indices = [idx.item() for idx in no_update_indices]

    # get last layer averages before modifying them
    ll_unmodified_avg_bias = w[0]['fc.bias']
    ll_unmodified_avg_weight = w[0]['fc.weight']

    for i in range(1, len(w)):
        ll_unmodified_avg_bias += w[i]['fc.bias']
        ll_unmodified_avg_weight += w[i]['fc.weight']

    ll_unmodified_avg_bias /= len(w)
    ll_unmodified_avg_weight /= len(w)

    # zero weights/biases not in client distribution
    zero_tensor = torch.zeros((1, 512))
    w_avg = copy.deepcopy(w[0])
    for i in range(num_classes):
        if i in client_labels[0] or i in no_update_indices:
            pass
        else:
            w_avg['fc.bias'][i] = 0
            w_avg['fc.weight'][i, :] = zero_tensor

    if len(w) > 1:
        for key in w_avg.keys():
            for i in range(1, len(w)):
                if key not in ['fc.bias', 'fc.weight']:
                    w_avg[key] += w[i][key]
                else:
                    for j in range(num_classes):
                        if j in client_labels[i] or j in no_update_indices:
                            if key == 'fc.bias':
                                w_avg['fc.bias'][j] += w[i]['fc.bias'][j]
                            else:
                                w_avg['fc.weight'][j, :] += w[i]['fc.weight'][j, :]
            if key not in ['fc.bias', 'fc.weight']:
                w_avg[key] = torch.div(w_avg[key], len(w))
            else:
                for k in range(num_classes):
                    if contributions[k] == 0:
                        contributions[k] = len(w)

                    if key == 'fc.bias':
                        w_avg[key][k] = w_avg[key][k] / contributions[k]
                    else:
                        w_avg[key][k] = torch.div(w_avg[key][k], contributions[k])
    return w_avg

def average_grads(local_grads):
    grad_avg = copy.deepcopy(local_grads[0])
    for i in range(1, len(local_grads)):
        for k, _ in grad_avg.items():
            grad_avg[k] += local_grads[i][k]
    for k, _ in grad_avg.items():
        try:
            grad_avg[k] *= 1 / len(local_grads)
        except RuntimeError:
            grad_avg[k] *= int(1 / len(local_grads))

    return grad_avg

def update_with_momentum(args, momentum, global_weights, global_deltas):
    # WIP for updates using deltas instead of weights
    momentum_update = copy.deepcopy(momentum)

    for k, v in global_weights.items():
        momentum_update[k] = (args.momentum * momentum[k]) + global_deltas[k]
        global_weights[k] -= args.global_lr * momentum_update[k]
    return momentum_update, global_weights

def update_with_momentum2(args, global_model, global_deltas):
    # option for sgd with weight decay and nesterov accelerate gradient as outlined in FedAvgM paper by Hsu et al.
    # TODO: fix, not working
    # optimizer = torch.optim.SGD(global_model.parameters(), lr=args.global_lr, weight_decay=1e-4, momentum=args.momentum, nesterov=True)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.global_lr, momentum=args.momentum)
    global_model.zero_grad()
    # manually set grads
    with torch.no_grad():
        for param, k in zip(global_model.parameters(), global_deltas.keys()):
            param.grad = global_deltas[k]
    optimizer.step()
    return global_model

def reset_linear_layer(weights, model):
    '''
    takes model weights and re-initializes last fc layer weights randomly. Only implemented for cnn_c_
    and resnet18
    '''
    assert model == 'cnn_c' or model == 'resnet18'
    def reinit(tensor, prev_tensor):
        '''this is pytroch default random initialization for fc layer, used for re initializing fc layers of both models'''
        try:
            bound = 1 / math.sqrt(tensor.size(1))
        except IndexError:
            # bias has same bound as it's layer, if tensor is 1d, it is a bais and bound is calculated from prev weight dims
            bound = 1 / math.sqrt(prev_tensor.size(1))
        return (2 * bound) * torch.rand_like(tensor, requires_grad=True) - bound

    if model == 'cnn_c':
        prev_weight = None
        for weight in weights:
            if 'fc3' in weight:
                new_init = reinit(weights[weight], prev_weight)
                weights[weight] = new_init
                prev_weight = weights[weight]
    if model == 'resnet18':
        prev_weight = None
        for key, value in weights.items():
            if 'fc' in key:
                new_init = reinit(value, prev_weight)
                weights.key = new_init
                prev_weight = value
    return weights

   # set random values for learning rates, local epochs, local batch size for hyperparameter search
def set_random_args(args):
    lrs = [7E-2, 5E-2, 3E-2, 1E-2, 7E-3, 5E-3, 3E-3, 1E-3, 7E-4]
    args.local_bs = random.randrange(5, 120, 5)  # sets a local batch size between 5 and 125 (intervals of 5)
    args.global_lr = 1
    idx = random.randrange(9)
    args.client_lr = lrs[idx]
    args.local_ep = random.randrange(4, 26)
    if idx < 4:
        args.epochs = random.randrange(1000, 3001, 250)
    else:
        args.epochs = random.randrange(2500, 4501, 250)


# method to empirically find a good number of workers for data loaders
def find_optimal_num_workers(training_data, bs):
    for num_workers in range(2, mp.cpu_count()):
        train_loader = DataLoader(training_data,shuffle=True,num_workers=num_workers,batch_size=bs,pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

def run_summary(args):
    print(f'Run Parameters:\n'
          f'\twandb name: {args.wandb_run_name}\n'
          f'\tlr: {args.client_lr}\n'
          f'\tiid: {args.iid}\n'
          f'\tclients: {args.num_clients}\n'
          f'\tfrcation of clients selected: {args.frac}\n'
          f'\tlocal bs: {args.local_bs}\n'
          f'\tlocal epochs: {args.local_ep}\n'
          f'\tlocal iterations: {args.local_iters}\n'
          f'\tmask: {args.mask}\n'
          f'\talpha: {args.alpha}\n'
          f'\tnorm: {args.norm}\n'
          f'\tdataset: {args.dataset}')

def wandb_setup(args):
    if args.wandb_run_name:
        os.environ['WANDB_NAME'] = args.wandb_run_name
        os.environ['WANDB_START_METHOD'] = "thread"
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    # need to set wandb run_dir to something I can access to avoid permission denied error.
    # See https://github.com/wandb/client/issues/3437
    wandb_path = f'/scratch/{os.environ.get("USER","glegate")}/wandb'
    if not os.path.isdir(wandb_path):
        os.makedirs(wandb_path, mode=0o755, exist_ok=True)

    # if using wandb check project and entity are set
    assert not args.wandb_project == '' and not args.wandb_entity == ''
    wandb.login()
    wandb.init(dir=wandb_path, project=args.wandb_project, entity=args.wandb_entity)
    general_args = {
        "client learning_rate": args.client_lr,
        "epochs": args.epochs,
        "dataset": args.dataset,
        "model": args.model,
        "number of kernels": args.kernel_num,
        "kernel size": args.kernel_sizes,
        "iid": args.iid,
        "clients": args.num_clients,
        "fraction of clients (C)": args.frac,
        "local epochs (E)": args.local_ep,
        "local iters": args.local_iters,
        "local batch size (B)": args.local_bs,
        "optimizer": args.optimizer,
        "mask": args.mask,
        "dirichlet": args.dirichlet,
        "alpha": args.alpha,
        "norm": args.norm,
        "decay": args.decay,
    }
    wandb.config.update(general_args)

def get_delta(params1, params2):
    '''
    Computes delta of model weights or gradients (params2-params1)
    grad update w+ = w - delta_w --> delta_w = w - w+
    Args:
        params1: state dict of weights or gradients
        params2: state dict of weights or gradients
    Returns:
        state dict of the delta between params1 and params2
    '''
    params1 = copy.deepcopy(params1)
    params2 = copy.deepcopy(params2)
    for k, _ in params1.items():
        params2[k] = params2[k].float()
        params2[k] -= params1[k].float()
    return params2

def zero_last_hundred():
    return [], [], [], []

def compute_accuracy(model, dataloader, device):
    """
    compute accuracy method from NIID-Bench, kept for consistancy with scaffold implementation
    """
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct/float(total)
