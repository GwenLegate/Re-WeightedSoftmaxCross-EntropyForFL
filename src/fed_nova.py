import wandb
from models import ResNet18, ResNet34
from utils import wandb_setup, zero_last_hundred, split_dataset, get_dataset, compute_accuracy
from data_utils import dataset_config
from client_utils import get_client_labels, train_test, DatasetSplit
from eval_utils import test_inference, validation_inference
from options import args_parser, validate_args
import os
import copy
import torch
import numpy as np
from torch import nn

def init_nets(num_clients, args):
    nets = {net_i: None for net_i in range(num_clients)}
    for net_i in range(num_clients):
        if args.model == "resnet18":
            net = ResNet18(args)
        elif args.model == 'resnet34':
            net = ResNet34(args)
        else:
            print('model not implemented')
            exit(1)

        if len(args.continue_train) > 0:
            net.load_state_dict(torch.load(args.continue_train))
        nets[net_i] = net

    return nets

'''def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    print('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    print('>> Pre-Training Training accuracy: {}'.format(train_acc))
    print('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    tau = 0

    for epoch in range(epochs):
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

    a_i = (tau - args.momentum * (1 - pow(args.momentum, tau)) / (1 - args.momentum)) / (1 - args.momentum)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)

    return train_acc, test_acc, a_i, norm_grad'''

def my_train_net_fednova(args, net_id, net, global_model, train_dataloader, proportions):
    proportions = np.asarray(proportions)
    print('Training network %s' % str(net_id))
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.client_lr,
                                     weight_decay=args.reg)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.client_lr,
                                    momentum=args.momentum, weight_decay=args.reg)

    if args.mask:
        criterion = nn.NLLLoss().to(args.device)
    else:
        criterion = nn.CrossEntropyLoss().to(args.device)

    train_dataloader = [train_dataloader]
    tau = 0

    for epoch in range(args.local_ep):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(args.device), target.to(args.device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                if args.mask:
                    out = weighted_log_softmax(args, out, proportions)

                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())

    a_i = (tau - args.momentum * (1 - pow(args.momentum, tau)) / (1 - args.momentum)) / (1 - args.momentum)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)

    return a_i, norm_grad

def my_local_train_net_fednova(args, nets, selected, global_model, client_datasets, train_data, validation_data):
    a_list = []
    d_list = []
    n_list = []
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        # move the model to cuda device:
        net.to(args.device)

        train_idxs = client_datasets[net_id]['train']
        validation_idxs = client_datasets[net_id]['validation']

        # label proportions in dataset
        proportions = get_client_labels(train_data, user_groups, args.num_workers, args.num_classes, proportions=True)
        client_labels = get_client_labels(train_data, user_groups, args.num_workers, args.num_classes)

        trainloader, testloader = train_test(args, train_data, validation_data, list(train_idxs), list(validation_idxs),
                                             args.num_workers)

        # move the model to cuda device:
        net.to(device)

        #trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        a_i, d_i = my_train_net_fednova(args, net_id, net, global_model, trainloader, proportions[net_id])

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(client_labels)
        n_list.append(n_i)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list

'''def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.num_clients - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.num_clients-1)
        else:
            noise_level = args.noise / (args.num_clients - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local)
        n_list.append(n_i)
        print("net %d final test acc %f" % (net_id, testacc))

        avg_acc += testacc


    avg_acc /= len(selected)


    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list'''

#computes softmax weighted by class proportions from the client
def weighted_log_softmax(args, logits, proportions):
    alphas = torch.from_numpy(proportions).to(args.device)
    log_alphas = alphas.log().clamp_(min=-1e9)
    deno = torch.logsumexp(log_alphas + logits, dim=-1, keepdim=True)
    return log_alphas + logits - deno

def evaluate_client_model(args, model, train_ds, val_ds, all_client_data):
    """ evaluates an individual client model on the datasets of all other clients
    Args:
        model: the client model to test
    Returns: A vector of accuracies using the model of client i and the datasets of all clients i and j (where j neq i)
            """
    all_acc = []
    model.to(args.device)
    model.eval()

    for idx in range(args.num_clients):
        _, testloader = train_test(args, train_ds, val_ds,
                                   list(all_client_data[idx]['train']),
                                   list(all_client_data[idx]['validation']),
                                   args.num_workers)
        total, correct = 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        all_acc.append(correct / total)
    return all_acc

if __name__ == '__main__':
    args = args_parser()
    device = torch.device(args.device)

    # need to set momentum to 0
    args.momentum = 0
    # set args dependent on dataset
    dataset_config(args)
    validate_args(args)

    # set up wandb connection
    if args.wandb:
        wandb_setup(args)

    # create dir to save run artifacts
    run_dir = f'/scratch/{os.environ.get("USER", "glegate")}/{args.wandb_run_name}'
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir, mode=0o755, exist_ok=True)

    # set lists for last 100 item average
    last_hundred_test_loss, last_hundred_test_acc, last_hundred_val_loss, last_hundred_val_acc = zero_last_hundred()

    # load dataset and user groups
    train_dataset, validation_dataset, test_dataset = get_dataset(args)
    # splits dataset among clients
    if len(args.continue_train) > 0:
        user_groups_path = f"{args.continue_train.rsplit('/', 1)[0]}/user_groups.pt"
        user_groups = torch.load(user_groups_path)
    else:
        user_groups = split_dataset(train_dataset, args)
    # list of set of labels present for each client
    client_labels = get_client_labels(train_dataset, user_groups, args.num_workers, args.num_classes)
    # save the user_groups dictionary for later access
    user_groups_to_save = f'/scratch/{os.environ.get("USER", "glegate")}/{args.wandb_run_name}/user_groups.pt'
    torch.save(user_groups, user_groups_to_save)


    # combine indicies for validation sets of each client to test global model on complete set
    # use same loop to get total number of samples in training set
    data_sum = 0
    for i in range(args.num_clients):
        data_sum += len(user_groups[i]['train'])
        if i == 0:
            idxs_val = user_groups[i]['validation']
        else:
            idxs_val = np.concatenate((idxs_val, user_groups[i]['validation']), axis=0)

    validation_dataset_global = DatasetSplit(validation_dataset, idxs_val)

    if args.dataset_compare:
        avg_forgetting_dict = {}

    print("Initializing nets (fednova)")

    nets = init_nets(args.num_clients, args)
    global_models = init_nets(1, args)
    global_model = global_models[0]
    global_model.to(device)
    global_para = global_model.state_dict()

    d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.num_clients)]
    d_total_round = copy.deepcopy(global_model.state_dict())
    for key in d_total_round:
        d_total_round[key] = 0
        for i in range(args.num_clients):
            d_list[i][key] = 0

    # portion of training samples at each client (current implementation all the same)
    portion = []
    for i in range(args.num_clients):
        portion.append(len(user_groups[i]['train']) / data_sum)

    #********* TRAINING STARTS HERE ******************
    for round in range(args.epochs):
        print("in Round:" + str(round))

        arr = np.arange(args.num_clients)
        np.random.shuffle(arr)
        selected = arr[:int(args.num_clients * args.frac)]

        if round in [1, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999]:
            if args.dataset_compare:
                # pre training all model weights the same so only have to do this once
                acc_pre_individual_training = evaluate_client_model(args, model=copy.deepcopy(global_model),
                                                                    train_ds=train_dataset, val_ds=validation_dataset,
                                                                    all_client_data=user_groups)

        global_para = global_model.state_dict()

        for idx in selected:
            nets[idx].load_state_dict(global_para)

        _, a_list, d_list, n_list = my_local_train_net_fednova(args, nets, selected, global_model, user_groups,
                                                               train_dataset, validation_dataset)
        total_n = sum(n_list)
        d_total_round = copy.deepcopy(global_model.state_dict())
        for key in d_total_round:
            d_total_round[key] = 0.0
        for i in range(len(selected)):
            d_para = d_list[i]
            for key in d_para:
                d_total_round[key] += d_para[key] * n_list[i] / total_n

        # update global model
        coeff = 0.0
        for i in range(len(selected)):
            coeff = coeff + a_list[i] * n_list[i]/total_n

        updated_model = global_model.state_dict()
        for key in updated_model:
            if updated_model[key].type() == 'torch.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
            elif updated_model[key].type() == 'torch.cuda.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
            else:
                updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)
            global_model.to(device)

        if round in [1, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999]:
            if args.dataset_compare:
                avg, total = 0.0, 0.0
                for idx in range(args.num_clients):
                    if idx not in selected:
                        pass
                    else:
                        acc_post_individual_training = evaluate_client_model(args,
                                                                        model=copy.deepcopy(nets[idx]),
                                                                        train_ds=train_dataset,
                                                                        val_ds=validation_dataset,
                                                                        all_client_data=user_groups)
                        diff = [x - y for x, y in zip(acc_pre_individual_training, acc_post_individual_training)]
                        for d in diff:
                            avg += d
                        avg -= diff[idx]
                        total += len(diff) - 1
                round_avg = avg / total
                avg_forgetting_dict[round] = round_avg

        if round % 50 == 0:
            # save model as a backup every 50 epochs
            model_path = f'/scratch/{os.environ.get("USER", "glegate")}/{args.wandb_run_name}/global_model.pt'
            torch.save(global_model.state_dict(), model_path)

        # Test global model inference on validation set after each round use model save criteria
        val_acc, val_loss = validation_inference(args, global_model, validation_dataset_global, args.num_workers)
        print(f'Epoch {round} Validation Accuracy {val_acc * 100}% \nValidation Loss {val_loss}')

        # print global training loss after every 'i' rounds
        if (round + 1) % args.print_every == 0:
            if args.wandb:
                wandb.log({f'val_acc': val_acc,
                           f'val_loss': val_loss
                           }, step=round)

        if args.epochs - (round + 1) <= 100:
            last_hundred_val_loss.append(val_loss)
            last_hundred_val_acc.append(val_acc)
            test_acc, test_loss = test_inference(args, global_model, test_dataset, args.num_workers)
            last_hundred_test_loss.append(test_loss)
            last_hundred_test_acc.append(test_acc)

    # save compare dict
    print(avg_forgetting_dict)
    if args.dataset_compare:
        comapre_dict_path = f'/scratch/{os.environ.get("USER", "glegate")}/{args.wandb_run_name}/avg_forgetting.pt'
        torch.save(avg_forgetting_dict, comapre_dict_path)

    # save final model after training
    model_path = f'/scratch/{os.environ.get("USER", "glegate")}/{args.wandb_run_name}/global_model.pt'
    torch.save(global_model.state_dict(), model_path)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset, args.num_workers)

    # last 100 avg acc and loss
    last_hundred_test_loss = sum(last_hundred_test_loss) / len(last_hundred_test_loss)
    last_hundred_test_acc = sum(last_hundred_test_acc) / len(last_hundred_test_acc)
    last_hundred_val_loss = sum(last_hundred_val_loss) / len(last_hundred_val_loss)
    last_hundred_val_acc = sum(last_hundred_val_acc) / len(last_hundred_val_acc)

    if args.wandb:
        wandb.log({f'val_acc': val_acc, f'test_acc': test_acc, f'last_100_val_acc': last_hundred_val_acc,
                   f'last_100_test_acc': last_hundred_test_acc})

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Validation Accuracy: {:.2f}%".format(100 * val_acc))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print("|---- Last 100 Validation Accuracy: {:.2f}%".format(100 * last_hundred_val_acc))
    print("|---- Last 100 Test Accuracy: {:.2f}%".format(100 * last_hundred_test_acc))
