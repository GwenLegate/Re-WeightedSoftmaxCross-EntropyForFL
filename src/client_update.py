#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from utils import reset_linear_layer, get_delta
from models import CNN_C, ResNet18
from client_utils import DatasetSplit, train_test

class LocalUpdate(object):
    def __init__(self, args, train_dataset, validation_dataset, idx, client_labels, all_client_data, round):
        self.args = args
        self.round = round
        self.client_idx = idx
        self.model_after_training = None
        self.lr = self.args.client_lr
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.all_client_data = all_client_data
        self.train_idxs = self.all_client_data[self.client_idx]['train']
        self.validation_idxs = self.all_client_data[self.client_idx]['validation']
        self.trainloader, self.testloader = train_test(self.args, self.train_dataset, self.validation_dataset,
                                                       list(self.train_idxs), list(self.validation_idxs), args.num_workers)
        self.device = args.device
        if args.dataset == 'celeba':
            if self.args.mask:
                self.criterion = nn.BCELoss().to(self.device)
            else:
                self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        else:
            if self.args.mask:
                self.criterion = nn.NLLLoss().to(self.device)
            else:
                self.criterion = nn.CrossEntropyLoss().to(self.device)
        # set of all labels (training and validation) in the client dataset
        self.labels = client_labels
        if self.args.dataset == 'celeba':
            self.client_data_proportions = self.data_proportions_multi_label()
        else:
            self.client_data_proportions = self.data_proportions()
        print(f'ln 41 client proportions {self.client_data_proportions}')

    def get_model(self):
        """
        Returns local model after update
        """
        return self.model_after_training

    def get_client_labels(self, dataset, train_idxs, validation_idxs, num_workers, unique=True):
        """
        Creates a set of all labels present in both train and validation sets of a client dataset
        Args:
            dataset: the complete dataset being used
            train_idxs: the indices of the training samples for the client
            validation_idxs: the indices of the validation samples for the client
            num_workers: how many sub processes to use for data loading
            unique: (bool) if True return the set of client labels, if False returns
        Returns: Set of all labels present in both train and validation sets of a client dataset.
        """
        all_idxs = np.concatenate((train_idxs, validation_idxs), axis=0)
        dataloader = DataLoader(DatasetSplit(dataset, all_idxs), batch_size=len(dataset), shuffle=False,
                        num_workers=num_workers, pin_memory=True)
        _, labels = zip(*[batch for batch in dataloader])

        if unique:
            return labels[0].unique()
        else:
            return labels[0]

    def data_proportions_multi_label(self):
        """
            computes the percentage of samples in the client dataset with a positive value for each label
            Returns:
                a vector of size num_classes with a value between 0 and 1 for each entry that represents the percentage of
                the client distribution that has that label
            """
        client_labels = np.array(self.get_client_labels(self.train_dataset, self.train_idxs, self.validation_idxs,
                                                        self.args.num_workers, False))
        print(f'ln 75 client labels {client_labels}\n {client_labels.shape}')
        count_labels = client_labels.shape[0]
        count_client_labels = np.sum(client_labels, axis=0)
        print(f'ln 89 count labels: {count_client_labels}\nprops: {count_client_labels / count_labels}')
        return count_client_labels / np.array(count_labels)

    def data_proportions(self):
        """
        computes the percentage of the client dataset each class label is responsible for
        Returns:
            a vector of size num_classes with a value between 0 and 1 for each entry that represents the percentage of
            the client distribution that label composes
        """
        client_labels = np.array(self.get_client_labels(self.train_dataset, self.train_idxs, self.validation_idxs,
                                                        self.args.num_workers, False))
        count_labels = client_labels.shape[0]
        count_client_labels = []
        for i in range(self.args.num_classes):
            count_client_labels.append(int(np.argwhere(client_labels == i).shape[0]))
        count_client_labels = np.array(count_client_labels)
        count_labels = np.array(count_labels)
        return count_client_labels / count_labels

    def compute_grad(self, model):
        """
        independently computes the gradient for evaluation purposes for one step without affecting the regular training.
        uses batch size that is the entire client dataset for this step
        Args:
            model: the model to use for the gradient computation
        Returns:
            state dict of gradients
        """
        trainloader = DataLoader(DatasetSplit(self.train_dataset, self.train_idxs),
                                 batch_size=int(len(self.train_idxs)), shuffle=True, num_workers=self.args.num_workers,
                                 pin_memory=True)
        model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            model.zero_grad()
            logits = model(images)

            if self.args.mask:
                logits = self.weighted_log_softmax(logits)

            loss = self.criterion(logits, labels)
            loss.backward()
            grad_dict = {k:v.grad for k, v in zip(model.state_dict(), model.parameters())}
            return grad_dict

    def weighted_log_softmax(self, logits):
        """
        computes softmax weighted by class proportions from the client
        Args:
            logits: logits for the mini batch under consideration
        Returns:
            softmax weighted by class proportion
        """
        alphas = torch.from_numpy(self.client_data_proportions).to(self.device)
        log_alphas = alphas.log().clamp_(min=-1e9)
        deno = torch.logsumexp(log_alphas + logits, dim=-1, keepdim=True)
        return log_alphas + logits - deno

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # array with tuple of (regular_acc, 10_epoch_1st_layer_train_acc, diff) before and after client training
        optional_eval_results = None

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        if self.args.decay == 1:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

        if self.args.weight_delta or self.args.momentum != 0:
            initial_weights = copy.deepcopy(model).state_dict()

        local_iter_count = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                print(f'ln153 img size {images.size()}, label size {labels.size()}')
                local_iter_count += 1
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                logits = model(images)
                print(f' ln 158 logits {logits.size()}')

                if self.args.mask:
                    logits = self.weighted_log_softmax(logits)
                    print(f' ln 162 masked logits {logits}')

                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                if self.args.decay == 1:
                    scheduler.step()

                batch_loss.append(loss.item())

                if self.args.local_iters is not None:
                    if self.args.local_iters == local_iter_count:
                        break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # save model after training for later eval
        self.model_after_training = model

        # get batch statistics for client and return them in a list
        if self.args.log_batch_stats:
            batch_stats = {}
            for k, v in model.state_dict().items():
                if 'running' in k:
                    batch_stats[k] = v

            optional_eval_results = batch_stats

        # get list of modified last layer weights (pos 0) and biases (pos 1)
        if self.args.weight_ll:
            optional_eval_results = self.zero_ll_noncontributing_classes(model.state_dict())

        if self.round in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500,
                     2000, 2500, 3000, 3500, 3999]:
            if self.args.grad_alignment:
                grad_eval_model = copy.deepcopy(self.model_after_training)
                grads = self.compute_grad(grad_eval_model)
                optional_eval_results = grads
            if self.args.weight_delta:
                optional_eval_results = get_delta(copy.deepcopy(self.model_after_training).state_dict(), initial_weights)
            if self.args.momentum != 0:
                delta = get_delta(copy.deepcopy(self.model_after_training).state_dict(), initial_weights)
                return delta, sum(epoch_loss) / len(epoch_loss), optional_eval_results

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), optional_eval_results

    def zero_ll_noncontributing_classes(self, weights):
        '''
        Zeros weights and biases of last layer rows where the client does not have that class label present in it's distribution
        Args:
            weights: state dict of model weights
        Returns:
            list containing adjusted weights at position 0 and adjusted biases at position 1
        '''
        client_labels = self.get_client_labels(self.train_dataset, self.train_idxs, self.validation_idxs,
                                           self.args.num_workers)
        zero_tensor = torch.zeros((1, 512))
        zeroed_bias = copy.deepcopy(weights['fc.bias'])
        zeroed_weights = copy.deepcopy(weights['fc.weight'])
        for i in range(self.args.num_classes):
            if i in client_labels:
                pass
            else:
                zeroed_bias[i] = 0
                zeroed_weights[i, :] = zero_tensor
        return [zeroed_weights, zeroed_bias]

    def probe(self, model_weights):
        """
        Train the client model with all but last linear layer frozen for 15 epochs (train with masking)
        Args:
            model_weights: the weights of the client model to train
        Returns: accuracy after training with frozen layers (acc without masking)
        """
        model_weights = reset_linear_layer(model_weights, self.args.model)
        if self.args.model == 'cnn_c':
            model = CNN_C(args=self.args)
        else:
            model = ResNet18(args=self.args)

        # freeze weights
        for name, param in model.named_parameters():
            if param.requires_grad and 'fc' not in name:
                param.requires_grad = False

        model.load_state_dict(model_weights)
        model.to(self.device)
        model.train()
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        for epoch in range(15):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                logits = model(images)

                if self.args.mask:
                    logits = self.weighted_log_softmax(logits)

                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

        acc, _ = self.inference(model)
        return acc

    def evaluate_client_model(self, idxs_clients, model, lp=False):
        """ evaluates an individual client model on the datasets of all other clients
        Args:
            idxs_clients: array of indices of the other clients selected for this round
            model: the client model to test
            lp: train linear probe on local client data prior to testing on each client's local data
        Returns: A vector of accuracies using the model of client i and the datasets of all clients i and j (where j neq i)
                """
        all_acc = []
        model.eval()

        for idx in range(100):
            if lp:
                lp_model = copy.deepcopy(model).state_dict()
                acc = self.probe(lp_model)
            else:
                _, self.testloader = train_test(self.args, self.train_dataset, self.validation_dataset,
                                            list(self.all_client_data[idx]['train']),
                                            list(self.all_client_data[idx]['validation']), self.args.num_workers)
                acc, _ = self.inference(model)
            all_acc.append(acc)
        return all_acc

    def inference(self, model):
        """
        Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

    def freeze(self, model):
        # NOTE: code not currently in use
        """
        freezes weights of last linear layer of the model so the gradients will not propagate and creates a new linear
        layer with the appropriate gradients frozen
        Args:
            model: the model to act on
        Returns weights and biases of the last linear layer with the appropriate rows frozen
        """
        if self.args.model == 'cnn_c':
            mask = torch.zeros_like(model.fc3.weight)
            mask[self.labels] = 1
            pos = model.fc3.weight * mask
            neg = model.fc3.weight * (1 - mask)
            backward_masked = pos + neg.detach()
            (backward_masked - model.fc3.weight).abs().sum()  # make sure W has not changed
            backward_masked.retain_grad = True

            mask_b = torch.zeros_like(model.fc3.bias)
            mask_b[self.labels] = 1
            pos_b = model.fc3.bias * mask_b
            neg_b = model.fc3.bias * (1 - mask_b)
            backward_masked_b = pos_b + neg_b.detach()
            (backward_masked_b - model.fc3.bias).abs().sum()  # make sure B has not changed
            backward_masked_b.retain_grad = True

        elif self.args.model == 'resnet18':
            mask = torch.zeros_like(model.fc.weight)
            mask[self.labels] = 1
            pos = model.fc.weight * mask
            neg = model.fc.weight * (1 - mask)
            backward_masked = pos + neg.detach()
            (backward_masked - model.fc.weight).abs().sum()  # make sure W has not changed
            backward_masked.retain_grad = True

            mask_b = torch.zeros_like(model.fc.bias)
            mask_b[self.labels] = 1
            pos_b = model.fc.bias * mask_b
            neg_b = model.fc.bias * (1 - mask_b)
            backward_masked_b = pos_b + neg_b.detach()
            (backward_masked_b - model.fc.bias).abs().sum()  # make sure B has not changed
            backward_masked_b.retain_grad = True
        else:
            print(f'current model is {self.args.model}. Model must be one of resnet18 or cnn_c')

        return backward_masked, backward_masked_b


