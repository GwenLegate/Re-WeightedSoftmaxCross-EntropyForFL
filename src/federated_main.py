#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import numpy as np
import scipy.stats
import wandb
import torch
from options import args_parser, validate_args
from client_update import LocalUpdate, DatasetSplit
from models import MLP, CNN_A, CNN_B, CNN_C, CNNLeNet, ResNet34, ResNet18
from utils import get_dataset, average_weights, average_select_weights, set_random_args, split_dataset, run_summary, \
    wandb_setup, zero_last_hundred, get_delta, ll_modify
from eval_utils import grad_cos_sim, validation_inference, test_inference, grad_diff, grad_diversity, wsm_inference, celeba_inference
from client_utils import get_client_labels
from data_utils import dataset_config

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    start_time = time.time()
    args = args_parser()
    # need to set momentum to 0 so FedAvgM code isn't triggered in client_update.py
    args.momentum = 0

    # set args dependent on dataset
    dataset_config(args)

    # set random values for a hyperparameter search
    if args.hyperparam_search:
        set_random_args(args)

    validate_args(args)

    # create dir to save run artifacts
    run_root = f'scratch/{os.environ.get("USER", "glegate")}'
    run_dir = f'/{run_root}/{args.wandb_run_name}'
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir, mode=0o755, exist_ok=True)

    # set up wandb connection
    if args.wandb:
        wandb_setup(args)

    # set lists for last 100 item average
    last_hundred_test_loss, last_hundred_test_acc, last_hundred_val_loss, last_hundred_val_acc = zero_last_hundred()

    # load dataset and user groups
    train_dataset, validation_dataset, test_dataset = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn_a':
        global_model = CNN_A(args=args)
    elif args.model == 'cnn_b':
        global_model = CNN_B(args=args)
    elif args.model == 'cnn_c':
        global_model = CNN_C(args=args)
    elif args.model == 'lenet':
        global_model = CNNLeNet(args=args)
    elif args.model == 'resnet18':
        global_model = ResNet18(args=args)
    elif args.model == 'resnet34':
        global_model = ResNet34(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # if this run is a continuation of training for a failed run, load previous model and client distributions
    if len(args.continue_train) > 0:
        global_model.load_state_dict(torch.load(args.continue_train))
        user_groups_path = f"{args.continue_train.rsplit('/', 1)[0]}/user_groups.pt"
        user_groups = torch.load(user_groups_path)
    else:
        # splits dataset among clients
        user_groups = split_dataset(train_dataset, args)

    # list of set of labels present for each client
    client_labels = get_client_labels(train_dataset, user_groups, args.num_workers, args.num_classes)

    # save the user_groups dictionary for later access
    user_groups_to_save = f'/{run_root}/{args.wandb_run_name}/user_groups.pt'
    torch.save(user_groups, user_groups_to_save)

    # combine indicies for validation sets of each client to test global model on complete set
    for i in range(args.num_clients):
        if i == 0:
            idxs_val = user_groups[i]['validation']
        else:
            idxs_val = np.concatenate((idxs_val, user_groups[i]['validation']), axis=0)

    validation_dataset_global = DatasetSplit(validation_dataset, idxs_val)

    # Set the model to training mode and send it to device.
    global_model.to(args.device)
    global_model.train()

    # log every 'print_every' epochs
    if args.wandb:
        wandb.watch(global_model, log_freq=args.print_every)
    run_summary(args)
    print(global_model)

    # copy global model weights
    global_weights = global_model.state_dict()
    train_loss, train_accuracy = [], []
    epoch = 0
    grad_eval = False
    delta_eval = False

    if args.linear_probe:
        lp_pre_train_dict = {}
        lp_post_train_dict = {}

    if args.dataset_compare:
        dataset_compare_pre_train_dict = {}
        dataset_compare_post_train_dict = {}
        dataset_compare_diff_dict = {}

    if args.grad_alignment or args.weight_delta:
        cos_sim_dict = {}
        grad_div_dict = {}

    if args.grad_diff:
        grad_diffs = []

    if args.weight_ll:
        modified_last_layer = []

    # log batch statistics for each client
    if args.log_batch_stats:
        batch_stats_dict = {}
        prev_stats_dict = {}
        for i in range(args.epochs):
            batch_stats_dict[i]= {}

    # **** TRAINING LOOPS STARTS HERE ****
    while epoch < args.epochs:
        local_losses = []
        local_weights = []

        if epoch in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500,
                     2000, 2500, 3000, 3500, 3999]:
            if args.grad_alignment:
                local_grads = []
                grad_eval = True
            if args.weight_delta:
                local_deltas = []
                delta_eval = True

        if epoch in [0, 1, 4, 5, 14, 15, 29, 30, 59, 60, 119, 120, 239, 240, 479, 480]:
            if args.grad_diff:
                prev_weight = copy.deepcopy(global_model).state_dict()
                local_deltas = []

        if epoch in [1, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999]:
            if args.dataset_compare:
                epoch_dataset_compare_pre_train_dict = {}
                epoch_dataset_compare_post_train_dict = {}
                epoch_dataset_compare_diff_dict = {}
                # pre training all model weights the same so only have to do this once
                acc_pre_individual_training = local_model.evaluate_client_model(idxs_clients, model=copy.deepcopy(global_model))
            if args.linear_probe:
                epoch_lp_pre_train_dict = {}
                epoch_lp_post_train_dict = {}
                # pre training all model weights the same so only have to do this once
                acc_pre_individual_training = local_model.evaluate_client_model(idxs_clients, model=copy.deepcopy(global_model))
                lp_acc_pre_individual_training = local_model.evaluate_client_model(idxs_clients, model=copy.deepcopy(global_model), lp=True)

        if args.log_batch_stats:
            batch_stats = []

        global_round = f'\n | Global Training Round : {epoch+1} |\n'
        print(global_round)

        global_model.train()
        m = max(int(args.frac * args.num_clients), 1)
        idxs_clients = np.random.choice(range(args.num_clients), m, replace=False)

        # for each selected client, init model weights with global weights and train lcl model for local_ep epochs
        pr = []
        for idx in idxs_clients:
            local_model = LocalUpdate(args=args, train_dataset=train_dataset, validation_dataset=validation_dataset,
                                      idx=idx, client_labels=client_labels[idx], all_client_data=user_groups, round=epoch)
            pr.append(local_model.client_data_proportions)

            # evaluate each client selected for this update round on the datasets of each other client selected for this
            # round (pre client training)
            if epoch in [1, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999]:
                if args.dataset_compare:
                    epoch_dataset_compare_pre_train_dict[idx] = acc_pre_individual_training
                if args.linear_probe:
                    epoch_lp_pre_train_dict[idx] = {}
                    epoch_lp_pre_train_dict[idx]['no_lp'] = acc_pre_individual_training
                    epoch_lp_pre_train_dict[idx]['lp'] = lp_acc_pre_individual_training


            # global weight update is the average of the client weights after training round
            w, loss, results = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            if args.log_batch_stats:
                batch_stats.append([idx, results])

            if args.weight_ll:
                modified_last_layer.append(results)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if args.grad_alignment and grad_eval:
                assert results is not None
                local_grads.append([results, idx])

            if args.weight_delta and delta_eval:
                assert results is not None
                local_deltas.append([results, idx])

            # evaluate each client selected for this update round on the datasets of each other client selected for this
            # round (post client training). For lp also do post with lp accs
            if epoch in [1, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999]:
                if args.dataset_compare:
                    epoch_dataset_compare_post_train_dict[idx] = local_model.evaluate_client_model(idxs_clients, model=local_model.get_model())
                    epoch_dataset_compare_diff_dict[idx] = [pre - post for pre, post in zip(epoch_dataset_compare_pre_train_dict[idx], epoch_dataset_compare_post_train_dict[idx])]
                if args.linear_probe:
                    epoch_lp_post_train_dict[idx] = {}
                    epoch_lp_post_train_dict[idx]['no_lp'] = local_model.evaluate_client_model(idxs_clients, model=local_model.get_model())
                    epoch_lp_post_train_dict[idx]['lp'] = local_model.evaluate_client_model(idxs_clients, model=local_model.get_model(), lp=True)

        # compute pairwise Wasswestien distance (distance metric between 2 prob distributions) between clients (and prev epochs of same client)
        if args.log_batch_stats:
            for i in range(len(batch_stats) - 1):
                idx_i = batch_stats[i][0]
                # a dict of k, v pairs of the batch stats for client i
                stats_i = batch_stats[i][1]

                for j in range(1, len(batch_stats)):
                    idx_j = batch_stats[j][0]
                    stats_j = batch_stats[j][1]

                    wasserstien_dict = {}
                    for k, v in stats_i.items():
                        wasserstien_dict[k] = scipy.stats.wasserstein_distance(np.asarray(stats_j[k].cpu()).ravel(), np.asarray(v.cpu()).ravel())

                    try:
                        batch_stats_dict[epoch][idx_i][idx_j] = wasserstien_dict
                    except KeyError:
                        batch_stats_dict[epoch][idx_i] = {}
                        batch_stats_dict[epoch][idx_i][idx_j] = wasserstien_dict

        # save comparison dict
        if epoch in [1, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999]:
            if args.dataset_compare:
                dataset_compare_pre_train_dict[epoch] = epoch_dataset_compare_pre_train_dict
                dataset_compare_post_train_dict[epoch] = epoch_dataset_compare_post_train_dict
                dataset_compare_diff_dict[epoch] = epoch_dataset_compare_diff_dict

            if args.linear_probe:
                lp_pre_train_dict[epoch] = epoch_lp_pre_train_dict
                lp_post_train_dict[epoch] = epoch_lp_post_train_dict

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # update global weights with the average of the obtained local weights
        global_weights = average_weights(local_weights)
        if args.weight_ll:
            global_weights = ll_modify(global_weights, modified_last_layer, args.num_classes, args.gamma)

        global_model.load_state_dict(global_weights)

        if args.grad_diff:
            if epoch in [1, 5, 15, 30, 60, 120, 240, 480]:
                entry = [epoch]
                for weight in local_weights:
                    entry.append(grad_diff(get_delta(prev_weight, weight), prev_grad))
                entry.append(grad_diff(get_delta(prev_weight, global_weights), prev_grad))
                grad_diffs.append(entry)
            if epoch in [0, 4, 14, 29, 59, 119, 239, 479]:
                prev_grad = get_delta(prev_weight, global_weights)

        if grad_eval or delta_eval:
            if args.grad_alignment:
                grad_eval = False
            if args.weight_delta:
                delta_eval = False
            cos_list = []
            if args.grad_alignment:
                for grad_idx_i in range(len(local_grads)):
                    grad_i = local_grads[grad_idx_i][0]
                    for grad_idx_j in range(grad_idx_i+1, len(local_grads)):
                        grad_j = local_grads[grad_idx_j][0]
                        cos_sim = grad_cos_sim(grad_i, grad_j)
                        cos_list.append([[local_grads[grad_idx_i][1], local_grads[grad_idx_j][1]], cos_sim])
                diversity = grad_diversity(local_grads)
            if args.weight_delta:
                for idx_i in range(len(local_deltas)):
                    delta_i = local_deltas[idx_i][0]
                    for idx_j in range(idx_i + 1, len(local_deltas)):
                        delta_j = local_deltas[idx_j][0]
                        cos_sim = grad_cos_sim(delta_i, delta_j)
                        cos_list.append([[local_deltas[idx_i][1], local_deltas[idx_j][1]], cos_sim])
                diversity = grad_diversity(local_deltas)

            cos_sim_dict[epoch] = cos_list
            grad_div_dict[epoch] = diversity

        if epoch % 50 == 0:
            # save model as a backup every 50 epochs
            model_path = f'/{run_root}/{args.wandb_run_name}/global_model.pt'
            torch.save(global_model.state_dict(), model_path)

        # Test global model inference on validation set after each round use model save criteria
        if args.dataset == 'celeba':
            val_acc, per_feature_val_acc, val_loss = celeba_inference(global_model, validation_dataset_global,
                                                                      args.device, args.num_workers)
            print(f'Epoch {epoch} Validation Accuracy {val_acc * 100}% \nValidation Loss {val_loss} '
                  f'\nTraining Loss (average loss of clients evaluated on their own in distribution validation set): {loss_avg}'
                  f'Per Feature Validation Accuracy {per_feature_val_acc * 100}')
        else:
            val_acc, val_loss = validation_inference(args, global_model, validation_dataset_global, args.num_workers)
            print(f'Epoch {epoch} Validation Accuracy {val_acc * 100}% \nValidation Loss {val_loss} '
                  f'\nTraining Loss (average loss of clients evaluated on their own in distribution validation set): {loss_avg}')

        exit()
        if args.epochs - (epoch + 1) <= 100:
            last_hundred_val_loss.append(val_loss)
            last_hundred_val_acc.append(val_acc)
            if args.dataset == 'celeba':
                test_acc, per_feature_test_acc, test_loss = celeba_inference(global_model, test_dataset, args.device,
                                                                             args.num_workers)
            else:
                test_acc, test_loss = test_inference(args, global_model, test_dataset, args.num_workers)
            last_hundred_test_loss.append(test_loss)
            last_hundred_test_acc.append(test_acc)

        # print global training loss after every 'i' rounds
        if (epoch + 1) % args.print_every == 0:
            if args.wandb:
                wandb.log({f'val_acc': val_acc,
                           f'val_loss': val_loss,
                           f'train_loss': loss_avg
                           #f'global model test accuarcy': test_acc,
                           #f'global model test loss': test_loss
                           }, step=epoch)

        epoch += 1

    # save batch statistics
    if args.log_batch_stats:
        batch_stats_path = f'/{run_root}/{args.wandb_run_name}/batch_stats.pt'
        torch.save(batch_stats_dict, batch_stats_path)

    # save dataset comparison dict
    if args.dataset_compare:
        comparisons_to_save_pre = f'/{run_root}/{args.wandb_run_name}/dataset_comparisons_pre.pt'
        comparisons_to_save_post = f'/{run_root}/{args.wandb_run_name}/dataset_comparisons_post.pt'
        comparisons_to_save_diff = f'/{run_root}/{args.wandb_run_name}/dataset_comparisons_diff.pt'
        torch.save(dataset_compare_pre_train_dict, comparisons_to_save_pre)
        torch.save(dataset_compare_post_train_dict, comparisons_to_save_post)
        torch.save(dataset_compare_diff_dict, comparisons_to_save_diff)

    # save linear probe dict
    if args.linear_probe:
        comparisons_to_save_pre = f'/{run_root}/{args.wandb_run_name}/lp_pre.pt'
        comparisons_to_save_post = f'/{run_root}/{args.wandb_run_name}/lp_post.pt'
        torch.save(lp_pre_train_dict, comparisons_to_save_pre)
        torch.save(lp_post_train_dict, comparisons_to_save_post)

    # save grad alignment dict
    if args.grad_alignment or args.weight_delta:
        if args.grad_alignment:
            cos_sim_path = f'/{run_root}/{args.wandb_run_name}/grad_cosine_similarity.pt'
            grad_div_path = f'/{run_root}/{args.wandb_run_name}/grad_diversity.pt'
        else:
            cos_sim_path = f'/{run_root}/{args.wandb_run_name}/delta_cosine_similarity.pt'
            grad_div_path = f'/{run_root}/{args.wandb_run_name}/delta_diversity.pt'

        torch.save(cos_sim_dict, cos_sim_path)
        torch.save(grad_div_dict, grad_div_path)

    # save final model after training
    model_path = f'/{run_root}/{args.wandb_run_name}/global_model.pt'
    torch.save(global_model.state_dict(), model_path)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset, args.num_workers)

    # wsm_inference after completion of training
    wsm_acc, wsm_loss = wsm_inference(args, global_model, validation_dataset, user_groups, args.num_workers)

    # last 100 avg acc and loss
    last_hundred_test_loss = sum(last_hundred_test_loss) / len(last_hundred_test_loss)
    last_hundred_test_acc = sum(last_hundred_test_acc) / len(last_hundred_test_acc)
    last_hundred_val_loss = sum(last_hundred_val_loss) / len(last_hundred_val_loss)
    last_hundred_val_acc = sum(last_hundred_val_acc) / len(last_hundred_val_acc)

    if args.wandb:
        wandb.log({f'val_acc': val_acc,
                   f'test_acc': test_acc,
                   f'last_100_val_acc': last_hundred_val_acc,
                   f'last_100_test_acc': last_hundred_test_acc
                   })

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- WSM Accuracy: {:.2f}%".format(100 * wsm_acc))
    print("|---- Validation Accuracy: {:.2f}%".format(100 * val_acc))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    if args.dataset == 'celeba':
        print(f"|---- Per Feature Validation Accuracy {per_feature_val_acc * 100}")
    print("|---- Last 100 Validation Accuracy: {:.2f}%".format(100 * last_hundred_val_acc))
    print("|---- Last 100 Test Accuracy: {:.2f}%".format(100 * last_hundred_test_acc))
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))