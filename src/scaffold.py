import wandb
from models import ResNet18, ResNet34
from utils import wandb_setup, zero_last_hundred, split_dataset, get_dataset
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

def my_train_net_scaffold(args, net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, proportions, client_labels):
    print('Training network %s' % str(net_id))
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.client_lr,
                                     weight_decay=args.reg)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.client_lr,
                                    momentum=args.momentum, weight_decay=args.reg)

    if args.mask:
        criterion = nn.NLLLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(args.local_ep):
        #epoch_loss_collector = []
        for tmp in train_dataloader:
            x, target = tmp[0].to(device), tmp[1].to(device)

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

            net_para = net.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - args.client_lr * (c_global_para[key] - c_local_para[key])
            net.load_state_dict(net_para)

            cnt += 1
            #epoch_loss_collector.append(loss.item())

        #epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.client_lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    #train_acc = compute_accuracy(net, train_dataloader, device=args.device)
    #test_acc = compute_accuracy(net, test_dataloader, device=args.device)

    #return train_acc, test_acc, c_delta_para
    return c_delta_para

def my_local_train_net_scaffold(args, nets, selected, global_model, c_nets, c_global, client_datasets, train_data,
                                validation_data):
    #avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        # move the model to cuda device:
        net.to(args.device)
        c_nets[net_id].to(device)

        train_idxs = client_datasets[net_id]['train']
        validation_idxs = client_datasets[net_id]['validation']
        # label proportions in dataset
        proportions = get_client_labels(train_data, user_groups, args.num_workers, args.num_classes, proportions=True)
        client_labels = get_client_labels(train_data, user_groups, args.num_workers, args.num_classes)
        trainloader, testloader = train_test(args, train_data, validation_data, list(train_idxs), list(validation_idxs),
                                             args.num_workers)
        '''trainacc, testacc, c_delta_para = my_train_net_scaffold(args, net_id, net, global_model, c_nets[net_id],
                                                                c_global, trainloader, testloader, proportions[net_id],
                                                                client_labels[net_id])'''
        c_delta_para = my_train_net_scaffold(args, net_id, net, global_model, c_nets[net_id], c_global, trainloader,
                                             testloader, proportions[net_id],  client_labels[net_id])

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        #avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.num_clients
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    #avg_acc /= len(selected)
    nets_list = list(nets.values())
    return nets_list

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
                                   list(all_client_data[idx]['validation']), args.num_workers)
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
    for i in range(args.num_clients):
        if i == 0:
            idxs_val = user_groups[i]['validation']
        else:
            idxs_val = np.concatenate((idxs_val, user_groups[i]['validation']), axis=0)

    validation_dataset_global = DatasetSplit(validation_dataset, idxs_val)

    if args.dataset_compare:
        avg_forgetting_dict = {}

    print("Initializing nets (scaffold)")

    nets = init_nets(args.num_clients, args)
    global_models = init_nets(1, args)
    global_model = global_models[0]

    c_nets = init_nets(args.num_clients, args)
    c_globals = init_nets(1, args)
    c_global = c_globals[0]
    c_global_para = c_global.state_dict()
    for net_id, net in c_nets.items():
        net.load_state_dict(c_global_para)

    global_para = global_model.state_dict()
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

        my_local_train_net_scaffold(args, nets, selected, global_model, c_nets, c_global, user_groups, train_dataset,
                                    validation_dataset)
        # update global model
        total_data_points = sum([len(user_groups[r]['train']) for r in selected])
        fed_avg_freqs = [len(user_groups[r]['train']) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_para = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_para)
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
