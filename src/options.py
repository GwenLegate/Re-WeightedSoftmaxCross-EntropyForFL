#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import warnings

def args_parser():
    parser = argparse.ArgumentParser()
    # wandb arguments
    parser.add_argument('--wandb', type=bool, default=False, help='enables wandb logging and disables local logfiles')
    parser.add_argument("--wandb_project", type=str, default='', help='specifies wandb project to log to')
    parser.add_argument("--wandb_entity", type=str, default='',
                        help='specifies wandb username to team name where the project resides')
    parser.add_argument("--wandb_run_name", type=str,
                        help="set run name to differentiate runs, if you don't set this wandb will auto generate one")
    parser.add_argument("--offline", type=bool, default=False, help="set wandb to run in offline mode")

    # Tested on CC, in practice num_workers should equal num cpus
    parser.add_argument('--num_workers', type=int, default=0, help="how many subprocesses to use for data loading.")

    # federated arguments (Notation for the arguments followed from Mcmahan(2017))
    parser.add_argument('--epochs', type=int, default=4000, help="number of rounds of training")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_iters', type=int, default=None,
                        help="if set, stops training after local_iters mini-batchs of training")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--global_lr', type=float, default=1,
                        help='learning rate for global model, always 1 for FedAvg version')
    parser.add_argument('--client_lr', type=float, default=0.1, help='learning rate for client models')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--momentum', type=float, default=0.97, help='SGD momentum, momentum parameter has no effect on'
                                                                     '\FedAvg, needs to be set to >0 for FedAvgM. '
                                                                     'default is 0.97 ')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model name, options: mlp, cnn_a, cnn_b, cnn_c, lenet, resnet18, resnet34')
    parser.add_argument('--width', type=int, default=2, help='model width factor')
    parser.add_argument('--mask', type=bool, default=False, help='enables logit masking for clients (new soft mask)')
    #parser.add_argument('--freeze', type=bool, default=False, help='freezes weights for indices before softmax for all indices outside the client distribution')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='group_norm', help="batch_norm, group_norm, layer_norm")
    parser.add_argument('--log_batch_stats', type=bool, default=False, help="keep log of batch statistics per client")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--k', type=int, default=1, help='factor to widen resnet')
    # other arguments
    parser.add_argument('--weight_ll', type=bool, default=False,
                        help="weight last layer of model based on client labels")
    parser.add_argument('--gamma', type=float, default=0., help="hyperparameter for last layer scaling weights between 0 and 1")
    parser.add_argument('--linear_probe', type=bool, default=False, help="evaluate with linear probe")
    parser.add_argument('--grad_alignment', type=bool, default=False, help="checks gradient alignment across clients")
    parser.add_argument('--grad_diff', type=bool, default=False,
                        help="calculates the norm of the difference between grads before and after training")
    parser.add_argument('--weight_delta', type=bool, default=False,
                        help="checks alignment of weight delta across clients")
    parser.add_argument('--dataset_compare', type=bool, default=False,
                        help="at regular intervals, evaluate each client model against the datasets of each other "
                             "randomly selected client for that update round")
    parser.add_argument('--continue_train', type=str, default='', help="path to model to load to continue training")
    parser.add_argument('--hyperparam_search', type=bool, default=False,
                        help="sets random values within a specified range for a hyper parameter search")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="name of dataset. mnist, fmnist, cifar10, cifar100")
    parser.add_argument('--img_dim', type=list, default=[32, 32], help="H, W of input images")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer (adam or sgd)")
    parser.add_argument('--decay', type=int, default=0,
                        help="Use learning rate decay. 1->use 0->don't use. Default = 0.")
    parser.add_argument('--iid', type=int, default=0, help='Default set to non-IID. Set to 1 for IID.')
    parser.add_argument('--dirichlet', type=int, default=1,
                        help='1 uses a dirichlet distribution to create non-iid data, 0 uses shards according to \
                        Mcmahan(2017) et. al. Default = 1.')
    parser.add_argument('--alpha', type=float, default=0.1, help="alpha of dirichlet, value between 0 and infinity\
                        more homogeneous when higher, more heterogeneous when lower")
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    return args


def validate_args(args):
    # ensure gamma for last layer weight scaling is between 0 and 1
    if args.gamma > 1 or args.gamma < 0:
        raise ValueError(f'gamma for last layer weight scaling is set to {args.gamma}, gamma must be a value between 0 and 1')

    # set normalization parameter
    if args.model == "resnet18" and args.norm == 'batch_norm':
        warnings.warn('Using batch norm with resnet18, should this be group norm?')

    # check number of classes and number of input channels matches dataset
    if args.dataset == 'cifar100' and args.num_classes != 100:
        raise ValueError(
            f'number of classes is set to {args.num_classes}, needs to be 100 for cifar100 dataset'
        )
    if args.dataset in ['cifar10', 'fmnist', 'mnist'] and args.num_classes != 10:
        raise ValueError(
            f'number of classes is set to {args.num_classes}, needs to be 10 for {args.dataset} dataset'
        )
    if args.dataset == 'femnist' and args.num_classes != 62:
        raise ValueError(
            f'number of classes is set to {args.num_classes}, needs to be 62 for {args.dataset} dataset'
        )
    if args.dataset == 'celeba' and args.num_classes != 40:
        raise ValueError(
            f'number of classes is set to {args.num_classes}, needs to be 40 for CelebA dataset'
        )
    if args.dataset == 'cifar100' or args.dataset == 'cifar10' or args.dataset == 'celeba':
        if args.num_channels != 3:
            raise ValueError(
                f'number of input channels is set to {args.num_channels}, needs to be 3 for {args.dataset} dataset'
            )
    if args.dataset == 'mnist' or args.dataset == 'fmnist' or args.dataset == 'femnist':
        if args.num_channels != 1:
            raise ValueError(
                 f'number of input channels is set to {args.num_channels}, needs to be 1 for {args.dataset} dataset'
            )
    if args.log_batch_stats:
        if args.norm != 'batch_norm':
            raise ValueError(
                 f'norm must be set to batch norm in order to log batch stats'
            )
 
