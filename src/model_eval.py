#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from options import args_parser
from eval_utils import test_model
from data_utils import dataset_config


if __name__ == '__main__':
    args = args_parser()
    # **** manually set appropriate dataset *******
    args.dataset = 'cifar10'
    # set args dependent on dataset
    dataset_config(args)
    user_groups = torch.load('./models/user_groups.pt')


    #test_acc, wsm_acc = test_model(args, './models/global_model.pt', wsm=True)
    test_acc, wsm_acc = test_model(args, './models/global_model.pt', wsm=False), 0
    print(f"|---- Test Accuracy: {100 * test_acc}%")
    print(f"|---- WSM Accuracy: {100 * wsm_acc}%")