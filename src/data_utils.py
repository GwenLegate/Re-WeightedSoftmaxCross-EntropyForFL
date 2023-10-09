from re import I
import torch
import torch.nn as nn
import numpy as np
import logging

class FedSampler(torch.utils.data.Sampler):
    def __init__(self, client_groups, training=True):
        self.client_groups = client_groups
        self.active_clients = None
        self.epoch = None
        self.training=training

        self.inv_mapping = {}
        for id, client in client_groups.items():
            for idx in client['train' if training else 'validation']:
                self.inv_mapping[idx] = id


    def set_clients(self, active_clients):
        self.active_clients = active_clients

        np.random.seed(self.epoch)
        key = 'train' if self.training else 'validation'
        self._indices = []
        for client in active_clients:
            client_idx = self.client_groups[client][key]
            np.random.shuffle(client_idx) # inpalce
            self._indices += [client_idx]

        self._indices = torch.from_numpy(np.stack(self._indices))

    def set_epoch(self, epoch):
        # fixing the seed of the sampling
        self.epoch = epoch

    def __iter__(self):

        lens = [len(x) for x in self._indices]
        min_qty, max_qty = min(lens), max(lens)
        if min_qty != max_qty: 
            logging.warning('clients do not have all the same number of examples. got' + 
            f'{min_qty} to {max_qty} examples, will therefore restrict each client' + 
            f'to {min_qty} samples'
            )

        total = 0
        for idx in range(min_qty):
            for client_idx, client in enumerate(self.active_clients):
                out = self._indices[client_idx][idx]
                yield out
                total += 1 

    @property
    def num_clients(self):
        return 0 if self.active_clients is None else len(self.active_clients)

    def __len__(self):
        return min(len(x) for x in self._indices) * self.num_clients

    def client_split(self, batch_output):
        idxs_, (xs_, ys_) = torch.utils.data.default_collate(batch_output)

        # rearrange in clients
        idxs = torch.stack([idxs_[i::self.num_clients] for i in range(self.num_clients)])
        xs   = torch.stack([xs_[i::self.num_clients] for i in range(self.num_clients)])
        ys   = torch.stack([ys_[i::self.num_clients] for i in range(self.num_clients)])

        # Making sure we are feeding the appropriate data
        for client_idx, idx_list in enumerate(idxs):
            client_id = self.active_clients[client_idx]
            for idx in idx_list:
                assert self.inv_mapping[idx.item()] == client_id

        return xs, ys


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.ds = dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return idx, self.ds[idx]

def dataset_config(args):
    '''
    sets dependent args based on selected dataset if the required args are different from the default
    Args:
        args: set of args passed to arg parser
    '''
    if args.dataset == 'cifar100':
        args.num_classes = 100
        args.img_dim = [32, 32]
    if args.dataset == 'femnist':
        args.num_classes = 62
        args.num_channels = 1
        args.img_dim = [28, 28]
    if args.dataset == 'imagenet32':
        args.num_classes = 1000
        args.img_dim = [32, 32]
    if args.dataset == 'celeba':
        args.num_classes = 40
        args.img_dim = [178, 218]

def get_num_samples_per_label(self, dataset_labels):
    labels = np.array(dataset_labels)
    examples_per_label = []
    for i in range(self.args.num_classes):
        examples_per_label.append(int(np.argwhere(labels == i).shape[0]))
    return examples_per_label