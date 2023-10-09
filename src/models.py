#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from residualBlock import ResidualBlock, norm2d, BasicBlock, Bottleneck
import torchvision.models as models

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNN_A(nn.Module):
    def __init__(self, args):
        super(CNN_A, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNN_B(nn.Module):
    def __init__(self, args):
        super(CNN_B, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, args.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNN_C(nn.Module):
    def __init__(self, args):
        super(CNN_C, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 64, 5)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(4)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x, freeze=False):
        x = self.lrn(self.pool(F.relu(self.conv1(x))))
        x = self.pool(self.lrn(F.relu(self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        fc3_input = F.relu(self.fc2(x))
        x = self.fc3(fc3_input)
        if freeze:
            return fc3_input
        else:
            return x


class CNNLeNet(nn.Module):
    def __init__(self, args):
        super(CNNLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=args.num_channels, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=(2, 2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


'''class CNNResNet18(nn.Module):
    def __init__(self, args):
        super(CNNResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=args.num_channels, out_channels=64, kernel_size=3, padding=1, stride=1),
            norm2d(64, self.H, self.W, 0, args.norm),
            nn.ReLU(inplace=True)
        )
        self.layer_1 = self.make_layer(ResidualBlock, 64, 64, self.H, self.W, stride=1, norm=args.norm)
        self.layer_2 = self.make_layer(ResidualBlock, 64, 128, self.H, self.W, stride=2, norm=args.norm)
        self.layer_3 = self.make_layer(ResidualBlock, 128, 256, self.H, self.W, stride=2, norm=args.norm)
        self.layer_4 = self.make_layer(ResidualBlock, 256, 512, self.H, self.W, stride=2, norm=args.norm)
        self.avgpool = nn.AvgPool2d((3, 3), stride=2)
        self.fc = nn.Linear(512 * 1 * 1, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        fc_input = x.view(-1, 512 * 1 * 1)
        x = self.fc(fc_input)

        return x

    def make_layer(self, block, inCh, outCh, H, W, stride, norm, block_num=2):
        layers = []
        layers.append(block(inCh, outCh, H, W, stride, norm))
        for i in range(block_num - 1):
            layers.append(block(outCh, outCh, H, W, 1, norm))
        return nn.Sequential(*layers)'''

class ResNet(nn.Module):
    def __init__(self, args, block_nums):
        super(ResNet, self).__init__()
        self.width = args.width
        self.block_nums = block_nums
        self.channels = [32, 64, 128, 256]
        self.channels = [i * self.width for i in self.channels]
        self.H = args.img_dim[0]
        self.W = args.img_dim[1]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=args.num_channels, out_channels=self.channels[0], kernel_size=3, padding=1, stride=1),
            norm2d(self.channels[0], self.H, self.W, 0, args.norm),
            nn.ReLU(inplace=True)
        )
        self.layer_1 = self.make_layer(ResidualBlock, self.channels[0], self.channels[0], self.H, self.W, stride=1,
                                       norm=args.norm, layer_num=1, block_num=self.block_nums[0])
        self.layer_2 = self.make_layer(ResidualBlock, self.channels[0], self.channels[1], self.H, self.W, stride=2,
                                       norm=args.norm, layer_num=2, block_num=self.block_nums[1])
        self.layer_3 = self.make_layer(ResidualBlock, self.channels[1], self.channels[2], self.H, self.W, stride=2,
                                       norm=args.norm, layer_num=3, block_num=self.block_nums[2])
        self.layer_4 = self.make_layer(ResidualBlock, self.channels[2], self.channels[3], self.H, self.W, stride=2,
                                       norm=args.norm, layer_num=4, block_num=self.block_nums[3])
        self.avgpool = nn.AvgPool2d((3, 3), stride=2)
        self.fc = nn.Linear(self.channels[3] * 1 * 1, args.num_classes)

    def forward(self, x):
        fc_width = 256 * self.width
        x = self.conv1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        fc_input = x.view(-1, fc_width * 1 * 1)
        x = self.fc(fc_input)
        return x

    def make_layer(self, block, inCh, outCh, H, W, stride, norm, layer_num, block_num=2):
        layers = []
        layers.append(block(inCh, outCh, H, W, layer_num, stride, norm))
        for i in range(block_num - 1):
            layers.append(block(outCh, outCh, H, W, layer_num, 1, norm))
        return nn.Sequential(*layers)

def ResNet18(args):
    return ResNet(args, [2, 2, 2, 2])


def ResNet34(args):
    return ResNet(args, [3, 4, 6, 3])

def get_model(name):
    return  {'cnn_a': CNN_A,
             'cnn_b': CNN_B,
             'cnn_c': CNN_C,
             'lenet': CNNLeNet,
             'resnet18': ResNet18,
             'resnet34': ResNet34}[name]

'''# General ResNet class from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class RawResNet(nn.Module):
    def __init__(self, args, block, num_blocks):
        super(RawResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm2d(64, H, W, args.norm),
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=args.norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=args.norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=args.norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=args.norm)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out'''

'''def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])'''
