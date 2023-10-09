
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inCh, outCh, H, W, layer_num, stride, norm):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inCh, out_channels=outCh,
                      kernel_size=3, padding=1, stride=stride),
            #nn.BatchNorm2d(outCh),
            norm2d(outCh, H, W, layer_num, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outCh, out_channels=outCh,
                      kernel_size=3, padding=1, stride=1),
            norm2d(outCh, H, W, layer_num, norm)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inCh != outCh:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inCh, out_channels=outCh,
                          kernel_size=1, stride=stride),
                norm2d(outCh, H, W, layer_num, norm)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def norm2d(outCh, H, W, layer_num, norm):
    if norm == 'group_norm':
        return nn.GroupNorm(2, outCh, affine=True)
    elif norm == 'batch_norm':
        return nn.BatchNorm2d(outCh)
    elif norm == 'layer_norm':
        if layer_num == 2:
            H = int(H/2)
            W = int(W/2)
        elif layer_num == 3:
            H = int(H/4)
            W = int(W/4)
        elif layer_num == 4:
            H = int(H/8)
            W = int(W/8)
        return nn.LayerNorm((outCh, H, W))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, H, W, stride, norm):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm2d(out_planes, H, W, norm)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = norm2d(out_planes, H, W, norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*in_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*in_planes,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(self.expansion*out_planes, H, W, norm)
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


