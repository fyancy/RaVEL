"""
 WaveletKernelNet: An interpretable deep neural network for industrial intelligent diagnosis,
IEEE Trans. Syst. Man Cybern.: Syst. 52 (4) (2021) 2302–2312.
1) https://github.com/HazeDT/WaveletKernelNet (Origianl codes)
2) https://github.com/liguge/DLWCB/blob/main/laplacewave.py (Corrected codes)
"""


import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F


def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y


class Laplace_fast(nn.Module):
    name = "LaplaceLayer"

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super().__init__()

        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % in_channels
            raise ValueError(msg)

        self.out_channels = out_channels
        # self.kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size

        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels).view(-1, 1))
        # self.a_ = nn.Parameter(torch.Tensor(out_channels, 1))
        # self.b_ = nn.Parameter(torch.Tensor(out_channels, 1))
        # self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)
        # print("查看Parameter属性")
        # todo: 注意源代码中的参数定义方式是不会更新参数的，要将view放在 Parameter 之内
        # self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, waveforms):
        dev = waveforms.device
        time_disc = torch.linspace(0, 1, steps=int(self.kernel_size)).to(dev)
        p1 = (time_disc - self.b_.to(dev)) / (self.a_.to(dev) + 1e-8)  # improved version: better
        # p1 = time_disc - self.b_ / (self.a_.cuda()+1e-8)  # original codes in WaveletKernelNet, WKN
        laplace_filter = Laplace(p1)
        filters = laplace_filter.view(self.out_channels, 1, self.kernel_size).to(dev)

        return F.conv1d(waveforms, filters, stride=1, padding=1)


def Mexh(p):
    # p = 0.04 * p  # 将时间转化为在[-5,5]这个区间内
    y = (1 - torch.pow(p, 2)) * torch.exp(-torch.pow(p, 2) / 2)

    return y


class Mexh_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1):
        super().__init__()

        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels).view(-1, 1))

    def forward(self, waveforms):
        dev = waveforms.device
        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))
        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.to(dev) - self.b_.to(dev) / (self.a_.to(dev) + 1e-8)
        p2 = time_disc_left.to(dev) - self.b_.to(dev) / (self.a_.to(dev) + 1e-8)

        Mexh_right = Mexh(p1)
        Mexh_left = Mexh(p2)

        Mexh_filter = torch.cat([Mexh_left, Mexh_right], dim=1)  # 40x1x250
        self.filters = Mexh_filter.view(self.out_channels, 1, self.kernel_size).to(dev)

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


def Morlet(p):
    C = pow(pi, 0.25)
    # p = 0.03 * p
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
    return y


class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super().__init__()

        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels).view(-1, 1))

    def forward(self, waveforms):
        dev = waveforms.device
        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))
        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.to(dev) - self.b_.to(dev) / (self.a_.to(dev) + 1e-8)
        p2 = time_disc_left.to(dev) - self.b_.to(dev) / (self.a_.to(dev) + 1e-8)

        Morlet_right = Morlet(p1)
        Morlet_left = Morlet(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).to(dev)

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


def Sine(p):
    y = torch.sin(p)
    return y


class Sine_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super().__init__()

        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels).view(-1, 1))

    def forward(self, waveforms):
        dev = waveforms.device
        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))
        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.to(dev) - self.b_.to(dev) / (self.a_.to(dev) + 1e-8)
        p2 = time_disc_left.to(dev) - self.b_.to(dev) / (self.a_.to(dev) + 1e-8)

        Morlet_right = Sine(p1)
        Morlet_left = Sine(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).to(dev)

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


# ----------------- A modified Wide-Kernel CNN2D
def dropout_layer(dropout=True, p=0.1):
    return nn.Dropout(p) if dropout else nn.Identity()


def bn_layer(bn=True, track_stats=True):
    return nn.LazyBatchNorm1d(momentum=0.1, affine=True, track_running_stats=track_stats) \
        if bn else nn.Identity()


class MyWKN(nn.Module):
    name = 'MyWKN'

    def __init__(self, num_classes, use_laplace=True, bn=False, dropout=False, drop_rate=0.1):
        super().__init__()
        conv_kernels = [64, 16, 4]
        conv_kernels = [k - 1 for k in conv_kernels]
        num_channels = (16, 32, 64)
        pool_size = [2, 2, 2]
        self.use_laplace = use_laplace

        if use_laplace:
            self.conv1 = Laplace_fast(32, 16, in_channels=1)
        else:
            self.conv1 = nn.LazyConv1d(num_channels[0], conv_kernels[0], 1,
                                       (conv_kernels[0] - 1) // 2, padding_mode='circular')

        self.conv_layers = nn.Sequential(
            bn_layer(bn),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
            nn.MaxPool1d(pool_size[0]),

            nn.LazyConv1d(num_channels[1], conv_kernels[1], 1, (conv_kernels[1] - 1) // 2, padding_mode='circular'),
            bn_layer(bn),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
            nn.MaxPool1d(pool_size[1]),

            nn.LazyConv1d(num_channels[2], conv_kernels[2], 1, (conv_kernels[2] - 1) // 2, padding_mode='circular'),
            bn_layer(bn),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
            nn.MaxPool1d(pool_size[2]),

            nn.Flatten(),
            nn.LazyLinear(128),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
        )
        self.classifier = nn.LazyLinear(num_classes)

    def control_bn(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.BatchNorm1d):
                layer.training = training

    def control_dropout(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.Dropout):
                layer.training = training

    def forward(self, x):
        features = self.conv_layers(self.conv1(x))
        logits = self.classifier(features)
        return logits


class SimpleCNN(nn.Module):
    name = 'SimpleCNN'

    def __init__(self, num_classes, bn=False, dropout=False, drop_rate=0.1):
        super().__init__()
        conv_kernels = [4, 4, 4]
        conv_kernels = [k - 1 for k in conv_kernels]
        num_channels = (32, 32, 32)

        # self.conv1 = nn.LazyConv1d(num_channels[0], conv_kernels[0], 1,
        #                            (conv_kernels[0] - 1) // 2, padding_mode='circular')
        self.conv1 = nn.Identity()
        act_fn = nn.ReLU6()

        self.conv_layers = nn.Sequential(
            # bn_layer(bn),
            # dropout_layer(dropout, drop_rate),
            # act_fn,
            # nn.MaxPool1d(pool_size[0]),

            # nn.LazyConv1d(num_channels[1], conv_kernels[1], 2, (conv_kernels[1] - 1) // 2),
            # bn_layer(bn),
            # dropout_layer(dropout, drop_rate),
            # act_fn,
            # nn.MaxPool1d(pool_size[1]),

            # nn.LazyConv1d(num_channels[2], conv_kernels[2], 2, (conv_kernels[2] - 1) // 2),
            # bn_layer(bn),
            # dropout_layer(dropout, drop_rate),
            # act_fn,
            # nn.MaxPool1d(pool_size[2]),

            nn.LazyConv1d(num_channels[2], conv_kernels[2], 2, (conv_kernels[2] - 1) // 2),
            bn_layer(bn),
            dropout_layer(dropout, drop_rate),
            act_fn,
            # nn.MaxPool1d(pool_size[2]),

            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.LazyBatchNorm1d(),
            act_fn,

            nn.LazyLinear(512),
            nn.LazyBatchNorm1d(),
            act_fn,

            nn.LazyLinear(256),
            nn.LazyBatchNorm1d(),
            act_fn,
        )
        self.classifier = nn.Sequential(nn.LazyLinear(128),  act_fn,
                                        nn.LazyLinear(num_classes))

    def control_bn(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.BatchNorm1d):
                layer.training = training

    def control_dropout(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.Dropout):
                layer.training = training

    def forward(self, x):
        features = self.conv_layers(self.conv1(x))
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    a = torch.ones([32, 1, 2048]).cuda()
    layer = Laplace_fast(1, 31).cuda()
    print(layer(a).shape)

    model = MyWKN(4).cuda()
    weight = model.conv1
    print(weight.a_.device, weight.b_.device)
    # print(weight.bias.device)
    # print(weight.a_.data, weight.b_.data)
    # for n, p in model.named_parameters():
    #     print(n, p)

    # for n, p in model.named_children():
    #     print(n, p)
    # print(model.device, model.conv_layers.device)
    c = model(a)
    print(c.shape)
