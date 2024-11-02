import torch.nn as nn
import torch
import torch.nn.functional as F

from .wkn import Laplace_fast


class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv1d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv1d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv1d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class LaplaceInception(nn.Module):
    name = "LaplaceInception"

    def __init__(self, num_classes=10, use_laplace=True, in_channel=1):
        super().__init__()

        self.use_laplace = use_laplace
        if use_laplace:
            self.conv1 = Laplace_fast(32, 16, in_channels=in_channel)
        else:
            self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)

        self.bn = nn.BatchNorm1d(32, eps=0.001)
        self.Conv1d_2a_3x3 = BasicConv1d(32, 32, kernel_size=3)
        self.Conv1d_2b_3x3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.Conv1d_3b_1x1 = BasicConv1d(64, 80, kernel_size=1)
        self.Conv1d_4a_3x3 = BasicConv1d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        # 299 x 299 x 3
        # print("x0:",x.size())
        x = self.conv1(x)
        x = self.bn(x)
        # print("x1:",x.size())
        # 64, 32, 511
        x = self.Conv1d_2a_3x3(x)
        # print("x2:",x.size())
        # 147 x 147 x 32
        x = self.Conv1d_2b_3x3(x)
        # print("x3:",x.size())
        # 147 x 147 x 64
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        # print("x4:",x.size())
        # 73 x 73 x 64
        x = self.Conv1d_3b_1x1(x)
        # print("x5:",x.size())
        # 73 x 73 x 80
        x = self.Conv1d_4a_3x3(x)
        # print("x6:",x.size())
        # 71 x 71 x 192
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        # print("x7:",x.size())
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # print("x8:",x.size())
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # print("x9:",x.size())
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # print("x10:",x.size())
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # print("x11:",x.size())
        # 17 x 17 x768
        x = nn.AdaptiveMaxPool1d(1)(x)
        # print("x12:",x.size())
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # print("x13:",x.size())
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # print("x14:",x.size())
        # 2048

        x = self.fc(x)
        # print("x15:",x.size())
        return x


if __name__ == "__main__":
    a = torch.ones([32, 1, 2048]).cuda()
    model = LaplaceInception(num_classes=7).cuda()
    b = model(a)
    print(b.shape)

