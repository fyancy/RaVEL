import torch
import torch.nn as nn
from .wkn import Laplace_fast


# ------------------- [1] Wavelet Kernel Net -----------------


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    name = "LaplaceRes18"

    def __init__(self, block, layers, laplace: bool, num_classes=10, in_channel=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_laplace = laplace
        if laplace:
            self.conv1 = Laplace_fast(64, 16)
        else:
            self.conv1 = nn.Conv1d(in_channel, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def Laplace_ResNet(num_classes, use_laplace):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], laplace=use_laplace, num_classes=num_classes)

    return model


# ------------------- [2] 1D DCNN-MLC -----------------
# [2] An Intelligent Compound Fault Diagnosis Method Using One-Dimensional Deep Convolutional Neural
# Network With Multi-Label Classifier


class CNN1D(nn.Module):
    def __init__(self, num_classes, output_neurons, multilabel=False):
        super().__init__()

        self.backbone = self.build_backbone()
        self.classifier_sc = self.build_classifier(num_classes)
        self.classifier_ml = self.build_classifier(output_neurons)
        self.classifier = self.classifier_ml if multilabel else self.classifier_sc

    def build_backbone(self):
        backbone = nn.Sequential(nn.LazyConv1d(256, 64, 16, 32), nn.LazyBatchNorm1d(), nn.ReLU(),
                                 nn.LazyConv1d(256, 3, 1, 1), nn.LazyBatchNorm1d(), nn.ReLU(),
                                 nn.AvgPool1d(2),
                                 nn.LazyConv1d(128, 3, 1, 1), nn.LazyBatchNorm1d(), nn.ReLU(),
                                 nn.MaxPool1d(2),
                                 nn.Flatten(),
                                 )
        return backbone

    def build_classifier(self, num_neurons):
        return nn.Sequential(nn.LazyLinear(128), nn.ReLU(),
                             nn.LazyLinear(64), nn.ReLU(),
                             nn.LazyLinear(num_neurons)
                             )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ----------------- A modified Wide-Kernel CNN2D
def dropout_layer(dropout=True, p=0.1):
    return nn.Dropout(p) if dropout else nn.Identity()


def bn1d_layer(bn=True, track_stats=True):
    return nn.LazyBatchNorm1d(momentum=0.1, affine=True, track_running_stats=track_stats) \
        if bn else nn.Identity()


class CNN1Dv2(nn.Module):
    name = 'DCNNv2'

    def __init__(self, num_classes, output_neurons=None, multilabel=False,
                 bn=False, dropout=False, drop_rate=0.1):
        super().__init__()
        conv_kernels = [64, 32, 16, 4]
        conv_kernels = [k - 1 for k in conv_kernels]
        num_channels = (16, 32, 32, 64)
        pool_size = [4, 2, 2, 2]

        self.conv_layers = nn.Sequential(
            nn.LazyConv1d(num_channels[0], conv_kernels[0], 1, (conv_kernels[0] - 1) // 2),
            bn1d_layer(bn),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
            nn.MaxPool1d(pool_size[0]),

            nn.LazyConv1d(num_channels[1], conv_kernels[1], 1, (conv_kernels[1] - 1) // 2),
            bn1d_layer(bn),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
            nn.MaxPool1d(pool_size[1]),

            nn.LazyConv1d(num_channels[2], conv_kernels[2], 1, (conv_kernels[2] - 1) // 2),
            bn1d_layer(bn),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
            nn.MaxPool1d(pool_size[2]),

            nn.LazyConv1d(num_channels[3], conv_kernels[3], 1, (conv_kernels[3] - 1) // 2),
            bn1d_layer(bn),
            dropout_layer(dropout, drop_rate),
            nn.ReLU(),
            nn.MaxPool1d(pool_size[3]),
        )

        self.classifier = self.build_classifier(output_neurons) if multilabel \
            else self.build_classifier(num_classes)

    def control_bn(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.BatchNorm1d):
                layer.training = training

    def control_dropout(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.Dropout):
                layer.training = training

    def build_classifier(self, num_neurons):
        return nn.Sequential(nn.Flatten(), nn.LazyLinear(256),
                             nn.ReLU(), nn.LazyLinear(128), nn.ReLU(),
                             nn.LazyLinear(num_neurons))

    def forward(self, x):
        features = self.conv_layers(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    # loss_fun = nn.MultiLabelMarginLoss()
    loss_fun = nn.BCEWithLogitsLoss()

    inputs = torch.zeros([32, 1, 2048])
    model = CNN1D(8, 8, multilabel=False)
    outs = model(inputs)
    print(outs.shape)
