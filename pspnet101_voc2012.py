import torch
import torch.nn as nn


class _Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, downsample=None):
        super(_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.95, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.95, affine=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=1e-5, momentum=0.95, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _PyramidPool(nn.Module):
    def __init__(self):
        super(_PyramidPool, self).__init__()

        self.avg_pool1 = nn.AvgPool2d(kernel_size=60, stride=60)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.95, affine=True)

        self.avg_pool2 = nn.AvgPool2d(kernel_size=30, stride=30)
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.95, affine=True)

        self.avg_pool3 = nn.AvgPool2d(kernel_size=20, stride=20)
        self.conv3 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.95, affine=True)

        self.avg_pool4 = nn.AvgPool2d(kernel_size=10, stride=10)
        self.conv4 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.95, affine=True)

        self.upsample = nn.Upsample(size=(60, 60), mode='bilinear')

    def forward(self, x):
        results = []
        # size (1,1)
        out = self.avg_pool1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.upsample(out)
        results.append(out)
        # size (2,2)
        out = self.avg_pool2(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.upsample(out)
        results.append(out)
        # size (3,3)
        out = self.avg_pool3(x)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.upsample(out)
        results.append(out)
        # size (6,6)
        out = self.avg_pool4(x)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.upsample(out)
        results.append(out)

        results.append(x)
        # concat order [x, (6,6), (3,3), (2,2), (1,1)]
        results.reverse()
        out = torch.cat(results, dim=1)
        return out


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.95, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.95, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.95, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = self._make_layer(_Bottleneck, 128, 64, num_blocks=3, stride=1, dilation=1)
        self.layer3 = self._make_layer(_Bottleneck, 256, 128, num_blocks=4, stride=2, dilation=1)
        self.layer4 = self._make_layer(_Bottleneck, 512, 256, num_blocks=23, stride=1, dilation=2)
        self.layer5 = self._make_layer(_Bottleneck, 1024, 512, num_blocks=3, stride=1, dilation=4)
        self.ppm = _PyramidPool()
        self.pred = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.95, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(512, 21, kernel_size=1, stride=1),
            nn.Upsample(size=(473, 473), mode='bilinear')
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.ppm(out)
        out = self.pred(out)
        return out

    @staticmethod
    def _make_layer(block, inplanes, planes, num_blocks, stride, dilation):
        layer = []
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 4, eps=1e-5, momentum=0.95, affine=True)
        )
        padding = dilation
        layer.append(block(inplanes, planes, stride, padding, dilation, downsample))
        for _ in range(num_blocks - 1):
            layer.append(block(planes * 4, planes, stride=1, padding=padding, dilation=dilation))
        return nn.Sequential(*layer)
