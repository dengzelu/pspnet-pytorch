import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, inner_channels, stride, padding, dilation, shortcut=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels, affine=True)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channels, affine=True)
        self.conv3 = nn.Conv2d(inner_channels, inner_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inner_channels * 4, affine=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        
    def forward(self, input_):
        residual = input_
        
        output = self.conv1(input_)
        output = self.bn1(output)
        output = self.relu(output)
        
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        
        if self.shortcut is not None:
            residual = self.shortcut(input_)
        
        output += residual
        output = self.relu(output)
        return output
        
        
class PyramidPooling(nn.Module):
    def __init__(self, in_channels=2048):
        super(PyramidPooling, self).__init__()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=60, stride=60)
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512, affine=True)
        
        self.avg_pool2 = nn.AvgPool2d(kernel_size=30, stride=30)
        self.conv2 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512, affine=True)
        
        self.avg_pool3 = nn.AvgPool2d(kernel_size=20, stride=20)
        self.conv3 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512, affine=True)
        
        self.avg_pool4 = nn.AvgPool2d(kernel_size=10, stride=10)
        self.conv4 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512, affine=True)
        
        self.upsample = nn.Upsample(size=(60, 60), mode='bilinear')
    
    def forward(self, input_):
        print(input_.size())
        results = []
        results.append(input_)
        
        output = self.avg_pool1(input_)
        output = self.conv1(output)
        output = self.bn1(output)
        results.append(self.upsample(output))
        
        output = self.avg_pool2(input_)
        output = self.conv2(output)
        output = self.bn2(output)
        results.append(self.upsample(output))
        
        output = self.avg_pool3(input_)
        output = self.conv3(output)
        output = self.bn3(output)
        results.append(self.upsample(output))
        
        output = self.avg_pool4(input_)
        output = self.conv4(output)
        output = self.bn4(output)
        results.append(self.upsample(output))
    
        output = torch.cat(results, dim=1)
        return output

class PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super(PSPNet, self).__init__()
        self.num_classes = 21
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = self._make_layer(Bottleneck, 128, 64, num_blocks=3, stride=1, dilation=1)
        self.layer3 = self._make_layer(Bottleneck, 256, 128, num_blocks=4, stride=2, dilation=1)
        self.layer4 = self._make_layer(Bottleneck, 512, 256, num_blocks=23, stride=1, dilation=2)
        self.layer5 = self._make_layer(Bottleneck, 1024, 512, num_blocks=3, stride=1, dilation=4)
        self.pyramid_pooling = PyramidPooling()
        self.pred = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(512, 21, kernel_size=1, stride=1),
            nn.Upsample(size=(483, 483), mode='bilinear')
        )
    
    def forward(self, input_):
        output = self.layer1(input_)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.pyramid_pooling(output)
        output = self.pred(output)
        return output
    
    @staticmethod
    def _make_layer(block, in_channels, inner_channels, num_blocks, stride=1, dilation=1):
        layer = []
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels * 4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(inner_channels * 4, affine=True)
        )
        padding = dilation
        layer.append(Bottleneck(in_channels, inner_channels, stride=stride,
                                padding=padding, dilation=dilation, shortcut=shortcut))
        for _ in range(num_blocks - 1):
            layer.append(Bottleneck(inner_channels * 4, inner_channels, stride=1,
                                    padding=padding, dilation=dilation))
        return nn.Sequential(*layer)
