import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, image_channels, block, layers, num_classes):
        """
        This defines the sequential CNNs that are stacked on top of each other as layers. We do a global convolution
        and then each layer perform block CNNs.
        :param block: A block type [BasicBlock]
        :param layers: List of number of blocks per layer
        :param num_classes: Number of output classes
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(image_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * 4 * 19, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 10x6x40x100
        x = self.conv1(x)
        # 10x64x20x50
        x = self.bn1(x)
        # 10x64x20x50
        x = self.relu(x)
        # 10x64x20x50
        x = self.maxpool(x)

        # 10x64x10x25
        x = self.layer1(x)
        # 10x128x10x25
        x = self.layer2(x)
        # 10x256x10x25
        x = self.layer3(x)
        # 10x512x10x25
        x = self.layer4(x)
        # 10x512x10x25
        x = self.avgpool(x)
        # 10x512x4x19
        x = x.view(x.size(0), -1)
        # 10x38912
        x = self.fc(x)
        # 10x20
        return x


class SeqResNet(nn.Module):
    def __init__(self, image_channels, seq_length, num_classes):
        super(SeqResNet, self).__init__()
        self.image_channels = image_channels
        self.block = BasicBlock
        self.layers = [3, 3, 3, 3]
        self.num_classes = num_classes
        self.all_cells = nn.ModuleList([ResNet(self.image_channels, self.block, self.layers, self.num_classes)
                                      for i in range(seq_length)])

    def _make_cell_for_each_base(self, sequence_length):
        cells = list()
        for i in range(sequence_length):
            cells.append(ResNet(self.image_channels, self.block, self.layers, self.num_classes))

        return cells

    def forward(self, whole_image):
        start_base_index = 20   # based on flanking region
        end_base_index = start_base_index + 20  # based on amount of image encoded in one image for training
        cell_index = 0
        seq_preds = []
        for base_index in range(start_base_index, end_base_index):
            segment_start = base_index - 20
            segment_end = base_index + 20
            base_segment_image = whole_image[:, :, segment_start:segment_end, :]
            cell_logits = self.all_cells[cell_index](base_segment_image)
            seq_preds.append(cell_logits)
            cell_logits += 1
        sequence_predictions = torch.stack(seq_preds, dim=1)
        return sequence_predictions
