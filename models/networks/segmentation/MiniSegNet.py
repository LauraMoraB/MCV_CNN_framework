import sys
import torch
import numpy as np
from torch import nn

sys.path.append('../')
from models.networks.network import Net

class MiniSegNet(Net):

    def __init__(self, cf, num_classes=21, pretrained=False, net_name='minisegnet'):
        
        super(MiniSegNet, self).__init__(cf)

        self.url = ''
        self.pretrained = pretrained
        self.net_name = net_name


        self.features = nn.Sequential(

            nn.BatchNorm2d(3),
            nn.Conv2d(3, 48, kernel_size=5, padding=0, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.2),
            
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 24, kernel_size=3, padding=0, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=3, padding=0, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.2),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.BatchNorm2d(24),
            nn.Conv2d(24, 16, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.2),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.2)
        )

        
        # Get a new 1x1 convolution and randomly initialize
        score_32s = nn.Conv2d(16, num_classes,kernel_size=1)
        self._normal_initialization(score_32s)
        self.score_32s = score_32s

        
    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        h = x
        input_spatial_dim = x.size()[2:]
        h = self.features(h)
        #h = h.view(h.size(0), -1)
        """
        h = self.drop1_1(self.relu1_1(self.conv1_1(self.bn2d1_1(h))))
        h = self.relu2_1(self.conv2_1(self.bn2d2_1(h)))
        h = self.drop3_1(self.relu3_1(self.conv3_1(self.bn2d3_1(h))))
        h = self.pool3(h)
        """
        h = self.score_32s(h)
        output =  nn.functional.interpolate(input=h, size=input_spatial_dim)
        
        return output

