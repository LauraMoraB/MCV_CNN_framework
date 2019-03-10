import sys
from torch import nn
import torchvision.models.vgg as models
sys.path.append('../')
from models.networks.network import Net
import math

class Mini_net(Net):
    def __init__(self, cf, num_classes=21, pretrained=False, net_name='mini_net'):
        super(Mini_net, self).__init__(cf)

        self.url = ''
        self.pretrained = pretrained
        self.net_name = net_name

        self.model = None

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

        self.classifier = nn.Sequential(
            # Block:
            nn.BatchNorm1d(144),
            nn.Linear(144, 64),
            nn.ReLU(False),
            nn.Dropout(p=0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(16, num_classes)

        )

        # if(not pretrained):
        #     self._initialize_weights()

    def forward(self, x):
        print("mininet!")
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
