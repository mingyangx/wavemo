from torch import nn
import torch.nn.functional as F

from torch import nn
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, img_ch=3,output_ch=3, n_layers=18, residual=False):
        super(DnCNN, self).__init__()
        # in layer
        self.conv1 = nn.Conv2d(in_channels=img_ch, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        # hidden layers
        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            hidden_layers.append(nn.BatchNorm2d(64))
            hidden_layers.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*hidden_layers)
        # out layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=output_ch, kernel_size=3, padding=1, bias=False)
        self.input_channels = img_ch
        self.residual = residual

    def forward(self, x):

        ##################################

        x_zero = x.contiguous()[:, 0, ...]
        x = x.contiguous().squeeze(2)  
        x = x[:, :self.input_channels, ...] 

        ##################################

        out = self.relu1(self.conv1(x))
        out = self.mid_layer(out)
        out = self.conv3(out)

        if self.residual:       
            return x_zero - out
        
        return out