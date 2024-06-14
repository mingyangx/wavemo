import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from utils import info
from termcolor import colored
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn


import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from utils import info
from termcolor import colored
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn


class MaxPoolUnet(nn.Module):
    def __init__(self, input_channel=3, vgg_weights = None, residual=False):
        super(MaxPoolUnet, self).__init__()
        # Encoder pre-trained vgg features
        """ Usage:
           orig_vgg = torchvision.models.vgg16(pretrained = True)
           model = VGG16UnetMaxPool(features=orig_vgg.features)
        """
        print('UNetMaxPool has not been implemented yet.')
        exit()
        self.features = vgg_weights
        self.residual = residual
        # upsamplig block
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2.0)

        # Rest blocks
        # Encoder
        self.enc_block0 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        self.enc_block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )

        self.enc_block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )

        self.enc_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(256, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )

        self.enc_block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )

        self.bottle = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )

        # Decoder
        # unet connection with enc_block5 (input will be 1024)
        self.dec_block4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )

        # unet connection with enc_block5 (input will be 512)
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )

        self.dec_block2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )

        self.dec_block1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        self.dec_block0 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        self.prediction_N = nn.Conv2d(64, 3, kernel_size = 3, padding = 1, stride = 1)
        # self.prediction_D = nn.Conv2d(64, 1, kernel_size = 3, padding = 1, stride = 1)

    def init_weights(self, init = 'xavier'):
        if init == 'xavier':
            init_func = torch.nn.init.xavier_normal_
        else:
            init_func = torch.nn.init.normal_

        init_func(self.enc_block0._modules['0'].weight)
        init_func(self.enc_block0._modules['3'].weight)

        if self.features is not None:
            # pretrained weights initialization
            # self.enc_block0._modules['0'].weight = self.features._modules['0'].weight
            # self.enc_block0._modules['3'].weight = self.features._modules['2'].weight
            self.enc_block1._modules['1'].weight = self.features._modules['5'].weight
            self.enc_block1._modules['4'].weight = self.features._modules['7'].weight
            self.enc_block2._modules['1'].weight = self.features._modules['10'].weight
            self.enc_block2._modules['4'].weight = self.features._modules['12'].weight
            self.enc_block2._modules['7'].weight = self.features._modules['14'].weight
            self.enc_block3._modules['1'].weight = self.features._modules['17'].weight
            self.enc_block3._modules['4'].weight = self.features._modules['19'].weight
            self.enc_block3._modules['7'].weight = self.features._modules['21'].weight
            self.enc_block4._modules['1'].weight = self.features._modules['24'].weight
            self.enc_block4._modules['4'].weight = self.features._modules['26'].weight
            self.enc_block4._modules['7'].weight = self.features._modules['28'].weight
        else:
            # init_func(self.enc_block0._modules['0'].weight)
            # init_func(self.enc_block0._modules['3'].weight)
            init_func(self.enc_block1._modules['1'].weight)
            init_func(self.enc_block1._modules['4'].weight)
            init_func(self.enc_block2._modules['1'].weight)
            init_func(self.enc_block2._modules['4'].weight)
            init_func(self.enc_block2._modules['7'].weight)
            init_func(self.enc_block3._modules['1'].weight)
            init_func(self.enc_block3._modules['4'].weight)
            init_func(self.enc_block3._modules['7'].weight)
            init_func(self.enc_block4._modules['1'].weight)
            init_func(self.enc_block4._modules['4'].weight)
            init_func(self.enc_block4._modules['7'].weight)

        #### Decoder Initialization
        init_func(self.bottle._modules['1'].weight)
        init_func(self.dec_block4._modules['0'].weight)
        init_func(self.dec_block4._modules['3'].weight)
        init_func(self.dec_block4._modules['6'].weight)
        init_func(self.dec_block3._modules['0'].weight)
        init_func(self.dec_block3._modules['3'].weight)
        init_func(self.dec_block3._modules['6'].weight)
        init_func(self.dec_block2._modules['0'].weight)
        init_func(self.dec_block2._modules['3'].weight)
        init_func(self.dec_block2._modules['6'].weight)
        init_func(self.dec_block1._modules['0'].weight)
        init_func(self.dec_block1._modules['3'].weight)
        init_func(self.dec_block0._modules['0'].weight)
        init_func(self.dec_block0._modules['3'].weight)
        #init_func(self.prediction_flow.weight)
        # init_func(self.prediction_D.weight)
        init_func(self.prediction_N.weight)

    def forward(self, x):
        x_mean = x.mean(axis=0).unsqueeze(0)
        nframe, nchannel, h, w = x.size()
        x = x.view(1, -1, h, w)

        x0 = self.enc_block0(x)
        x1 = self.enc_block1(x0)
        x2 = self.enc_block2(x1)
        x3 = self.enc_block3(x2)
        x4 = self.enc_block4(x3)

        b = self.bottle(x4)


        y4 = self.up(b)


        y4 = torch.cat((x4, y4), dim = 1)
        y4 = self.dec_block4(y4)

        y3 = self.up(y4)
        y3 = torch.cat((x3, y3), dim = 1)
        y3 = self.dec_block3(y3)

        y2 = self.up(y3)
        y2 = torch.cat((x2, y2), dim = 1)
        y2 = self.dec_block2(y2)

        y1 = self.up(y2)
        y1 = torch.cat((x1, y1), dim = 1)
        y1 = self.dec_block1(y1)

        y0 = self.up(y1)
        y0 = torch.cat((x0, y0), dim = 1)
        y0 = self.dec_block0(y0)

        pred_N = self.prediction_N(y0)


        if self.residual:
            return x_mean - pred_N
        else:
            return pred_N
