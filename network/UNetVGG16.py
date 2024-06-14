import math
import torch
from torch import nn
import torch.nn.functional as F

from torch import nn
import torch
import torch.nn as nn


import math
import torch
from torch import nn
import torch.nn.functional as F

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class SepDepUNet(nn.Module):
    def __init__(self, in_channels, nf0, dropout_prob=0.5, norm=nn.BatchNorm2d, outermost_linear=False, depth_conf=False):
        '''
        :param in_channels: Number of input channels
        :param nf0: Number of features at highest level of U-Net
        :param dropout_prob: Dropout probability.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        self.depth_conf = depth_conf

        leaky_slope = 0.2

        depth_factor = 1
        self.down_conv1 = nn.Sequential(
            SeparableConv2D(in_channels = in_channels, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
        )
        self.pool_1 = nn.MaxPool2d(kernel_size = 2)

        depth_factor = 2
        self.down_conv2 = nn.Sequential(
            SeparableConv2D(in_channels = (depth_factor // 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
        )
        self.pool_2 = nn.MaxPool2d(kernel_size = 2)

        depth_factor = 4
        self.down_conv3 = nn.Sequential(
            SeparableConv2D(in_channels = (depth_factor // 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
        )
        self.pool_3 = nn.MaxPool2d(kernel_size = 2)

        depth_factor = 8
        self.down_conv4 = nn.Sequential(
            SeparableConv2D(in_channels = (depth_factor // 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
        )
        self.pool_4 = nn.MaxPool2d(kernel_size = 2)

        depth_factor = 16
        self.bottle_neck = nn.Sequential(
            SeparableConv2D(in_channels = (depth_factor // 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
        )

        depth_factor = 8
        self.up_trans4 = nn.ConvTranspose2d(in_channels = (depth_factor * 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 2, stride = 2, padding = 0, bias = False)
        self.up_conv4 = nn.Sequential(
            nn.Conv2d(in_channels = (depth_factor * 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1, bias = False),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            nn.Dropout2d(p=dropout_prob),
        )

        depth_factor = 4
        self.up_trans3 = nn.ConvTranspose2d(in_channels = (depth_factor * 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 2, stride = 2, padding = 0, bias = False)
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(in_channels = (depth_factor * 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1, bias = False),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            nn.Dropout2d(p=dropout_prob),
        )

        depth_factor = 2
        self.up_trans2 = nn.ConvTranspose2d(in_channels = (depth_factor * 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 2, stride = 2, padding = 0, bias = False)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(in_channels = (depth_factor * 2) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1, bias = False),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            nn.Dropout2d(p=dropout_prob),
        )

        depth_factor = 2
        self.up_trans1 = nn.ConvTranspose2d(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 2, stride = 2, padding = 0, bias = False)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(in_channels = (depth_factor + 1) * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1, bias = False),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            SeparableConv2D(in_channels = depth_factor * nf0, out_channels = depth_factor * nf0, kernel_size = 3, padding = 1),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            norm(num_features = depth_factor * nf0, affine=True),
            nn.Dropout2d(p=dropout_prob),
        )

        self.dense_1 = nn.Sequential(
            nn.Linear(depth_factor * nf0, depth_factor * 2 * nf0, bias=True),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            nn.Dropout2d(p=dropout_prob),
            nn.Linear(depth_factor * 2 * nf0, depth_factor * nf0, bias=True),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace = True),
            nn.Linear(depth_factor * nf0, nf0, bias=True),
            nn.Linear(nf0, nf0 // 2, bias=True),
        )
        self.dense_n = nn.Linear(nf0 // 2, 3, bias=True)
        if self.depth_conf:
            # add confidence score
            self.dense_d = nn.Linear(nf0 // 2, 2, bias=True)
            self.conf_to_prob = nn.Sigmoid()
        else:
            self.dense_d = nn.Linear(nf0 // 2, 1, bias=True)

    def forward(self, x):
        down_feat1 = self.down_conv1(x)
        down_pool1 = self.pool_1(down_feat1)

        down_feat2 = self.down_conv2(down_pool1)
        down_pool2 = self.pool_2(down_feat2)

        down_feat3 = self.down_conv3(down_pool2)
        down_pool3 = self.pool_3(down_feat3)

        down_feat4 = self.down_conv4(down_pool3)
        down_pool4 = self.pool_4(down_feat4)

        bott_feat = self.bottle_neck(down_pool4)

        up_feat4 = self.up_trans4(bott_feat)
        up_feat4 = torch.cat([up_feat4, down_feat4], dim = 1)
        up_feat4 = self.up_conv4(up_feat4)

        up_feat3 = self.up_trans3(up_feat4)
        up_feat3 = torch.cat([up_feat3, down_feat3], dim = 1)
        up_feat3 = self.up_conv3(up_feat3)

        up_feat2 = self.up_trans2(up_feat3)
        up_feat2 = torch.cat([up_feat2, down_feat2], dim = 1)
        up_feat2 = self.up_conv2(up_feat2)

        up_feat1 = self.up_trans1(up_feat2)
        up_feat1 = torch.cat([up_feat1, down_feat1], dim = 1)
        up_feat1 = self.up_conv1(up_feat1)

        up_feat1 = up_feat1.permute(0, 2, 3, 1)
        out_shape = up_feat1.shape

        dense_feat = self.dense_1(up_feat1)

        out_normal = self.dense_n(dense_feat).view(out_shape[0], out_shape[1], out_shape[2], -1).permute(0, 3, 1, 2)
        out_depth = self.dense_d(dense_feat).view(out_shape[0], out_shape[1], out_shape[2], -1).permute(0, 3, 1, 2)

        return out_depth, out_normal


class VGG16Unet(nn.Module):
    def __init__(self, input_channel=3, vgg_weights = None, residual=True, color_channel=1):
        super(VGG16Unet, self).__init__()
        # Encoder pre-trained vgg features
        """ Usage:
           orig_vgg = torchvision.models.vgg16(pretrained = True)
           model = VGG16Unet(features=orig_vgg.features)
        """

        self.features = vgg_weights
        self.residual = residual
        self.input_channel = input_channel
        self.color_channel = color_channel
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

        self.prediction_N = nn.Conv2d(64, self.color_channel, kernel_size = 3, padding = 1, stride = 1)
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

        x_mean = x.contiguous()[:, 0, ...]

        x = x.contiguous().squeeze(2)  
        x = x[:, :self.input_channel, ...] 


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
