import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import Parameter
from functools import wraps
import torch.nn.utils.weight_norm as WeightNorm

class GlobalMaxPool(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, input):
        output = torch.max(input, dim=1)[0]

        return torch.unsqueeze(output, 1)


class FredoUNetMemory(nn.Module):

    # an implementation of Unet with global maxpooling presented in
    # http://people.csail.mit.edu/miika/eccv18_deblur/aittala_eccv18_deblur_preprint.pdf
    # dimensions are specified at http://people.csail.mit.edu/miika/eccv18_deblur/aittala_eccv18_deblur_appendix.pdf
    # Implemented by Frederik Warburg
    # For implementation details contact: frewar1905@gmail.com

    def conv_elu(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(out_channels)
        )

        return block

    def contracting_block(self, in_channels, out_channels, kernel_size=4):
        block = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, padding=1), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(out_channels),
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, stride = 2, padding=0), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=mid_channel, padding=1), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(mid_channel),
            WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel), name = "weight"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(mid_channel),
            WeightNorm(torch.nn.ConvTranspose2d(kernel_size=4, in_channels=mid_channel, out_channels=out_channels, stride=2, padding=1), name = "weight")
        )
        return block

    def final_block(self, in_channels, out_channels, mid_channel, kernel_size=3, final_sigmoid=False, use_weight_norm=True):
        if final_sigmoid:
            if use_weight_norm:
                # print(';lakjsdf;laskdfja;lsfj')
                # exit()
                block = torch.nn.Sequential(
                    WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding = 1), name = "weight"),
                    torch.nn.ELU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding = 1), name = "weight"),
                    torch.nn.Sigmoid()
                )
            else:
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding = 1),
                    torch.nn.ELU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding = 1),
                    torch.nn.Sigmoid()
                )
        else:
            block = torch.nn.Sequential(
                WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding = 1), name = "weight"),
                torch.nn.ELU(),
                torch.nn.BatchNorm2d(mid_channel),
                WeightNorm(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding = 1), name = "weight"),
            )

        return block

    def __init__(self, in_channel, out_channel, residual=False, sigmoid=False, save_memory=False, residual_alpha=1.0,
                 final_weight_norm=True, shrink_ratio=1.0):
        super(FredoUNetMemory, self).__init__()

        # Encode
        self.in_channel = in_channel
        self.conv_encode1 = self.conv_elu(in_channels=in_channel,   out_channels= 64,  kernel_size= 3) #id 1
        self.conv_encode2 = self.contracting_block(in_channels= 2*64,  out_channels= 128, kernel_size= 4) #id 4, 5
        self.conv_encode3 = self.contracting_block(in_channels= 2*128, out_channels= 256, kernel_size= 4) #id 8, 9
        self.conv_encode4 = self.contracting_block(in_channels= 2*256, out_channels= 384, kernel_size= 4) #id 12, 13
        self.residual = residual
        self.sigmoid = sigmoid
        self.save_memory = save_memory
        self.residual_alpha = residual_alpha
        self.final_weight_norm = final_weight_norm
        self.shrink_ratio = shrink_ratio
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            WeightNorm(torch.nn.Conv2d(kernel_size=1, in_channels=2*384, out_channels=384), name = "weight"), #id 16
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(384),
            WeightNorm(torch.nn.Conv2d(kernel_size=4, in_channels=384, out_channels=384, stride=2), name = "weight"), #id 17
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(384),
            WeightNorm(torch.nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, padding=1), name = "weight"), #id 18
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(384),
            WeightNorm(torch.nn.ConvTranspose2d(kernel_size=4, in_channels=384, out_channels=384, stride=2), name = "weight") #id 19
        )
        # Decode
        self.conv_decode3 = self.expansive_block(in_channels = 3 * 384, out_channels= 256, mid_channel= 384) #id 22, 23, 24
        self.conv_decode2 = self.expansive_block(in_channels = 3 * 256, out_channels= 192, mid_channel= 256) #id 27, 28, 29
        self.conv_decode1 = self.expansive_block(in_channels = 2 * 192 + 128, out_channels= 96 , mid_channel= 192) #id 32, 33, 34
        self.conv_decode0 = self.conv_elu(in_channels= 2*96 + 64, out_channels= 96,  kernel_size= 3)             #id 37, 38
        self.final_layer  = self.final_block(in_channels=96, out_channels = out_channel, mid_channel= 64,
                                             final_sigmoid=self.sigmoid, use_weight_norm=self.final_weight_norm)      #id 40, 41

        self.pooling = GlobalMaxPool()

    def concat(self, x, max_):

        b, im, c, h, w = x.size()

        x = x.view(b*im, c, h, w)
        max_ = max_.repeat(1, im, 1, 1, 1).view(b*im, c, h, w)
        output = torch.cat([x, max_], dim=1)

        return output

    def concat2(self, x, max_, y):
        b, im, c1, h, w = x.size()
        _, _, c2, _, _ = y.size()

        x = x.view(b * im, c1, h, w)
        y = y.view(b * im, c2, h, w)
        max_ = max_.repeat(1, im, 1, 1, 1).view(b * im, c1, h, w)
        output = torch.cat([x, max_, y], dim=1)

        return output

    def forward(self, x, output_many_frames=False):

        x_mean = x.contiguous()[:, :, :3, :, :].mean(axis=1)
 

        b, im, c, h, w = x.size()

        # Encode
        encode_block1 = self.conv_encode1(x.view((b*im, c, h, w))) # id = 1

        _, _, h, w = encode_block1.size()
        features1 = encode_block1.view((b, im, -1, h, w))

        max_global_features1 = self.pooling(features1)  # id = 2

        encode_pool1 = self.concat(features1, max_global_features1) # id = 3
        encode_block2 = self.conv_encode2(encode_pool1) # id = 4, 5

        if self.save_memory:
            del encode_block1, max_global_features1, encode_pool1
            torch.cuda.empty_cache()

        _, _, h, w = encode_block2.size()
        features2 = encode_block2.view((b, im, -1, h, w))
        max_global_features2 = self.pooling(features2) # id = 6

        encode_pool2 = self.concat(features2, max_global_features2)
        encode_block3 = self.conv_encode3(encode_pool2) # id = 8, 9

        if self.save_memory:
            del encode_block2, max_global_features2, encode_pool2
            torch.cuda.empty_cache()

        _, _, h, w = encode_block3.size()
        features3 = encode_block3.view((b, im, -1, h, w))
        max_global_features3 = self.pooling(features3) # id = 10

        encode_pool3 = self.concat(features3, max_global_features3)
        encode_block4 = self.conv_encode4(encode_pool3) # id = 12, 13

        if self.save_memory:
            del encode_block3, max_global_features3, encode_pool3
            torch.cuda.empty_cache()


        _, _, h, w = encode_block4.size()
        features4 = encode_block4.view((b, im, -1, h, w))
        max_global_features4 = self.pooling(features4) # id = 14

        # Bottleneck
        encode_pool4 = self.concat(features4, max_global_features4)
        bottleneck = self.bottleneck(encode_pool4) # id = 16, 17, 18, 19

        if self.save_memory:
            del encode_block4, max_global_features4, encode_pool4
            torch.cuda.empty_cache()


        _, _, h, w = bottleneck.size()
        features5 = bottleneck.view((b, im, -1, h, w))
        max_global_features5 = self.pooling(features5) # id = 20

        # Decode
        decode_block4 = self.concat2(features5, max_global_features5, features4)
        cat_layer3 = self.conv_decode3(decode_block4) # id = 22, 23, 24


        if self.save_memory:
            del bottleneck, max_global_features5, features4, features5, decode_block4
            torch.cuda.empty_cache()

        _, _, h, w = cat_layer3.size()
        features6 = cat_layer3.view((b, im, -1, h, w))
        max_global_features6 = self.pooling(features6) # id = 25

        decode_block3 = self.concat2(features6, max_global_features6, features3)

        if self.save_memory:
            del cat_layer3, max_global_features6, features3, features6; torch.cuda.empty_cache()
        cat_layer2 = self.conv_decode2(decode_block3) # id = 27, 28, 29

        if self.save_memory:
            del decode_block3; torch.cuda.empty_cache()

        _, _, h, w = cat_layer2.size()
        features7 = cat_layer2.view((b, im, -1, h, w))
        max_global_features7 = self.pooling(features7) # id = 30
        decode_block2 = self.concat2(features7, max_global_features7, features2)
        # gpuinfo()

        if self.save_memory:
            del cat_layer2, max_global_features7, features2, features7; torch.cuda.empty_cache()
        cat_layer1 = self.conv_decode1(decode_block2) # id = 32, 33, 34
        # gpuinfo()

        if self.save_memory:
            del decode_block2; torch.cuda.empty_cache()

        _, _, h, w = cat_layer1.size()
        features8 = cat_layer1.view((b, im, -1, h, w))
        max_global_features8 = self.pooling(features8) # id = 35


        decode_block1 = self.concat2(features8, max_global_features8, features1)

        if self.save_memory:
            del cat_layer1, max_global_features8, features1, features8; torch.cuda.empty_cache()

        cat_layer0 = self.conv_decode0(decode_block1) # id = 37, 38

        if self.save_memory:
            del decode_block1; torch.cuda.empty_cache()



        _, _, h, w = cat_layer0.size()
        features9 = cat_layer0.view((b, im, -1, h, w))
        max_global_features9 = self.pooling(features9) # id = 39

        if not output_many_frames:
            final_layer = self.final_layer(torch.squeeze(max_global_features9, dim = 1))
        else:
            final_layer = self.final_layer(torch.squeeze(features9, dim = 0))

        if self.save_memory:
            del cat_layer0, max_global_features9, features9
        if self.residual:

            if self.sigmoid:
                final_layer = (final_layer * 2 - 1) * self.shrink_ratio

                final_layer = torch.clamp(final_layer, -0.19, 0.19)

            if not output_many_frames:
                return x_mean - final_layer
            else:
                return x.squeeze(0) - final_layer
        else:
            return final_layer
