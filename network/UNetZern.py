import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from modulate import generate_zernike_basis
from torchvision import transforms, models


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x



class conv_block_zern_end(nn.Module):
    def __init__(self,ch_in,ch_out, zern_basis=None):
        super(conv_block,self).__init__()

        ch_out = ch_out - zern_basis.shape[0] if zern_basis is not None else ch_out

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.zern_basis = zern_basis


    def forward(self,x):
        x = self.conv(x)
        if self.zern_basis is not None:
            x = torch.cat((x, self.zern_basis.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)), dim=1)
        return x
    

class conv_block_zern_middle(nn.Module):
    def __init__(self,ch_in,ch_out, zern_basis=None):
        super(conv_block_zern_middle,self).__init__()

        ch_middle = ch_out - zern_basis.shape[0] if zern_basis is not None else ch_out

        self.conv_01 = nn.Sequential(
            nn.Conv2d(ch_in, ch_middle, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_middle),
            nn.ReLU(inplace=True),
        )

        self.conv_02 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.zern_basis = zern_basis


    def forward(self,x):
        x = self.conv_01(x)
        if self.zern_basis is not None:
            x = torch.cat((x, self.zern_basis.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)), dim=1)
        x = self.conv_02(x)
        return x
    


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class ZernUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=3, img_size = 256, zern_order=7, zern_insertion='none', device='cuda'):
        super(ZernUNet,self).__init__()
        
        self.zernike_basis = generate_zernike_basis(width=img_size, zern_order=zern_order) # torch.Size([28, 256, 256])
        self.zern_3 = transforms.Resize(size=img_size)(self.zernike_basis[21:28])
        self.zern_4 = transforms.Resize(size=img_size//2)(self.zernike_basis[15:21])
        self.zern_5 = transforms.Resize(size=img_size//4)(self.zernike_basis[10:15])
        self.zern_6 = transforms.Resize(size=img_size//8)(self.zernike_basis[6:10])
        self.zern_7 = transforms.Resize(size=img_size//16)(self.zernike_basis[3:6])  

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        scale_factor = 4
        dim0 = 32 // scale_factor
        dim1 = 64 // scale_factor
        dim2 = 128 // scale_factor
        dim3 = 256 // scale_factor
        dim4 = 512 // scale_factor
        dim5 = 1024 // scale_factor

        # dim0 = 64
        # dim1 = 96
        # dim2 = 128
        # dim3 = 160
        # dim4 = 192
        # dim5 = 256

        if zern_insertion == 'end':
            self.Conv1 = conv_block_zern_end(ch_in=img_ch,ch_out=dim1, zern_basis=self.zern_3)
            self.Conv2 = conv_block_zern_end(ch_in=dim1,ch_out=dim2, zern_basis=self.zern_4)
            self.Conv3 = conv_block_zern_end(ch_in=dim2,ch_out=dim3, zern_basis=self.zern_5)
            self.Conv4 = conv_block_zern_end(ch_in=dim3,ch_out=dim4, zern_basis=self.zern_6)
            self.Conv5 = conv_block_zern_end(ch_in=dim4,ch_out=dim5, zern_basis=self.zern_7)
        elif zern_insertion == 'middle':
            self.Conv1 = conv_block_zern_middle(ch_in=img_ch,ch_out=dim1, zern_basis=self.zern_3)
            self.Conv2 = conv_block_zern_middle(ch_in=dim1,ch_out=dim2, zern_basis=self.zern_4)
            self.Conv3 = conv_block_zern_middle(ch_in=dim2,ch_out=dim3, zern_basis=self.zern_5)
            self.Conv4 = conv_block_zern_middle(ch_in=dim3,ch_out=dim4, zern_basis=self.zern_6)
            self.Conv5 = conv_block_zern_middle(ch_in=dim4,ch_out=dim5, zern_basis=self.zern_7)     
        elif zern_insertion == 'none':
            self.Conv1 = conv_block(ch_in=img_ch,ch_out=dim1)
            self.Conv2 = conv_block(ch_in=dim1,ch_out=dim2)
            self.Conv3 = conv_block(ch_in=dim2,ch_out=dim3)
            self.Conv4 = conv_block(ch_in=dim3,ch_out=dim4)
            self.Conv5 = conv_block(ch_in=dim4,ch_out=dim5)
        else:
            raise ValueError('zern_insertion must be one of "end", "middle", "none"')   


        self.Up5 = up_conv(ch_in=dim5,ch_out=dim4)
        self.Att5 = Attention_block(F_g=dim4,F_l=dim4,F_int=dim3)
        self.Up_conv5 = conv_block(ch_in=dim5, ch_out=dim4)

        self.Up4 = up_conv(ch_in=dim4,ch_out=dim3)
        self.Att4 = Attention_block(F_g=dim3,F_l=dim3,F_int=dim2)
        self.Up_conv4 = conv_block(ch_in=dim4, ch_out=dim3)
        
        self.Up3 = up_conv(ch_in=dim3,ch_out=dim2)
        self.Att3 = Attention_block(F_g=dim2,F_l=dim2,F_int=dim1)
        self.Up_conv3 = conv_block(ch_in=dim3, ch_out=dim2)
        
        self.Up2 = up_conv(ch_in=dim2,ch_out=dim1)
        self.Att2 = Attention_block(F_g=dim1,F_l=dim1,F_int=dim0)
        self.Up_conv2 = conv_block(ch_in=dim2, ch_out=dim1)

        self.Conv_1x1 = nn.Conv2d(dim1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        
        # encoding path
        x1 = self.Conv1(x) # 256

        x2 = self.Maxpool(x1) # 128
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2) # 64
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3) # 32
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4) # 16
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # before 16, 1, 256, 256
        d1 = d1.permute(1, 2, 3, 0)
        # after 1, 256, 256, 16

        return d1
