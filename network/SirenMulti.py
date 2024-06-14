import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn
from siren_pytorch import SirenNet, SirenWrapper
from utils import info


class MultiSiren(nn.Module):
    def __init__(self, img_size=256, nframes=16, n_latent=1, device='cuda'):
        super().__init__()

        net = SirenNet(
            dim_in = 2,                        # input dimension, ex. 2d coor
            dim_hidden = 3,                  # hidden dimension
            dim_out = 1,                       # output dimension, ex. rgb value
            num_layers = 1,                    # number of layers
            w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        ).to(device)

        self.wrapper = SirenWrapper(
            net,
            latent_dim = n_latent,
            image_width = img_size,
            image_height = img_size
        ).to(device)
        
        self.latents = torch.zeros(nframes, n_latent).normal_(0, 1e-2).to(device)

    def forward(self, img):

        # img is never used here!
        outs = []
        for i in range(self.latents.shape[0]):
            out = self.wrapper(latent = self.latents[i])
            outs.append(out.squeeze(0))
        outs = torch.stack(outs, dim=0) # [16, 1, 256, 256]
        outs = outs.permute(1, 2, 3, 0) # [1, 256, 256, 16]        
        return outs

        # return torch.stack([self.wrapper(latent = latent) for latent in self.latents], dim=0)
