#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift
from aotools.functions import zernikeArray
from torch.fft import fft2, fftshift, irfftn, rfftn, ifftshift
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from utils import info
import torchgeometry as tgm

def conv_psf(obj, psf, use_FFT=True, mask=None, device='cuda'):


    n_batch = obj.shape[0]
    n_slm, n_mask, h, w = psf.shape

    psf = psf.view(-1, h, w).unsqueeze(1)

    if use_FFT:
        y = fft_2xPad_Conv2D(obj, psf)
    else:
        y = F.conv2d(obj, psf, padding='same')

    y = y.view(n_batch, n_slm, n_mask, h, w)
    mask = mask.unsqueeze(0)
    if mask is None:
        return y
    else:
        return (y * mask).sum(dim=2, keepdim=True)


def gen_masks(width=256, grid_size=2, mask_gaussian_std=15, mask_gaussian_size=15, DEVICE=torch.device('cuda'), vis=False):
    # output shape     [1, grid_size**2, 256, 256]
    block_size = width // grid_size
    initial_mask_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            single_mask = torch.zeros(1, 1, width, width).float().to(DEVICE)
            single_mask[..., i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 1
            initial_mask_list.append(single_mask)

    ###############################################################
    # add blur
    target_mask_blur_list = []
    sum_mask = torch.zeros_like(single_mask)
    for mask in initial_mask_list:
        if mask_gaussian_std is not None:
            mask = tgm.image.gaussian_blur(mask, (mask_gaussian_size, mask_gaussian_size), (mask_gaussian_std, mask_gaussian_std)).squeeze()
        target_mask_blur_list.append(mask)
        sum_mask += mask

    # normalization
    target_mask_norm_list = []
    for mask in target_mask_blur_list:
        mask_norm = mask / sum_mask
        target_mask_norm_list.append(mask_norm)

    target_mask_norm_list = torch.stack(target_mask_norm_list).squeeze(1).squeeze(1)

    if vis:
        plt.imshow(target_mask_norm_list[0].squeeze().cpu().numpy().squeeze(), cmap='gray'); plt.show()
        info(target_mask_norm_list)

    # [1, grid_size**2, 256, 256]
    return target_mask_norm_list.unsqueeze(0)


def complex_matmul(a, b, groups = 1):
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def fft_2xPad_Conv2D(signal, kernel, groups=1):
    size = signal.shape[-1]

    signal_fr = rfftn(signal, dim=[-2, -1], s=[2 * size, 2 * size])
    kernel_fr = rfftn(kernel, dim=[-2, -1], s=[2 * size, 2 * size])

    output_fr = complex_matmul(signal_fr, kernel_fr, groups)

    output = irfftn(output_fr, dim=[-2, -1], s=[-1, -1])

    s2 = size//2
    output = output[:, :, s2:-s2, s2:-s2]

    return output


def crop_image(tensor, width):
    start_idx = (tensor.shape[1] - width) // 2
    end_idx = start_idx + width
    # Extract central crop
    cropped_tensor = tensor[:, start_idx:end_idx, start_idx:end_idx]
    return cropped_tensor


def generate_zern_patterns(zern_alpha, crop_zernike, no_translation=True, device='cuda', 
                           return_no_exp=False, slm_mask=None):
    #
    zern_alpha_no_translation = zern_alpha.clone().to(device)

    if no_translation:
        zern_alpha_no_translation[:, :3] = 0

    weights = 2 * zern_alpha_no_translation[..., None, None].to(device)
    weighted_zernike = crop_zernike.unsqueeze(0) * weights
    slm_phs = weighted_zernike.sum(dim=1, keepdim=True).to(device)
    SLM = torch.exp(1j * slm_phs)
    SLM = SLM.permute(1, 0, 2, 3).to(device)


    width = SLM.size()[-1]
    height = int(width / 1920 * 1080)
    offset = (width - height) // 2

    # need to double-check if this line is correct!
    SLM = F.pad(SLM[..., offset:-offset, :], (0, 0, offset, offset), "constant", 0)

    if return_no_exp:
        return SLM, slm_phs

    return SLM


def generate_SLM_mask(width=256, device='cuda'):
    height = int(width / 1920 * 1080)
    a_slm = np.ones((height, width))
    a_slm = np.lib.pad(a_slm, (((width - height) // 2, (width - height) // 2), (0, 0)), 'constant', constant_values=(0, 0))
    a_slm = torch.from_numpy(a_slm).type(torch.float).to(device)
    return a_slm.unsqueeze(0).unsqueeze(0)


def generate_zernike_basis(width, zern_order=7, device='cuda'):
    # num_polynomials = 28  ## orders of Zernike
    # width = 256  ## image size
    num_polynomials = (zern_order*(zern_order+1))//2
    zernike_diam = np.ceil(width * np.sqrt(2))  # radius of 256
    zernike = zernikeArray(num_polynomials, zernike_diam)
    zernike = torch.FloatTensor(zernike).to(device)
    crop_zernike = crop_image(zernike, width).to(device)
    return crop_zernike


def generate_abe(width, mean=0, std=1.0):
    abe_phs = torch.randn(width, width) * std + mean
    abe = torch.exp(1j * abe_phs.unsqueeze(0))
    return abe


def generate_rand_pattern_like(target, std=1.75, downsample=4, mean=0, gaussian_std=None, gaussian_size=11, device='cuda', return_no_exp=False):

    abe_phs = torch.randn_like(torch.abs(target)).to(device) * std + mean

    abe_phs = transforms.Resize((target.size()[-2]//downsample, target.size()[-1]//downsample), interpolation=transforms.InterpolationMode.NEAREST)(abe_phs)
    abe_phs = transforms.Resize((target.size()[-2], target.size()[-1]), interpolation=transforms.InterpolationMode.NEAREST)(abe_phs)

    if gaussian_std is not None:
            abe_phs = tgm.image.gaussian_blur(abe_phs, (gaussian_size, gaussian_size), (gaussian_std, gaussian_std))

    abe = torch.exp(1j * abe_phs)

    width = target.size()[-1]
    height = int(width / 1920 * 1080)
    offset = (width - height) // 2

    # # need to double-check if this line is correct!
    abe = F.pad(abe[..., offset:-offset, :], (0, 0, offset, offset), "constant", 0)

    # #################################################

    if return_no_exp:
        return abe, abe_phs

    return abe


def gen_focus_sweep(focus_std=0.2): # should be 0.1-0.3

    abe_patterns = []
    for i in range(16):
        width = 256
        blur_std = i * focus_std
        defocus = gen_defocus(dim=width, blur_std=blur_std)
        abe_patterns.append(defocus.squeeze(0).squeeze(0))
    abe_patterns = torch.stack(abe_patterns).unsqueeze(0)
    info(abe_patterns, 'abe_patterns focus sweep')
    return abe_patterns

def gen_defocus(dim=256, blur_std=0.5):
    x = torch.linspace(-dim // 2, dim // 2 - 1, dim)
    y = torch.linspace(-dim // 2, dim // 2 - 1, dim)
    xx, yy = torch.meshgrid(x, y)
    phi = blur_std * (xx ** 2 + yy ** 2) * 1e-3
    defocus = phi.unsqueeze(0).unsqueeze(0)
    info(defocus, 'defocus')
    return defocus

def gen_psf(phs, vis=False): # asdfasdfasf
    _kernel = fftshift(fft2(phs, norm="forward"), dim=[-2, -1]).abs() ** 2
    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    _kernel = _kernel.flip(-2).flip(-1)

    if vis:
        plt.imshow(torch.abs(_kernel).numpy().squeeze(), cmap='gray'); plt.show()
    
    return _kernel


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def modulate(obj, phs, use_FFT=True):

    _kernel = fftshift(fft2(phs, norm="forward"), dim=[-2, -1]) .abs() ** 2
    _kernel = _kernel.unsqueeze(0)  ## batch=1
    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    _kernel = _kernel.flip(2).flip(3)

    if use_FFT:
        y = fft_2xPad_Conv2D(obj, _kernel).squeeze()
    else:
        y = F.conv2d(obj, _kernel, padding='same').squeeze()

    return y, _kernel


def test_modulation(test_parameter=None):

    # parameter to be swept
    abe_std = 3
    image_path = 'vis/fox.jpg'
    device = torch.device('cuda')
    width = 256
    zern_order = 7
    grid_size = 2
    slm_std = 1.5
    nframe = 10
    mask_std = 32
    mask_size = round_up_to_odd(width//2)

    obj = Image.open(image_path).convert('L')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    tensor_obj = transform(obj).unsqueeze(0).to(device)
    tensor_obj = tensor_obj.repeat(8, 1, 1, 1)

    zernike_basis = generate_zernike_basis(width=width, zern_order=zern_order)
    slm_mask = generate_SLM_mask(width=width, device=device)
    abe_alphas = abe_std*torch.rand(grid_size**2, (zern_order*(zern_order+1))//2)
    abe_patterns = generate_zern_patterns(abe_alphas, zernike_basis, slm_mask=slm_mask, device=device)
    abe_patterns = generate_rand_pattern_like(abe_patterns, std=test_parameter, device=device)


    abe_psfs = gen_psf(abe_patterns)

    slm_alphas = slm_std * torch.rand(nframe, (zern_order*(zern_order+1))//2)
    slm_patterns = generate_zern_patterns(slm_alphas, zernike_basis, slm_mask=slm_mask, device=device)
    mod_psfs = gen_psf(slm_patterns.permute(1, 0, 2, 3) * abe_patterns)
    mask_batch = gen_masks(width=width, grid_size=grid_size, mask_gaussian_std=mask_std, mask_gaussian_size=mask_size, DEVICE=torch.device('cuda'), vis=False)

    y_mod = conv_psf(tensor_obj, mod_psfs, mask=mask_batch)
    y_zero = conv_psf(tensor_obj, abe_psfs, mask=mask_batch)

    plt.title(f'Zero-{test_parameter}');plt.imshow(y_zero[0, 0].detach().cpu().numpy().squeeze(), cmap='gray'); plt.show()

