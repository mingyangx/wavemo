from termcolor import colored
import os
import matplotlib
from matplotlib import figure
from torchvision.utils import save_image
import gc
from shutil import copytree, ignore_patterns
import psutil
import shutil
import torch
import numpy as np
import cv2
from torch.fft import fft2, fftshift, irfftn, rfftn
import imageio
import time
from torch.autograd import Variable
import torchvision.transforms.functional as F
from torch.autograd import Variable
import random
from matplotlib import pyplot as plt
import os, random, time
from termcolor import colored
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift
import shutil, imageio
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity 
import torchvision
import piq.psnr as piq_psnr
import piq.ssim as piq_ssim
import warnings, logging
import numpy as np
import torch, time
from torchvision import transforms
import logging, warnings, wandb
from pynvml import *
import scipy.io as sio
import wandb


def loadGIF(filename):
    gif = cv2.VideoCapture(filename)
    frames = []
    while True:
        ret, cv2Image = gif.read()
        if not ret:
            break
        frames.append(cv2Image)
    return np.stack(frames)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def total_variation(img: torch.Tensor) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)

    return res1 + res2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".gif"])


def gaussian(ins, std=0):
    noise = Variable(ins.data.new(ins.size()).normal_(0, std/255.))
    return ins + noise, noise


def poisson(clean, alpha):
    if alpha == 0 or alpha in ['none', 'None', 'NONE'] or alpha is None:
        return clean, 0
    else:
        noisy = (alpha * clean) + torch.sqrt(alpha * clean) * torch.randn_like(clean)
        noisy = noisy / alpha
        return noisy, noisy - clean


def gpuinfo(text=None):
    nvmlInit()
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        print(colored(f'{text} :GPU {i}, total {info.total/1024**3}G, free {info.free/1024**3}G, used: {info.used/1024**3}G', 'red'))
    print()


def save_img(torchimg, save_path='img.png'):
    # torchimg = torchimg.squeeze()
    torchimg = (torchimg - torch.min(torchimg))/(torch.max(torchimg) - torch.min(torchimg))
    imageio.imsave(save_path, np.uint8(255 * torch.clamp(torchimg, 0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()).squeeze())


def save_gif(torchgif, save_path):
    imageio.mimsave(save_path, np.uint8(255 * torch.clamp(torchgif, 0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()), fps=5)


def torch2img(torchimg):
    return np.uint8(255 * torchimg.squeeze().permute(1, 2, 0).detach().cpu().numpy()).squeeze()


def log_image(log_tensor, log_title, log_path_no_ext, log_wandb=True):
    save_image(log_tensor, f'{log_path_no_ext}.png', normalize=True, scale_each=False)
    if log_wandb:
        wandb.log({log_title: wandb.Image(f'{log_path_no_ext}.png')})
    
    # get the folder of the log_path_no_ext
    log_folder = os.path.dirname(log_path_no_ext)
    # get the filename of the log_path_no_ext
    log_filename = os.path.basename(log_path_no_ext)
    mat_path = os.path.join(log_folder, f'zmat_{log_filename}.mat')
    sio.savemat(mat_path, {'matrix':log_tensor.detach().cpu().numpy()})


def save_captioned_imgs(img_list, caption_list=['Truth', 'Sample Mean', 'Recon'], is_torch_tensor=True, rescale=False,
                        save_path='no_name_defined.png', save_path_2=None, save_path_3=None, grayscale=False, flip=False, 
                        save_imgs_path=''):

    if len(img_list) != len(caption_list):
        print('number of images must equal to number of captions')

    def save_at_each_path(temp_path):
        if temp_path is not None:

            save_image(torch.stack(img_list).squeeze(1), save_path.replace('.png', '_combined.png'), normalize=True, scale_each=False)

            for i in range(len(img_list)):
                if caption_list[i] is not None:
                    save_image(img_list[i], temp_path.replace('.png', f'_{caption_list[i].lower()}.png'), normalize=False, scale_each=False)

        return
    save_at_each_path(save_path)
    save_at_each_path(save_path_2)
    save_at_each_path(save_path_3)
    return save_path.replace('.png', '_combined.png')


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def Power_Spectrum(img):
    Ifft = abs(fftshift(fft2(img, (img.shape[0], img.shape[1]))))
    return Ifft
    

def info(vector=None, name='', precision=4, color='red'):
    """
    check info
    :param name: name
    :param vector: torch tensor or numpy array or list of tensor/np array
    """
    if torch.is_tensor(vector):
        if torch.is_complex(vector):
            print(colored(name, color) + f' tensor size: {vector.size()}, mean: {torch.mean(vector).item():.{precision}f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
        else:
            try:
                print(colored(name, color) + f' tensor size: {vector.size()}, min: {torch.min(vector).item():.4f},  max: {torch.max(vector).item():.4f}, mean: {torch.mean(vector).item():.{precision}f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
            except:
                print(colored(name, color) + f' tensor size: {vector.size()}, min: {torch.min(vector).item():.4f},  max: {torch.max(vector).item():.4f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
    elif isinstance(vector, np.ndarray):
        try:
            print(colored(name, color) + f' numpy size: {vector.shape}, min: {np.min(vector):.4f},  max: {np.max(vector):.4f}, mean: {np.mean(vector):.4f}, dtype: {vector.dtype}')
        except:
            print(colored(name, color) + f' numpy size: {vector.shape}, min: {np.min(vector):.4f},  max: {np.max(vector):.4f}, dtype: {vector.dtype}')
    elif isinstance(vector, list):
        info(vector[0], f'{name} list of length: {len(vector)}, {name}[0]')
    else:
        print(colored(name, color) + 'Neither a torch tensor, nor a numpy array, nor a list of tensor/np array.' + f' type{type(vector)}')



def create_save_folder(folder_path, verbose=True):
    """
    1. check if there exists a folder with the same name,
    2. create folder
    3. save code in it
    :param folder_path:
    :return: none
    """
    if os.path.exists(folder_path):
        shutil.rmtree(os.path.abspath(folder_path), ignore_errors=True)
    if os.path.exists(folder_path) and 'debug' not in folder_path:
        if verbose:
            print("A folder with the same name already exists. \nPlease change a name for: " + colored(os.path.abspath(folder_path).split(os.sep)[-1], 'red'))

        print("If you wish to overwrite, please type: \"overwrite" + "\"")
        timeout = time.time() + 10
        while True:
            val = input("Type here: ")
            if val != ("overwrite"):
                print("Does not match. Please type again or exit (ctrl+c).")

            else:
                print("Removing '{:}'".format(folder_path))
                shutil.rmtree(os.path.abspath(folder_path), ignore_errors=True)
                break
            if time.time() > timeout:
                print('timed out')
                exit()
    abs_save_path = os.path.abspath(folder_path)
    if verbose:
        print("Allocating '{:}'".format(colored(abs_save_path, 'red')))
    
    if os.path.exists(abs_save_path):
        shutil.rmtree(abs_save_path, ignore_errors=True)
    os.makedirs(abs_save_path, exist_ok=True)

    # print("copying files from: ", os.getcwd())
    # print("to: ", abs_save_path + '/' + os.getcwd().split(os.sep)[-1])
    # copytree(os.getcwd(), abs_save_path + '/' + os.getcwd().split(os.sep)[-1], symlinks=True, ignore=ignore_patterns('backup', '__pycache__', 'data', 'data2', 'experiments', 'data/*', 'data2/*', 'outs', 'vis', 'wandb'))
    print("Done.")


def psnr_translation_invariant(corrected, truth, pad_margin=10, metric='psnr', istorch=True, normalize=True, verbose=False, device='cuda'):
    """

    if torch, input is [B, C, H, W];
    if numpy, input is [H, W]
    """

    if istorch:

        if normalize:
            truth = (truth - torch.min(truth)) / (torch.max(truth) - torch.min(truth))
            corrected = (corrected - torch.min(corrected)) / (torch.max(corrected) - torch.min(corrected))

        truth_padded = torchvision.transforms.Pad(pad_margin//2)(truth).float().to(device)

        best_psnr = 0
        best_h = 0
        for h in range(pad_margin):
            for w in range(pad_margin):
                corrected_padded = torchvision.transforms.Pad((w, h, pad_margin-w, pad_margin-h))(corrected).float().to(device)
                # corrected_padded = np.pad(corrected, ((h, pad_margin-h), (w, pad_margin-w)), mode='constant', constant_values=0)

                if 'psnr' in metric.lower():
                    psnr = piq_psnr(torch.clamp(corrected_padded, 0, 1), torch.clamp(truth_padded, 0, 1))
                if 'ssim' in metric.lower():
                    psnr = piq_ssim(torch.clamp(corrected_padded, 0, 1), torch.clamp(truth_padded, 0, 1))
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_h = h
        if verbose:
            print('best psnr: ', best_psnr, ', best h: ', best_h, ', total h: ', 'best')

    else:
        truth_padded = np.pad(truth, ((pad_margin//2, pad_margin//2), (pad_margin//2, pad_margin//2)), mode='constant', constant_values=0)

        if normalize:
            truth_padded = (truth_padded - np.min(truth_padded)) / (np.max(truth_padded) - np.min(truth_padded))  * 255
            corrected = (corrected - np.min(corrected)) / (np.max(corrected) - np.min(corrected)) * 255

        best_psnr = 0
        best_h = 0
        for h in range(pad_margin):
            for w in range(pad_margin):
                corrected_padded = np.pad(corrected, ((h, pad_margin-h), (w, pad_margin-w)), mode='constant', constant_values=0)

                if 'psnr' in metric.lower():
                    psnr = peak_signal_noise_ratio(corrected_padded, truth_padded)
                if 'ssim' in metric.lower():
                    psnr = structural_similarity(corrected_padded, truth_padded)

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_h = h
        if verbose:
            print('best psnr: ', best_psnr, ', best h: ', best_h, ', total h: ', 'best')

    return best_psnr


def genTarget(unpaddedWidth=16, totalWidth=16, info_ratio=0.25, device='cuda', seed=0):
    seed_torch(seed)
    DC = torch.arange(unpaddedWidth**2)
    DC = DC / len(DC)
    DC = DC < info_ratio
    DC = DC[torch.randperm(len(DC))]
    DC = DC.float().resize(unpaddedWidth, unpaddedWidth).to(device)
    DC = transforms.Pad(int((totalWidth - unpaddedWidth)/2))(DC[None, None, ...])
    return DC.squeeze(0)


def init_env():
    seed_torch(0)
    # nvmlInit()
    warnings.filterwarnings("ignore")
    os.environ["WANDB_SILENT"] = "True"
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

def cpuinfo(text=None):
    print(colored(f'{text} : CPU Percent {psutil.cpu_percent()}%, RAM Percent {psutil.virtual_memory().percent}%, Disk Percent {psutil.disk_usage(os.sep).percent}%', 'red'))
    print()

ang_to_unit = lambda x : ((x / np.pi) + 1) / 2

def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    if target_shape is None:
        return field
    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2
    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field


# https://github.com/fkodom/fft-conv-pytorch
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

    output = output[..., s2:-s2, s2:-s2]
    # output = output[:, :, s2:-s2, s2:-s2]
    return output


def normalize(x):
    '''
    normalize the max value to 1, and min value to 0
    '''
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))