from torch.utils.data import DataLoader, Dataset
import os
from utils import *
from PIL import Image, ImageOps
from re import L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from utils import info, seed_torch
import torchvision
import scipy.io as sio


def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


class ImageDataset(Dataset):
    def __init__(self, data_folder, input_transforms=None, has_subfolders=False, gray2rgb=True):
        super(ImageDataset, self).__init__()
        if has_subfolders:
            pass
        else:
            self.image_filenames = [os.path.join(data_folder, x) for x in os.listdir(data_folder) if is_image_file(x)]
        self.input_transforms = input_transforms
        self.grayscale2RGB = gray2rgb

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index])

        if self.grayscale2RGB:
            img = img.convert('RGB')
        if self.input_transforms:
            img = self.input_transforms(img)
        return img

    def __len__(self):
        return len(self.image_filenames)



class AlphabetDataset(Dataset):
    def __init__(self, data_folder, input_transforms=None, has_subfolders=False, gray2rgb=True):
        super(ImageDataset, self).__init__()
        if has_subfolders:
            pass

        self.input_transforms = input_transforms
        self.grayscale2RGB = gray2rgb

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index])

        if self.grayscale2RGB:
            img = img.convert('RGB')

        if self.input_transforms:
            img = self.input_transforms(img)

        return img

    def __len__(self):
        return len(self.image_filenames)


class OCRDataset(Dataset):
    def __init__(self, data_folder, input_transforms=None, has_subfolders=False, gray2rgb=True):
        super(ImageDataset, self).__init__()
        if has_subfolders:
            pass

        self.input_transforms = input_transforms
        self.grayscale2RGB = gray2rgb

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index])

        if self.grayscale2RGB:
            img = img.convert('RGB')

        if self.input_transforms:
            img = self.input_transforms(img)

        return img

    def __len__(self):
        return len(self.image_filenames)



class BatchDataset_v2(torch.utils.data.Dataset):
    def __init__(self, data_dir, begin_idx=1, width=128, im_prefix='SLM_raw', slm_prefix='SLM_sim', num=100, max_intensity=0, zero_freq=-1):
        self.data_dir = data_dir
        self.zero_freq = zero_freq
        self.width = width
        self.height = int(width / 16 * 9)
        a_slm = np.ones((self.height, self.width))
        a_slm = np.lib.pad(a_slm, (((self.width - self.height) // 2, (self.width - self.height) // 2), (0, 0)), 'constant', constant_values=(0, 0))
        self.a_slm = torch.from_numpy(a_slm).type(torch.float)
        self.max_intensity = max_intensity
        self.num = num
        self.begin_idx = begin_idx
        self.im_prefix, self.slm_prefix = im_prefix, slm_prefix
        self.load_in_cache()
        self.num = len(self.xs)
        print(f'Training with {self.num} frames.')

    def load_in_cache(self):
        x_list, y_list = [], []
        for idx in range(self.num):
            img_name = f'{self.data_dir}/{self.im_prefix}{idx+self.begin_idx}.mat'
            mat_name = f'{self.data_dir}/{self.slm_prefix}{idx+self.begin_idx}.mat'

            p_SLM = sio.loadmat(f'{mat_name}')
            p_SLM = p_SLM['proj_sim']   #(72, 128)
            p_SLM = np.lib.pad(p_SLM, (((self.width - self.height) // 2, (self.width - self.height) // 2), (0, 0)), 'constant', constant_values=(0, 0))
            p_SLM_train = torch.FloatTensor(p_SLM).unsqueeze(0)

            if self.zero_freq > 0 and idx % self.zero_freq == 0:
                p_SLM_train = torch.zeros_like(p_SLM_train)
                img_name = f'{self.data_dir}/../Zero/{self.im_prefix}{idx+1}.mat'
                print(f'#{idx} uses zero SLM')

            x_train = self.a_slm * torch.exp(1j * -p_SLM_train)
            ims = sio.loadmat(f'{img_name}')
            y_train = ims['imsdata']
            if np.max(y_train) > self.max_intensity:    self.max_intensity = np.max(y_train)
            y_train = torch.FloatTensor(y_train)
            x_list.append(x_train); y_list.append(y_train)

        y_list = [y / self.max_intensity for y in y_list]
        self.xs, self.ys = x_list, y_list

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx


class FFN(Dataset):
    def __init__(self, data_dir, input_transforms=None, nframe=100, return_fname=False, 
                 test_idx_set=None, use_transforms=False, just_save_images=False, vis_dir=None, train=True,
                 use_modulation=True, child_folder = ''):

        self.data_dir = data_dir
        self.nframe = nframe
        self.return_fname = return_fname
        self.transforms = input_transforms
        self.use_transforms = use_transforms
        self.just_save_images = just_save_images
        self.vis_dir = vis_dir

        self.use_modulation = use_modulation
        self.child_folder = child_folder


        idx_set = set() 
        for fname in os.listdir(f'{data_dir}/zero'):
            if 'zero' not in fname:
                fidx = fname.split('.')[0][3:]

                if test_idx_set is None:
                    idx_set.add(int(fidx))
                else:
                    if int(fidx) in test_idx_set:
                        idx_set.add(int(fidx)) 
        self.data = list(idx_set)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        gt_path = os.path.join(f'{self.data_dir}/GT', f'ims {self.data[index]}.png')
        if os.path.isfile(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt = cv2.imread(os.path.join(f'{self.data_dir}/GT', f'ims {self.data[index]}.png'), cv2.IMREAD_GRAYSCALE)
    
        gt = torch.tensor(gt.astype(np.float32) / 255.0)

        zero = sio.loadmat(os.path.join(f'{self.data_dir}/zero', f'raw{self.data[index]}.mat'))['imsdata']
        zero = torch.tensor(zero.astype(np.float32) / 1.0)

        if 'zero' not in self.child_folder.lower():

            try:
                samples = sio.loadmat(os.path.join(f'{self.data_dir}/{self.child_folder}', f'raw{self.data[index]}.mat'))['imsdata']  
            except:
                print(colored(os.path.join(f'{self.data_dir}/{self.child_folder}', f'raw{self.data[index]}.mat'), 'red'))

            samples = torch.tensor(samples.astype(np.float32) / 1.0)

            samples = samples[:self.nframe] 

            samples = torch.cat([zero.unsqueeze(0), samples], dim=0)    

        else:
            samples = zero.unsqueeze(0)

        crop_size = gt.size()[-1] // 16 * 16
        
        gt = torchvision.transforms.CenterCrop(crop_size)(gt)
        samples = torchvision.transforms.CenterCrop(crop_size)(samples)
        zero = torchvision.transforms.CenterCrop(crop_size)(zero)

        if self.use_transforms and self.transforms is not None:
            gt = self.transforms(gt.unsqueeze(0)).squeeze(0).contiguous()
            samples = self.transforms(samples).contiguous()


        if self.just_save_images:
            if not os.path.exists(f'{self.vis_dir}/data{index}'):
                os.makedirs(f'{self.vis_dir}/data{index}')
            save_image(samples[0], f'{self.vis_dir}/data{index}/unmodulated.png')   
            for i in range(len(samples)-1):
                save_image(samples[i], f'{self.vis_dir}/data{index}/modulated_{i+1}.png')
            return 0
        

        if self.return_fname:
            return gt.unsqueeze(0), samples.unsqueeze(1), self.data[index]


        return gt.unsqueeze(0), samples.unsqueeze(1)



class FFN_Sim(Dataset):
    def __init__(self, data_folder, input_transforms=None, has_subfolders=True, gray2rgb=False, rgb2gray=True, device='cuda'):
        super(FFN_Sim, self).__init__()
        if has_subfolders:
            _, self.data = run_fast_scandir(data_folder, ['.jpg', '.png', 'tiff', '.jpeg'])
            # self.data = glob.glob(data_folder + '/**/*.jpg', recursive=True)
            # https://stackoverflow.com/questions/18394147/how-to-do-a-recursive-sub-folder-search-and-return-files-in-a-list
        else:
            self.data = [os.path.join(data_folder, x) for x in os.listdir(data_folder) if is_image_file(x)]
        self.input_transforms = input_transforms
        self.grayscale2RGB = gray2rgb
        self.rgb2gray = rgb2gray
        self.device = device

    def __getitem__(self, index):
        img = Image.open(self.data[index])

        if self.grayscale2RGB:
            img = img.convert('RGB')
        if self.rgb2gray:
            img = ImageOps.grayscale(img)

        if self.input_transforms:
            img = self.input_transforms(img)
        
        return img

    def __len__(self):
        return len(self.data)

