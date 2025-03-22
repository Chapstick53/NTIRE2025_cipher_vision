import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision.transforms import ToTensor

    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        self.de_type = random.sample(self.de_type, len(self.de_type)) if isinstance(self.de_type, list) else self.de_type


    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
    

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
        if hasattr(self, "s25_ids"):  # ✅ Only add if it exists
            self.sample_ids += self.s25_ids
        if hasattr(self, "s50_ids"):  # ✅ Only add if it exists
            self.sample_ids += self.s50_ids

        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            #print("clean_id: ", clean_id)
            #img = Image.open(clean_id).convert('RGB')
            #print(img.size())           
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args, add_synthetic_noise=False, sigma=50):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = sigma
        self.add_synthetic_noise = add_synthetic_noise
        self._init_clean_ids()
        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = sorted(os.listdir(self.args.denoise_path))
        self.clean_ids = [os.path.join(self.args.denoise_path, fn) 
                        for fn in name_list if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.num_clean = len(self.clean_ids)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, index):
        clean_path = self.clean_ids[index]
        clean_name = os.path.splitext(os.path.basename(clean_path))[0]

        if self.add_synthetic_noise:
            clean_img = np.array(Image.open(clean_path).convert('RGB'))
            noise = np.random.normal(0, self.sigma, clean_img.shape).astype(np.float32)
            noisy_img = np.clip(clean_img + noise, 0, 255).astype(np.uint8)
            clean_img = self.toTensor(clean_img)
        else:
            # Load the noisy image directly; clean_img is a dummy tensor
            noisy_img = np.array(Image.open(clean_path).convert('RGB'))
            clean_img = np.zeros_like(noisy_img)  # Dummy clean image
            clean_img = self.toTensor(clean_img)

        noisy_img = self.toTensor(noisy_img)
        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean

# =========================================== PRE-NOISY IMAGE DATASET LOADER ================================================== #
# class DenoiseTestDataset(Dataset):
#     def __init__(self, args, add_synthetic_noise=True, sigma=50):  # Set default sigma=50
#         super(DenoiseTestDataset, self).__init__()
#         self.args = args
#         self.clean_ids = []
#         self.sigma = sigma  # Set to target sigma
#         self.add_synthetic_noise = add_synthetic_noise
        
#         self._init_clean_ids()
#         self.toTensor = ToTensor()

#     def _init_clean_ids(self):
#         name_list = sorted(os.listdir(self.args.denoise_path))
#         self.clean_ids = [os.path.join(self.args.denoise_path, fn) 
#                         for fn in name_list if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         self.num_clean = len(self.clean_ids)

#     def set_sigma(self, sigma):
#         self.sigma = sigma

#     def __getitem__(self, index):
#         clean_path = self.clean_ids[index]
#         clean_img = np.array(Image.open(clean_path).convert('RGB'))
#         clean_name = os.path.splitext(os.path.basename(clean_path))[0]

#         if self.add_synthetic_noise:
#             # Generate noise with current sigma (50)
#             noise = np.random.normal(0, self.sigma, clean_img.shape).astype(np.float32)
#             noisy_img = np.clip(clean_img + noise, 0, 255).astype(np.uint8)
#         else:
#             # For pre-noised images (ensure you have sigma_50 directory)
#             base_dir = os.path.dirname(os.path.dirname(clean_path))
#             noisy_dir = os.path.join(base_dir, "noisy", f"sigma_{self.sigma}")
#             noisy_path = os.path.join(noisy_dir, os.path.basename(clean_path))
#             noisy_img = np.array(Image.open(noisy_path).convert('RGB'))

#         clean_img = self.toTensor(clean_img)
#         noisy_img = self.toTensor(noisy_img)
        
#         return [clean_name], noisy_img, clean_img

#     def __len__(self):
#         return self.num_clean



class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    

