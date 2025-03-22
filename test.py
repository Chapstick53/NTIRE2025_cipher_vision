import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
# Use dynamic import for numeric filenames
import importlib.util


# OR use dynamic import (better for competition submission):
import importlib.util

# Define model file path
model_file = "models/36_Pureformer.py"
model_name = "Pureformer36"  # Arbitrary valid Python module name

# Load the module dynamically
spec = importlib.util.spec_from_file_location(model_name, model_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Get the Restormer class
Restormer = getattr(module, "Restormer")  # Case-sensitive class name

import lightning.pytorch as pl
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_default_tensor_type(torch.cuda.FloatTensor)



class NoiseReductorPureformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Restormer()
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        prediction_ = torch.zeros_like(x)  # Initialize cumulative prediction
        
        for k in range(4):  # Rotate 0, 90, 180, 270 degrees
            rotated_tensor = torch.rot90(x, k=k+1, dims=(2, 3))  # Rotate input tensor
            rotated_prediction = self.net(rotated_tensor)  # Forward pass
            prediction_ += torch.rot90(rotated_prediction, k=-(k+1), dims=(2, 3))  # Rotate back

        return prediction_ / 4  # Average the ensemble predictions
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.optimizer.param_groups[0]['lr']

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]


def pad_to_multiple(image, multiple=16):
    """Pads the image to make height and width multiples of `multiple`."""
    h, w = image.shape[2], image.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # Use reflection padding to avoid border artifacts
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_image, (pad_h, pad_w)

def pad_to_size(image, target_height, target_width):
    """Pads an image to the target height and width."""
    current_height, current_width = image.shape[1], image.shape[2]
    pad_height = target_height - current_height
    pad_width = target_width - current_width
    
    padding = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
    return F.pad(image, padding, mode='constant', value=0)  # Zero padding

def collate_fn(batch):
    # Simply stack images without padding
    clean_names, images, labels = zip(*batch)
    return list(clean_names), torch.stack(images, 0), torch.stack(labels, 0)

def test_Denoise(net, dataset, sigma=15, is_pre_noisy=False):
    output_path = testopt.output_path
    os.makedirs(output_path, exist_ok=True)
    
    if not is_pre_noisy:
        dataset.set_sigma(sigma)

    testloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    patch_size = 512

    with torch.no_grad():
        for (clean_name, degrad_patch, _) in tqdm(testloader):  # Ignore clean_patch
            degrad_patch = degrad_patch.cuda(non_blocking=True)
            B, C, H, W = degrad_patch.shape

            # Split into patches
            patches = []
            coords = []
            for i in range(0, H, patch_size):
                i_end = min(i + patch_size, H)
                for j in range(0, W, patch_size):
                    j_end = min(j + patch_size, W)
                    patch = degrad_patch[:, :, i:i_end, j:j_end]
                    patches.append(patch)
                    coords.append((i, i_end, j, j_end))

            # Process patches
            restored_patches = [net(p.half()).float() for p in patches]

            # Stitch patches
            restored = torch.zeros(B, C, H, W, device=degrad_patch.device)
            for idx, (i, i_end, j, j_end) in enumerate(coords):
                restored[:, :, i:i_end, j:j_end] = restored_patches[idx]
            
            # Save image
            filename = clean_name[0][0] if isinstance(clean_name[0], (list, tuple)) else clean_name[0]
            save_image_tensor(restored, os.path.join(output_path, f"{filename}.png"))

            del degrad_patch, restored
            torch.cuda.empty_cache()
            
    print(f"Denoising completed. Results saved to {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--denoise_path', type=str, default="data/input/noisy", help='path to noisy images')
    parser.add_argument('--output_path', type=str, default="results/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="36_Pureformer.ckpt", help='checkpoint path')
    testopt = parser.parse_args()
    
    # Initialize dataset with add_synthetic_noise=False
    denoise_testset = DenoiseTestDataset(testopt, add_synthetic_noise=False)
    
    # Load model and test
    ckpt_path = "model_zoo/" + testopt.ckpt_name
    net = NoiseReductorPureformer.load_from_checkpoint(ckpt_path, map_location="cuda").eval().half().cuda()
    
    print('Processing noisy images...')
    test_Denoise(net, denoise_testset, is_pre_noisy=True)