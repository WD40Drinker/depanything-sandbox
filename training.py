import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import random
import h5py

import sys
sys.path.append('/Depth-Anything-V2/metric_depth')

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import notebook_launcher
from accelerate import DistributedDataParallelKwargs

import transformers

import torch
import torchvision
from torchvision.transforms import v2
from torchvision.transforms import Compose
import torch.nn.functional as F
import albumentations as A

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.util.loss import SiLogLoss
from metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


train_paths = get_all_files(r'C:\Users\mitch\Documents\GitHub\depanything-sandbox\Training-Dir\Depth-Anything-V2\nyudepthv2\train')
val_paths   = get_all_files(r'C:\Users\mitch\Documents\GitHub\depanything-sandbox\Training-Dir\Depth-Anything-V2\nyudepthv2\val')

#NYU Depth V2 40k. Original NYU is 400k
class NYU(torch.utils.data.Dataset):
    def __init__(self, paths, mode, size=(518, 518)):
        
        self.mode = mode #train or val
        self.size = size
        self.paths = paths
        
        net_w, net_h = size
        #author's transforms
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
        # only horizontal flip in the paper
        self.augs = A.Compose([
            A.HorizontalFlip(),
            A.ColorJitter(hue = 0.1, contrast=0.1, brightness=0.1, saturation=0.1),
            A.GaussNoise(var_limit=25),
#             A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.1, rotate_limit=7, interpolation=cv2.INTER_CUBIC, border_mode=0)
        ])
    
    def __getitem__(self, item):
        path = self.paths[item]
        image, depth = self.h5_loader(path)
        
        if self.mode == 'train':
            augmented = self.augs(image=image, mask = depth)
            image = augmented["image"] / 255.0
            depth = augmented['mask']
        else:
            image = image / 255.0

          
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # sometimes there are masks for valid depths in datasets because of noise e.t.c
#         sample['valid_mask'] = ...

#         sample['image_path'] = path
        
        return sample

    def __len__(self):
        return len(self.paths)
    
    def h5_loader(self, path):
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
        return rgb, depth
    
def get_dataloaders(batch_size):
    
    train_dataset = NYU(train_paths, mode='train')
    val_dataset = NYU(val_paths, mode='val')
    
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True
#                                                   pin_memory=True
                                                  )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size = 1, #for dynamic resolution evaluations without padding
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True
#                                                 pin_memory=True
                                                )
    
    return train_dataloader, val_dataloader

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))

    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.detach(), 'abs_rel': abs_rel.detach(),'rmse': rmse.detach(), 'mae': mae.detach(), 'silog':silog.detach()}

model_weights_path =  'depth_anything_v2_vits.pth' 
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
model_encoder = 'vits'
max_depth = 10
batch_size = 2
lr = 5e-6
weight_decay = 0.01
num_epochs = 10
warmup_epochs = 0.5
scheduler_rate = 1
load_state = False

state_path      = r'C:\Users\mitch\Documents\GitHub\depanything-sandbox\Training-Dir\Depth-Anything-V2\checkpoints'
save_model_path = r'C:\Users\mitch\Documents\GitHub\depanything-sandbox\Training-Dir\Depth-Anything-V2\model.pth'
seed = 42
mixed_precision = 'fp16'

def train_fn():

    set_seed(seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # need for freezing layers with DDP
    accelerator = Accelerator(mixed_precision=mixed_precision, 
                              kwargs_handlers=[ddp_kwargs],
                             )
    
    # in the paper they initialize decoder randomly and use only encoder pretrained weights. Then full fine-tune
    # ViT-S encoder here
    model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth})
    model.load_state_dict({k: v for k, v in torch.load(model_weights_path).items() if 'pretrained' in k}, strict=False)
    
    optim = torch.optim.AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': lr*10}],
                      lr=lr, weight_decay=weight_decay)
    
    criterion = SiLogLoss() # author's loss
    
    train_dataloader, val_dataloader = get_dataloaders(batch_size)
    
    scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))
    
    model, optim, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optim, train_dataloader, val_dataloader, scheduler)
    
    
    if load_state:
        accelerator.wait_for_everyone()
        accelerator.load_state(state_path)
        
    best_val_absrel = 1000
    
    
    for epoch in range(1, num_epochs):
        
        model.train()
        train_loss = 0
        for sample in tqdm(train_dataloader, disable = not accelerator.is_local_main_process):
            optim.zero_grad()
            
            img, depth = sample['image'], sample['depth']
            
            pred = model(img) 
                                                     # mask
            loss = criterion(pred, depth, (depth <= max_depth) & (depth >= 0.001))
            
            accelerator.backward(loss)
            optim.step()
            scheduler.step()
            
            train_loss += loss.detach()
            
            
            
        train_loss /= len(train_dataloader)
        train_loss = accelerator.reduce(train_loss, reduction='mean').item()
        
        
        model.eval()
        results = {'d1': 0, 'abs_rel': 0,'rmse': 0, 'mae': 0, 'silog': 0}
        for sample in tqdm(val_dataloader, disable = not accelerator.is_local_main_process):
            
            img, depth = sample['image'].float(), sample['depth'][0]
            
            with torch.no_grad():
                pred = model(img)
                # evaluate on the original resolution 
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (depth <= max_depth) & (depth >= 0.001)
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            

        for k in results.keys():
            results[k] = results[k] / len(val_dataloader)
            results[k] = round(accelerator.reduce(results[k], reduction='mean').item(),3)
        
        
        accelerator.wait_for_everyone()
        accelerator.save_state(state_path, safe_serialization=False)
        
        if results['abs_rel'] < best_val_absrel:
            best_val_absrel = results['abs_rel']
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_local_main_process:
                torch.save(unwrapped_model.state_dict(), save_model_path)
        
        accelerator.print(f"epoch_{epoch},  train_loss = {train_loss:.5f}, val_metrics = {results}")
        
#P.S. While testing one configuration, I encountered an error in which the loss turned into nan. 
# This is fixed by adding a small epsilon to the predictions to prevent division by 0

print("Script loaded, num_workers will be 0")
#You can run this code with 1 gpu. Just set num_processes=1
if __name__ == '__main__':
    notebook_launcher(train_fn, num_processes=1)
# ignore the error. it's harmless