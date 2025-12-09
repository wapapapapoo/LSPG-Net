import math
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as TT
import random
from skimage import color

import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.25, 0.25],
                           [0.25, 0.25]], dtype=torch.float32)
        self.register_buffer("ll_kernel", ll[None, None, :, :])  # [1,1,2,2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        kernel = self.ll_kernel.repeat(C, 1, 1, 1).to(x.device)  # [C,1,2,2]
        out = F.conv2d(x, kernel, bias=None, stride=2, padding=0, groups=C)
        return out

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', crop_size=(512, 1024), num_classes=19, transform=True, limit_dataset=None, validation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.validation = validation
        self.split = split
        self.crop_size = crop_size

        self.dwt_forward = HaarDWT()

        if split == 'train':
            list_path = f"{root_dir}/trainImages.txt"
            with open(list_path, 'r') as f:
                self.image_ids = ['_'.join((line.strip().split('/')[3]).split('_')[:3]) for line in f]
        elif split == 'val':
            list_path = f"{root_dir}/valImages.txt"
            with open(list_path, 'r') as f:
                self.image_ids = ['_'.join((line.strip().split('/')[3]).split('_')[:3]) for line in f]
        
        # if split == 'train':
        #     new_list = []
        #     for img_id in self.image_ids:
        #         new_list.append(img_id)
        #         label_path = f"{self.root_dir}/gtFine/{self.split}/{img_id.split('_')[0]}/{img_id}_gtFine_labelTrainIds.png"
        #         label = np.array(Image.open(label_path), dtype=np.uint8)
        #         if (label == 15).any():
        #             for _ in range(9):
        #                 new_list.append(img_id)
        #         if (label == 16).any():
        #             for _ in range(19):
        #                 new_list.append(img_id)
        #     self.image_ids = new_list


    def __len__(self):
        return len(self.image_ids)

    def dwt(self, img_t: torch.Tensor) -> torch.Tensor:
        """
        img_t: [3, H, W] float32 tensor on GPU
        return: [3, H/2, W/2]
        """
        img_t = img_t.unsqueeze(0) # [B, C, H, W]
        LL = self.dwt_forward(img_t) # [B, C, H/2, W/2]
        return LL[0] # [C, H/2, W/2]

    def dwt_downsample(self, img_t, rate: int):
        times = rate.bit_length() - 1
        for _ in range(times):
            img_t = self.dwt(img_t)
        return img_t

    def label_downsample(self, label_t: torch.Tensor, rate: int):
        off_y = random.randint(0, rate - 1)
        off_x = random.randint(0, rate - 1)
        return label_t[off_y::rate, off_x::rate]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = f"{self.root_dir}/leftImg8bit/{self.split}/{img_id.split('_')[0]}/{img_id}_leftImg8bit.png"
        label_path = f"{self.root_dir}/gtFine/{self.split}/{img_id.split('_')[0]}/{img_id}_gtFine_labelTrainIds.png"

        # Load image in LAB mode
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path) # mode = 'P'

        # Apply transforms
        if self.transform:
            image, label = self._transform(image, label)
        
        # # random corp downsample 2x
        # if self.crop_size == (1024, 2048):
        #     pass
        # else:
        #     if random.random() > 0.5 and self.validation == False:
        #         i, j, h, w = TT.RandomCrop.get_params(image, output_size=(512, 1024))
        #         image = TF.crop(image, i, j, h, w)
        #         label = TF.crop(label, i, j, h, w)
        #     else:
        #         image = TF.center_crop(image, (512, 1024))
        #         label = TF.center_crop(label, (512, 1024))

        # To tensor
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # [3, H, W], RGB
        label_np = np.array(label, dtype=np.int64)  # [H, W]
        label = torch.from_numpy(label_np).long()  # [H, W]

        image[0, :, :] /= 255.0
        image[1, :, :] /= 255.0
        image[2, :, :] /= 255.0

        # if self.crop_size == (1024, 2048):
        #     pass
        # elif self.crop_size == (512, 1024):
        #     # image = self.dwt_downsample(image, 2)
        #     # label = self.label_downsample(label, 2)
        #     pass
        # elif self.crop_size == (256, 512):
        #     image = self.dwt_downsample(image, 2)
        #     label = self.label_downsample(label, 2)
        # elif self.crop_size == (128, 256):
        #     image = self.dwt_downsample(image, 4)
        #     label = self.label_downsample(label, 4)
        # elif self.crop_size == (64, 128):
        #     image = self.dwt_downsample(image, 8)
        #     label = self.label_downsample(label, 8)
        # else:
        #     raise NotImplementedError(f'corp size unsupport')
        
        rgb_image = image

        image = np.array(image.permute(1, 2, 0).contiguous().float())
        image = color.rgb2lab(image)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image[0, :, :] /= 100.0
        image[1, :, :] = (image[1, :, :] + 128.0) / 255.0
        image[2, :, :] = (image[2, :, :] + 128.0) / 255.0

        if np.all(label_np == 255): # redo
            return self.__getitem__(random.randint(0, len(self.image_ids) - 1))

        return image, label, self.image_ids[idx], rgb_image



    def _transform(self, image, label):
        th, tw = self.crop_size

        # ----- 1) multi-scale (只做几何，不做过滤)
        scale = random.uniform(0.25, 0.5)

        new_w = int(image.width * scale)
        new_h = int(image.height * scale)

        image = image.resize((new_w, new_h), Image.BICUBIC)
        label = label.resize((new_w, new_h), Image.NEAREST)

        new_h = th
        new_w = tw

        # ----- 2) random crop
        if new_h > th and new_w > tw:
            i, j, h_crop, w_crop = TT.RandomCrop.get_params(image, (th, tw))
            image = TF.crop(image, i, j, h_crop, w_crop)
            label = TF.crop(label, i, j, h_crop, w_crop)
        else:
            # 如果不够大，pad 再 crop
            pad_h = max(th - new_h, 0)
            pad_w = max(tw - new_w, 0)
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
            label = TF.pad(label, (0, 0, pad_w, pad_h), fill=255)
            image = image.crop((0, 0, tw, th))
            label = label.crop((0, 0, tw, th))

        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        if random.random() > 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))
        if random.random() > 0.8:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.95, 1.05))
        if random.random() > 0.8:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.95, 1.05))

        return image, label


def get_dataset(split='train', crop_size=(256, 256), batch_size=8, num_classes=21, shuffle=True, num_workers=4, limit_dataset=None, validation=False):
    prefix = "data/input/dataset/cityscapes"
    dataset = CityscapesDataset(prefix, split, crop_size, num_classes, transform=True, limit_dataset=limit_dataset, validation=validation) # (split in ['trainval', 'val']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, persistent_workers=True)
    return dataloader
