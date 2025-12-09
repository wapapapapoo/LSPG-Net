from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as TT
import random
from skimage import color

class VOC2012Dataset(Dataset):
    def __init__(self, root_dir, split='train', crop_size=(256, 256), num_classes=21, transform=True, limit_dataset=None, validation=False):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.transform = transform
        self.num_classes = num_classes
        self.validation = validation

        list_path = f"{root_dir}/ImageSets/Segmentation/{split}.txt"
        with open(list_path, 'r') as f:
            self.image_ids = [line.strip() for line in f]
        if limit_dataset != None:
            self.image_ids = self.image_ids[:limit_dataset]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = f"{self.root_dir}/JPEGImages/{img_id}.jpg"
        label_path = f"{self.root_dir}/SegmentationClass/{img_id}.png"

        # Load image in LAB mode
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)  # mode = 'P'

        # Apply transforms
        if self.transform:
            image, label = self._transform(image, label)

        image_rgb = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image_rgb[0, :, :] /= 255.
        image_rgb[1, :, :] /= 255.
        image_rgb[2, :, :] /= 255.
        
        # Convert to LAB
        image = color.rgb2lab(image)
        image[:, :, 0] /= 100.0
        image[:, :, 1] = (image[:, :, 1] + 128) / 255.0
        image[:, :, 2] = (image[:, :, 2] + 128) / 255.0

        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # [3, H, W], L,A,B channels
        # image = TF.normalize(image, mean=[.485,.456,.406], std=[.229,.224,.225]) # for RGB
        # image = TF.normalize(
        #     image,
        #     mean=[0.46436887979507446, 0.3918650150299072, 0.3235301375389099],
        #     std=[0.26780685782432556, 0.46369469165802, 0.4196985363960266])

        # Convert label to numpy array and one-hot encode
        label_np = np.array(label, dtype=np.int64)  # [H, W]
        label_tensor = torch.from_numpy(label_np).long()  # [H, W]

        if np.all(label_np == 255): # redo
            return self.__getitem__(random.randint(0, len(self.image_ids) - 1))

        return image, label_tensor, self.image_ids[idx], image_rgb

    def _transform(self, image, label):
        if not self.validation:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            # Random rotate
            # angle = random.uniform(-5, 5)
            # image = TF.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)
            # label = TF.rotate(label, angle, interpolation=Image.NEAREST, fill=255)
            # gaussian blur
            if random.random() > 0.8:
                self.blur = TT.GaussianBlur(kernel_size=5, sigma=(0.5,1.5))
            # Random brightness / contrast
            if random.random() > 0.8:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.8:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.8:
                w, h = image.size
                scale_w = random.uniform(0.8, 1.2)
                scale_h = random.uniform(0.8, 1.2)
                new_w = max(1, int(w * scale_w))
                new_h = max(1, int(h * scale_h))
                image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
                label = TF.resize(label, (new_h, new_w), interpolation=Image.NEAREST)

        th, tw = self.crop_size
        w, h = image.size  # pil image (width, height)

        # pad to th, tw
        pad_w = max(tw - w, 0)
        pad_h = max(th - h, 0)
        pad_l = int(random.random() * pad_w)
        pad_r = pad_w - pad_l
        pad_t = int(random.random() * pad_h)
        pad_b = pad_h - pad_t
        # pad argument: left, top, right, bottom
        if pad_w > 0 or pad_h > 0:
            image = TF.pad(image, (pad_l, pad_t, pad_r, pad_b), fill=0)
            label = TF.pad(label, (pad_l, pad_t, pad_r, pad_b), fill=255)  # 255 为 ignore_index

        if not self.validation:
            # Random crop
            i, j, h, w = TT.RandomCrop.get_params(image, output_size=self.crop_size)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
        else:
            image = TF.center_crop(image, self.crop_size)
            label = TF.center_crop(label, self.crop_size)

        return image, label


def get_dataset(split='train', crop_size=(256, 256), batch_size=8, num_classes=21, shuffle=True, num_workers=4, limit_dataset=None, validation=False):
    prefix = "data/input/dataset/VOCdevkit/VOC2012"
    dataset = VOC2012Dataset(prefix, split, crop_size, num_classes, transform=True, limit_dataset=limit_dataset, validation=validation) # (split in ['trainval', 'val']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, persistent_workers=True)
    return dataloader
