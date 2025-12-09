import os
import random
from skimage import color
import torch
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix

from input.load_cityscapes import CityscapesDataset, get_dataset
from model.plmodel import OursModel

from util.util import sfcn_Q_argmax

cityscapes_palette = {
    0:  [128, 64,128],   # road
    1:  [244, 35,232],   # sidewalk
    2:  [70,  70, 70],   # building
    3:  [102,102,156],   # wall
    4:  [190,153,153],   # fence
    5:  [153,153,153],   # pole
    6:  [250,170, 30],   # traffic light
    7:  [220,220,  0],   # traffic sign
    8:  [107,142, 35],   # vegetation
    9:  [152,251,152],   # terrain
    10: [70, 130,180],   # sky
    11: [220, 20, 60],   # person
    12: [255,  0,  0],   # rider
    13: [0,   0, 142],   # car
    14: [0,   0,  70],   # truck
    15: [0,  60,100],    # bus
    16: [0,  80,100],    # train
    17: [0,   0,230],    # motorcycle
    18: [119, 11, 32],   # bicycle
}


def decode_segmap(pred_mask, palette=cityscapes_palette):
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in palette.items():
        color_mask[pred_mask == label] = color
    return Image.fromarray(color_mask)


def lab_tensor_to_rgb_uint8(lab_tensor: torch.Tensor) -> np.ndarray:
    lab = lab_tensor.cpu().detach().permute(1, 2, 0).numpy().astype(np.float32)
    lab_real = np.empty_like(lab, dtype=np.float32)
    lab_real[..., 0] = lab[..., 0] * 100.0
    lab_real[..., 1] = lab[..., 1] * 255.0 - 128.0
    lab_real[..., 2] = lab[..., 2] * 255.0 - 128.0

    rgb = color.lab2rgb(lab_real)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    return rgb_uint8


def make_palette_flat(palette_dict: dict) -> list:
    flat = [0] * (256 * 3)
    for idx, col in palette_dict.items():
        i = int(idx) & 0xFF
        flat[i * 3 + 0] = int(col[0])
        flat[i * 3 + 1] = int(col[1])
        flat[i * 3 + 2] = int(col[2])
    return flat


def ensure_dirs(base='result'):
    dirs = ['image', 'label', 'spix', 'pred']
    for d in dirs:
        p = os.path.join(base, d)
        os.makedirs(p, exist_ok=True)
    return {name: os.path.join(base, name) for name in dirs}


def runtest():
    from skimage.segmentation import mark_boundaries

    with open(f"config/plan/{input('plan: ')}.yml", 'r') as fd:
        sparam = yaml.safe_load(fd)

    device = torch.device('cuda')

    split = input("split (val/test): ")

    # ---------------------------------- #
    #         Cityscapes Dataset
    # ---------------------------------- #
    test_loader = get_dataset(
        split=split,
        crop_size=(sparam['image_height'], sparam['image_width']),
        batch_size=sparam['train']['batch_size'],
        num_classes=sparam['n_classes'],  # must be 19
        shuffle=False,
        num_workers=sparam['train']['num_workers'],
        validation=True
    )

    ckpt = input('ckpt: ')
    model = OursModel.load_from_checkpoint(ckpt, sparam=sparam, device=device)
    model.eval().to(device)

    n_classes = sparam['n_classes']   # 19
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    out_dirs = ensure_dirs('result')
    palette_flat = make_palette_flat(cityscapes_palette)

    sampler_rng = random.Random(42)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels, idxs, Xrgb = batch

            images = images.to(device)
            labels = labels.to(device)
            Xrgb = Xrgb.to(device)

            Y_pred, Q, C, G, Logit_node, Logit_spixl, Logit, Graph = model.model(images, Xrgb)
            preds = torch.argmax(Y_pred, dim=1)

            preds_np = preds.cpu().numpy().astype(np.uint8)
            labels_np = labels.cpu().numpy().astype(np.int64)

            B = images.shape[0]

            # 忽略标签 255
            for b in range(B):
                p_flat = preds_np[b].flatten()
                l_flat = labels_np[b].flatten()

                valid_mask = (l_flat >= 0) & (l_flat < n_classes)

                cm += confusion_matrix(
                    l_flat[valid_mask],
                    p_flat[valid_mask],
                    labels=list(range(n_classes))
                )

            spix_map_batch = sfcn_Q_argmax(
                Q,
                cell_size=sparam.get('cell_size', 16),
                batch_unique_id=False,
                device=device
            )

            for b in range(B):
                if sampler_rng.random() > 0.05:
                    continue

                idx = idxs[b]
                idx_str = idx.replace(os.sep, '_')

                p_flat = preds_np[b].flatten()
                l_flat = labels_np[b].flatten()
                valid_mask = (l_flat >= 0) & (l_flat < n_classes)

                if valid_mask.sum() > 0:
                    cm_img = confusion_matrix(
                        l_flat[valid_mask],
                        p_flat[valid_mask],
                        labels=list(range(n_classes))
                    )
                    intersection_img = np.diag(cm_img)
                    union_img = cm_img.sum(axis=1) + cm_img.sum(axis=0) - np.diag(cm_img)
                    iou_per_class_img = intersection_img / np.maximum(union_img, 1)
                    miou_img = float(np.nanmean(iou_per_class_img))
                else:
                    miou_img = 0.0

                miou_str = f"{miou_img:.4f}"
                base_name = f"{miou_str}_{idx_str}"

                # 原图
                lab_tensor = images[b].cpu()
                rgb_uint8 = lab_tensor_to_rgb_uint8(lab_tensor)
                Image.fromarray(rgb_uint8).save(os.path.join(out_dirs['image'], base_name + ".png"))

                # GT label
                label_np = labels_np[b].astype(np.uint8)
                label_img = Image.fromarray(label_np, mode='P')
                label_img.putpalette(palette_flat)
                label_img.save(os.path.join(out_dirs['label'], base_name + ".png"))

                # spix overlay
                sp_map = spix_map_batch[b].cpu().numpy().astype(np.int32)
                img_rgb_for_sp = rgb_uint8.astype(np.float32) / 255.0
                bound = mark_boundaries(img_rgb_for_sp, sp_map, color=(1, 0, 0), mode='thick')
                bound_uint8 = (bound * 255.0).astype(np.uint8)
                Image.fromarray(bound_uint8).save(os.path.join(out_dirs['spix'], base_name + ".png"))

                # pred
                pred_np = preds_np[b].astype(np.uint8)
                pred_img = Image.fromarray(pred_np, mode='P')
                pred_img.putpalette(palette_flat)
                pred_img.save(os.path.join(out_dirs['pred'], base_name + ".png"))

    # 全局 MIoU
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    iou_per_class = intersection / np.maximum(union, 1)
    miou = np.nanmean(iou_per_class)

    print(f"\nCityscapes Validation, checkpoint `{ckpt}`, split `{split}`")

    cityscapes_classes = [
        'road','sidewalk','building','wall','fence','pole',
        'traffic light','traffic sign','vegetation','terrain',
        'sky','person','rider','car','truck','bus','train',
        'motorcycle','bicycle'
    ]

    for i, iou in enumerate(iou_per_class):
        print(f"Class {i} {cityscapes_classes[i]}: IoU = {iou:.4f}")

    print(f"Mean IoU: {miou:.4f}\n")
    print(f"Positive mIoU: {(iou_per_class[15] + iou_per_class[13] + iou_per_class[16]) / 3}")
