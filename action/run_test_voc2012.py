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

from input.load_voc2012 import VOC2012Dataset, get_dataset
from model.plmodel import OursModel

from util.util import sfcn_Q_argmax
from config.palatte import palette

def decode_segmap(pred_mask, palette=palette):
    """
    pred_mask: numpy array H x W with integer labels
    返回：PIL Image RGB
    """
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in palette.items():
        color_mask[pred_mask == label] = color
    return Image.fromarray(color_mask)

def lab_tensor_to_rgb_uint8(lab_tensor: torch.Tensor) -> np.ndarray:
    """
    将单张 LAB tensor -> H x W x 3 uint8 RGB
    输入:
      lab_tensor: torch.Tensor, shape [3, H, W], values in [0,1]
    按你提供的示例反归一化并转换为 RGB
    """
    lab = lab_tensor.cpu().detach().permute(1, 2, 0).numpy().astype(np.float32)  # [H, W, 3]
    lab_real = np.empty_like(lab, dtype=np.float32)
    lab_real[..., 0] = lab[..., 0] * 100.0
    lab_real[..., 1] = lab[..., 1] * 255.0 - 128.0
    lab_real[..., 2] = lab[..., 2] * 255.0 - 128.0

    rgb = color.lab2rgb(lab_real)  # [H, W, 3], float in [0,1]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    return rgb_uint8  # H x W x 3 uint8


def make_palette_flat(palette_dict: dict) -> list:
    """
    Convert palette mapping {idx: [r,g,b], ...} into a 768-length flat list for PIL putpalette.
    Missing indices are filled with 0.
    """
    flat = [0] * (256 * 3)
    for idx, col in palette_dict.items():
        # ensure idx in 0..255
        i = int(idx) & 0xFF
        flat[i * 3 + 0] = int(col[0])  # R
        flat[i * 3 + 1] = int(col[1])  # G
        flat[i * 3 + 2] = int(col[2])  # B
    return flat


def ensure_dirs(base='result'):
    dirs = ['image', 'label', 'spix', 'pred']
    for d in dirs:
        p = os.path.join(base, d)
        os.makedirs(p, exist_ok=True)
    return {name: os.path.join(base, name) for name in dirs}


def runtest():
    from skimage.segmentation import mark_boundaries
    from skimage.color import label2rgb

    with open(f"config/plan/{input('plan: ')}.yml", 'r') as fd:
        sparam = yaml.safe_load(fd)

    device = torch.device('cuda')

    split = input("split: ")
    test_loader = get_dataset(
        split=split,
        crop_size=(sparam['image_height'], sparam['image_width']),
        batch_size=sparam['train']['batch_size'],
        num_classes=sparam['n_classes'],
        shuffle=False,
        num_workers=sparam['train']['num_workers'],
        validation=True)

    ckpt = input('ckpt: ')
    model = OursModel.load_from_checkpoint(ckpt, sparam=sparam, device=device)
    model.eval().to(device)

    n_classes = sparam['n_classes']
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)  # overall 混淆矩阵

    out_dirs = ensure_dirs('result')
    palette_flat = make_palette_flat(palette)  # 用于 P 模式的调色板

    sampler_rng = random.Random(42)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels, idxs, Xrgb = batch  # images: Bx3xHxW (LAB in [0,1]), labels: BxH x W or Bx1xHxW
            images = images.to(device)
            labels = labels.to(device)
            Xrgb = Xrgb.to(device)

            # forward: 按模型返回顺序
            Y_pred, Q, C, G, Logit_node, Logit_spixl, Logit, Graph = model.model(images, Xrgb)
            preds = torch.argmax(Y_pred, dim=1)  # [B, H, W]

            preds_np = preds.cpu().numpy().astype(np.uint8)
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels_np = labels.squeeze(1).cpu().numpy().astype(np.int64)
            else:
                labels_np = labels.cpu().numpy().astype(np.int64)

            B = images.shape[0]

            # accumulate global confusion matrix
            for b in range(B):
                p_flat = preds_np[b].flatten()
                l_flat = labels_np[b].flatten()
                valid_mask = (l_flat >= 0) & (l_flat < n_classes)
                if valid_mask.sum() > 0:
                    cm += confusion_matrix(
                        l_flat[valid_mask],
                        p_flat[valid_mask],
                        labels=list(range(n_classes))
                    )

            # spix map batch from sfcn_Q_argmax (used only for visualization overlay)
            spix_map_batch = sfcn_Q_argmax(Q, cell_size=sparam.get('cell_size', 16),
                                           batch_unique_id=False, device=device)  # [B, H, W]

            # per-sample save and per-sample miou
            for b in range(B):
                idx = idxs[b]
                idx_str = idx.replace(os.sep, '_')
                if idx_str not in ['2009_003304', '2009_003105']:
                    if sampler_rng.random() > 0.05:
                        continue

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

                # 1) 保存原图：LAB -> RGB -> uint8
                lab_tensor = images[b].cpu()  # [3, H, W]
                rgb_uint8 = lab_tensor_to_rgb_uint8(lab_tensor)
                Image.fromarray(rgb_uint8).save(os.path.join(out_dirs['image'], base_name + ".png"))

                # 2) 保存 label（彩色 可视化 RGB）
                label_np = labels_np[b].astype(np.uint8)
                label_img = Image.fromarray(label_np, mode='P')
                label_img.putpalette(palette_flat)
                label_img.save(os.path.join(out_dirs['label'], base_name + ".png"))

                # 3) spix overlay 可视化
                sp_map = spix_map_batch[b].cpu().numpy().astype(np.int32)  # [H, W]
                img_rgb_for_sp = rgb_uint8.astype(np.float32) / 255.0
                bound = mark_boundaries(img_rgb_for_sp, sp_map, color=(1, 0, 0), mode='thick')
                bound_uint8 = (bound * 255.0).astype(np.uint8)
                Image.fromarray(bound_uint8).save(os.path.join(out_dirs['spix'], base_name + ".png"))

                # 4) 保存 pred（索引图像，用 P 模式并设置调色板）
                pred_np = preds_np[b].astype(np.uint8)  # H x W
                pred_img = Image.fromarray(pred_np, mode='P')
                pred_img.putpalette(palette_flat)
                pred_img.save(os.path.join(out_dirs['pred'], base_name + ".png"))

    # overall IoU
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    iou_per_class = intersection / np.maximum(union, 1)
    miou = np.nanmean(iou_per_class)

    print(f"\nValidation Results, checkpoint `{ckpt}`, split `{split}`")
    voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor', 'ignore'
    ]
    for i, iou in enumerate(iou_per_class):
        print(f"Class {i} {voc_classes[i]}: IoU = {iou:.4f}")
    print(f"Mean IoU: {miou:.4f}\n")
    print(f"Positive mIoU: {(iou_per_class[6] + iou_per_class[7] + iou_per_class[19]) / 3}")
