import time
import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LinearLR
from torchmetrics import JaccardIndex
from PIL import Image
from torchvision import transforms as T
from lightning.pytorch import Trainer
from argparse import ArgumentParser
from util.util import sfcn_Q_argmax
from skimage.segmentation import mark_boundaries
import numpy as np
from torchvision.utils import make_grid

from model.model import get_model
from loss.loss import Loss
from config.palatte import palette


class OursModel(pl.LightningModule):
    def __init__(self, sparam, device):
        super().__init__()
        self.sparam = sparam
        # core model and loss
        self.model, self.model_sfcn, self.model_encoder, self.model_gnn, self.model_decoder = get_model(sparam, device)
        self.loss_fn = Loss(
            lambda_main=sparam['loss']['lambda_main'],
            lambda_spix=sparam['loss']['lambda_spix'],
            lambda_sfcn=sparam['loss']['lambda_sfcn'],
            cell_size=sparam['cell_size'],
            class_weight=sparam['loss'].get('weight', None),
            device=device,
            ignore_index=sparam['loss'].get('ignore_index', 255),
            pix_loss_type=sparam['loss'].get('pix_loss_type', None),
        )

        # metrics
        num_classes = sparam['n_classes']
        ignore_idx = sparam['loss'].get('ignore_index', 255)

        # per-class IoU: average=None makes compute() return a tensor of shape [num_classes]
        self.train_iou = JaccardIndex(task='multiclass', num_classes=num_classes, average=None, ignore_index=ignore_idx)
        self.val_iou   = JaccardIndex(task='multiclass', num_classes=num_classes, average=None, ignore_index=ignore_idx)


    def on_train_epoch_start(self):
        # record start time
        self._epoch_start_time = time.time()


    def training_step(self, batch, batch_idx):
        X, Y, idx, Xrgb = batch
        Y_pred, Q, C, G, Ln, Lsp, L, BG = self.model(X, Xrgb)
        l_total, l_pix, l_spix, loss_sem_pos = self.loss_fn(Q, C, G, Ln, Lsp, L, BG, Y)

        # log train loss (step/epoch)
        self.log('train_step_loss', l_total, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_loss', l_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({'train_loss/pixel_ce': l_pix, 'train_loss/spixl_kl': l_spix, 'train_loss/sfcn_sem_poss': loss_sem_pos},
                      on_step=False, on_epoch=True, prog_bar=False)

        # log lr (safe access)
        try:
            opt = self.trainer.optimizers[0]
            current_lr = opt.param_groups[0]['lr']
            self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=False)
        except Exception:
            # trainer not attached or optimizer not ready
            pass

        # preds: convert logits -> labels, ensure long dtype
        preds = torch.argmax(Y_pred, dim=1).long()
        targets = Y.long()

        # update metric only (do NOT compute/reset here)
        self.train_iou.update(preds, targets)

        return l_total


    def on_train_epoch_end(self):
        # compute per-class IoU: tensor shape [num_classes]
        per_class = self.train_iou.compute()  # tensor on device
        # ensure on cpu and python floats for logging/backends
        per_class_cpu = per_class.detach().cpu().numpy()
        mean_iou = float(per_class_cpu.mean())

        # log mean IoU to prog_bar
        self.log('train_miou', mean_iou, prog_bar=True)

        # # log per-class IoU (as separate keys). If too many classes, user may opt to log subset.
        # per_class_dict = {f'train_iou/class_{i}': float(per_class_cpu[i]) for i in range(per_class_cpu.shape[0])}
        # self.log_dict(per_class_dict, prog_bar=False)
        if per_class_cpu.shape[0] == 21:
            self.log_dict({
                'train_positive_miou': (per_class_cpu[6] + per_class_cpu[7] + per_class_cpu[19]) / 3,
                'train_positive_miou/bus': per_class_cpu[6],
                'train_positive_miou/car': per_class_cpu[7],
                'train_positive_miou/train': per_class_cpu[19],
            })
        elif per_class_cpu.shape[0] == 19:
            self.log_dict({
                'train_positive_miou': (per_class_cpu[15] + per_class_cpu[13] + per_class_cpu[16]) / 3,
                'train_positive_miou/bus': per_class_cpu[15],
                'train_positive_miou/car': per_class_cpu[13],
                'train_positive_miou/train': per_class_cpu[16],
            })
        else:
            pass

        # reset metric for next epoch
        self.train_iou.reset()

        duration = time.time() - getattr(self, '_epoch_start_time', time.time())
        self.log('epoch_time', duration, prog_bar=True)


    # def on_after_backward(self):
    #     # log grad norm
    #     total_norm = torch.norm(
    #         torch.stack([
    #             p.grad.detach().norm(2)
    #             for p in self.parameters() if p.grad is not None
    #         ]),
    #         p=2
    #     )
    #     self.log('grad_norm', total_norm, on_step=True, on_epoch=False, prog_bar=False)

    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             grad_norm = param.grad.detach().norm(2)
    #             # careful: too many logged keys may be heavy; keep if you need them
    #             self.log(f'grad_norm/{name}', grad_norm, on_step=True, on_epoch=True, prog_bar=False)


    def validation_step(self, batch, batch_idx):
        X, Y, idx, Xrgb = batch
        Y_pred, Q, C, G, Ln, Lsp, L, BG = self.model(X, Xrgb)
        l_total, l_pix, l_spix, loss_sem_pos = self.loss_fn(Q, C, G, Ln, Lsp, L, BG, Y)

        # compute predicted mask and label
        preds = torch.argmax(Y_pred, dim=1).long()
        targets = Y.long()

        # update IoU metric (do not compute/reset here)
        self.val_iou.update(preds, targets)

        # log val loss components (epoch aggregation)
        self.log('val_loss', l_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({'val_loss/pixel_ce': l_pix, 'val_loss/spixl_kl': l_spix, 'val_loss/sfcn_sem_poss': loss_sem_pos},
                      on_step=False, on_epoch=True, prog_bar=False)

        return {'val_loss': l_total}


    def on_validation_epoch_end(self):
        per_class = self.val_iou.compute()
        per_class_cpu = per_class.detach().cpu().numpy()
        mean_iou = float(per_class_cpu.mean())

        # log mean IoU to prog_bar
        self.log('val_miou', mean_iou, prog_bar=True)

        # # log per-class IoU
        # per_class_dict = {f'val_iou/class_{i}': float(per_class_cpu[i]) for i in range(per_class_cpu.shape[0])}
        # self.log_dict(per_class_dict, prog_bar=False)
        if per_class_cpu.shape[0] == 21:
            self.log_dict({
                'val_positive_miou': (per_class_cpu[6] + per_class_cpu[7] + per_class_cpu[19]) / 3,
                'val_positive_miou/bus': per_class_cpu[6],
                'val_positive_miou/car': per_class_cpu[7],
                'val_positive_miou/train': per_class_cpu[19],
            })
        elif per_class_cpu.shape[0] == 19:
            self.log_dict({
                'val_positive_miou': (per_class_cpu[15] + per_class_cpu[13] + per_class_cpu[16]) / 3,
                'val_positive_miou/bus': per_class_cpu[15],
                'val_positive_miou/car': per_class_cpu[13],
                'val_positive_miou/train': per_class_cpu[16],
            })
        else:
            pass

        # reset for next epoch
        self.val_iou.reset()


    def configure_optimizers(self):
        base_lr = self.sparam['train']['lr']
        wd = self.sparam['train']['weight_decay']

        sfcn_params = [p for p in self.model_sfcn.parameters() if p.requires_grad]
        sfcn_param_ids = {id(p) for p in sfcn_params}
        other_params = [p for p in self.parameters() if p.requires_grad and id(p) not in sfcn_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": base_lr, "weight_decay": wd},
                {"params": sfcn_params, "lr": base_lr * 0.1, "weight_decay": wd},
            ]
        )

        warmup_steps = self.sparam['train']['lr_schedule']['warmup_steps']
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        optimizers = [optimizer]

        lr_cfg = self.sparam['train']['lr_schedule']
        if lr_cfg['type'] == 'cosine':
            batch_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=lr_cfg['t_max'],
                eta_min=lr_cfg.get('eta_min', 0),
                last_epoch=lr_cfg.get('last_epoch', -1)
            )

            schedulers = [
                {'scheduler': warmup_scheduler, 'interval': 'step', 'frequency': 1},
                {'scheduler': batch_scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_loss'}
            ]

            return optimizers, schedulers

        elif lr_cfg['type'] == 'step':
            batch_scheduler = MultiStepLR(
                optimizer,
                milestones=lr_cfg['milsestone'],
                gamma=lr_cfg['gamma']
            )

            schedulers = [
                {'scheduler': warmup_scheduler, 'interval': 'step', 'frequency': 1},
                {'scheduler': batch_scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_loss'}
            ]

            return optimizers, schedulers

        else:
            schedulers = [
                {'scheduler': warmup_scheduler, 'interval': 'step', 'frequency': 1}
            ]

            return optimizers, schedulers

    def configure_callbacks(self):
        return [ModelCheckpoint(
            every_n_epochs=1,
            save_top_k=200,
            monitor='val_miou',
            filename='{epoch:04d}',
        )]
