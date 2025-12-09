import lightning as pl

from input.load_cityscapes import get_dataset as get_cityscapes
from model.plmodel import OursModel

import config.header


def train(sparam, device):
    # voc_train_loader = get_voc(
    #     split='train',
    #     crop_size=(sparam['image_height'], sparam['image_width']),
    #     batch_size=sparam['train']['batch_size'],
    #     num_classes=sparam['n_classes'],
    #     shuffle=True,
    #     num_workers=sparam['train']['num_workers'])
    
    train_loader = get_cityscapes(
        split='train',
        crop_size=(sparam['image_height'], sparam['image_width']),
        batch_size=sparam['train']['batch_size'],
        num_classes=sparam['n_classes'],
        shuffle=True,
        num_workers=sparam['train']['num_workers'],
        validation=False)

    val_loader = get_cityscapes(
        split='val',
        crop_size=(sparam['image_height'], sparam['image_width']),
        batch_size=sparam['train']['batch_size'],
        num_classes=sparam['n_classes'],
        shuffle=False,
        num_workers=sparam['train']['num_workers'],
        validation=True)

    # train_loader = get_dataset(
    #     split='train',
    #     crop_size=(sparam['image_height'], sparam['image_width']),
    #     batch_size=sparam['train']['batch_size'],
    #     num_classes=sparam['n_classes'],
    #     shuffle=True,
    #     num_workers=sparam['train']['num_workers'],
    #     validation=False)

    # val_loader = get_dataset(
    #     split='val',
    #     crop_size=(sparam['image_height'], sparam['image_width']),
    #     batch_size=sparam['train']['batch_size'],
    #     num_classes=sparam['n_classes'],
    #     shuffle=False,
    #     num_workers=sparam['train']['num_workers'],
    #     validation=True)
    
    ckpt = sparam['train'].get('checkpoint', None)
    if ckpt is not None:
        model = OursModel.load_from_checkpoint(ckpt, sparam=sparam, device=device)
    else:
        model = OursModel(sparam, device)

    trainer = pl.Trainer(
        max_epochs=sparam['train']['max_epochs'],
        limit_train_batches=sparam['train'].get('limit_train_batches', 1.0),
        limit_val_batches=sparam['train'].get('limit_val_batches', 1.0),
        # precision=16,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt)

    # common_trainer_kwargs = dict(
    #     limit_train_batches=sparam['train'].get('limit_train_batches', 1.0),
    #     limit_val_batches=sparam['train'].get('limit_val_batches', 1.0),
    #     accumulate_grad_batches=sparam['train'].get('accumulate_grad_batches', 1),
    # )

    # trainer_stage1 = pl.Trainer(max_epochs=sparam['train']['pre_train_epoch'], **common_trainer_kwargs)
    # trainer_stage1.fit(model=model, train_dataloaders=voc_train_loader, val_dataloaders=val_loader)

    # trainer_stage2 = pl.Trainer(max_epochs=sparam['train']['max_epochs'], logger=trainer_stage1.logger, **common_trainer_kwargs)
    # trainer_stage2.fit(model=model, train_dataloaders=join_train_loader, val_dataloaders=val_loader)

