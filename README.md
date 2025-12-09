# LSPG-Net PyTorch Impl

## prepare

1. Install requirements in `@/requirements.txt`

2. Download SpixelFCN weight from [here](https://github.com/fuy34/superpixel_fcn/blob/master/pretrain_ckpt/SpixelNet_bsd_ckpt.tar), and put it into `@/data/input/weight/`

3. Download datasets, extract them into `@/data/input/dataset/`, the final directory tree like:

```
@:
    data:
        input:
            dataset:
                VOCdevkit: # extract voc2012 here as is
                    ...
                cityscapes:
                    gtFine:
                        train:
                            ...
                        val:
                            ...
                        test:
                            ...
                    leftImg8bit:
                        train:
                            ...
                        val:
                            ...
                        test:
                            ...
            weight:
                SpixelNet_bsd_ckpt.tar
    ...
```

then excute:

```bash
cd @/data/input/dataset/cityscapes
find leftImg8bit/train -type f > trainImages.txt
find leftImg8bit/val -type f > valImages.txt
```

## train

```bash
cd @
echo "voc2012" | python train_voc2012.py
echo "cityscapes" | python train_cityscapes.py
```

The checkpoints and metrics will save in the `@/lightning_logs/`, you can view it by tensorboard:

```bash
tensorboard --logdir @/lightning_logs/
```

## test

Run `test_voc2012.py` or `test_cityscapes.py`. The scripts will require for version, split and checkpoint, like:

```bash
printf "voc2012\nval\n@/lightning_logs/version_1/checkpoints/epoch=0099.ckpt\n" | python test_voc2012.py
```
