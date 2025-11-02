# Implementation for [One Step Learning, One Step Review](https://arxiv.org/abs/2401.10962)
---

## Key features

We implemented OLOR for SGD and Adam. You can find **SGD-OLOR** in [OLOR/utils/SGDB.py](https://github.com/xiaol827/OLOR-AAAI-2024/blob/main/OLOR/utils/SGDB.py), and Adam-OLOR in [OLOR/utils/AdamB.py](https://github.com/xiaol827/OLOR-AAAI-2024/blob/main/OLOR/utils/AdamB.py).

---
## Data preprocess
Before starting, please download the following datasets:

[Cifar-100](https://www.cs.toronto.edu/~kriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/), [CUB-200](https://www.vision.caltech.edu/datasets/cub_200_2011/), [StanfordCars](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset), [Places-LT](https://opendatalab.com/OpenDataLab/Places-LT), [IP102](https://github.com/xpwu95/IP102), [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html), [PACS](https://sketchx.eecs.qmul.ac.uk/downloads/).

Once the raw data are downloaded, you can preprocess them for training using the provided **Data_Preprocess.ipynb** script.

## Quik training

```
CUDA_VISIBEL_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
/root/OLOR/train.py \
--finetune-mode AdamB \
--model-type vit \
--csv-dir ./CIFAR100/Cifar_100_train_10fold.csv \
--config-name 'config' \
--image-size 224 \
--epochs 50 \
--init-lr 1e-4 \
--batch-size 128 \
--num-workers 6 \
--nbatch_log 300 \
--warmup_epochs 0 \
--val_fold 0
```
 

## Quik test

```
!python ./OLOR/test.py \
--image-size 224 \
--csv-dir './CIFAR100/Cifar_100_test.csv' \
--model-path [to be provided] \
```


