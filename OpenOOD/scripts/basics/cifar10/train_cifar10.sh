#!/bin/bash
# sh scripts/basics/cifar10/train_cifar10.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/preprocessors/base_preprocessor.yml \
configs/networks/lenet.yml \
configs/pipelines/train/baseline.yml \

# configs/networks/resnet18_32x32.yml \
