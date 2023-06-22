#!/bin/bash
# sh scripts/ood/msp/mnist_test_ood_msp.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
--num_workers 8 \
--network.checkpoint 'results/checkpoints/best-mnist-lenet.ckpt' \
--mark 0

# configs/networks/resnet18_32x32.yml \
# --network.checkpoint 'results/checkpoints/best-mnist-resnet18.ckpt' \
