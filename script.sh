python ./code/main.py --nn resnet18 --mode train --train_dataset cifar10 --length 16 --n_holes 1
# python ./code/main.py --nn resnet18 --checkpoint model-resnet18-epoch-1-CrossEntropyLoss-1.19569108.pth --mode measure --in_dataset cifar10
# python ./code_kopia/main.py --nn resnet18 --checkpoint model-resnet18-epoch-174-CrossEntropyLoss-0.64874127.pth --mode extract --in_dataset cifar10 --ood_datasets mnist cifar100
# python ./code/main.py --nn densenet121 --checkpoint densenet10.pth --mode extract --in_dataset cifar10
