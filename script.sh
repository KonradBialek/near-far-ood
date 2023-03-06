# python ./src/calculateNormalization.py --dataset mnist --mode train
python ./src/main.py --nn resnet18 --mode train --train_dataset cifar10 --length 16 --n_holes 1
python ./src/main.py --nn resnet18 --mode extract --process_datasets cifar10 mnist cifar100 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth
python ./src/main.py --nn resnet18 --mode measure --process_datasets cifar10 cifar100 mnist --method knn --method_args 50 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth
python ./src/main.py --nn resnet18 --mode measure --process_datasets cifar10 cifar100 mnist --method odin --method_args 1000 1.4e-3 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth
python ./src/reduceDim.py

