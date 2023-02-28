# python ./src/main.py --nn resnet18 --mode train --train_dataset cifar10 --length 16 --n_holes 1
python ./src/main.py --nn resnet18 --mode extract --in_dataset cifar10 --ood_datasets mnist cifar100 --checkpoint model-resnet18-epoch-1-CrossEntropyLoss-1.38263078.pth
# python ./src/main.py --nn resnet18 --mode measure --feature_datasets cifar10 cifar100 mnist --method knn --method_args 50 --checkpoint model-resnet18-epoch-174-CrossEntropyLoss-0.64874127.pth
# python ./src/main.py --nn resnet18 --mode measure --feature_datasets cifar10 cifar100 --method odin --checkpoint model-resnet18-epoch-174-CrossEntropyLoss-0.64874127.pth 
# python ./src/main.py --nn resnet18 --checkpoint model-resnet18-epoch-174-CrossEntropyLoss-0.64874127.pth --mode extract --in_dataset cifar10 --ood_datasets mnist cifar100
# python ./src/main.py --nn densenet121 --checkpoint densenet10.pth --mode extract --in_dataset cifar10
