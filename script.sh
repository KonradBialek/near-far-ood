# python ./src/calculateNormalization.py --dataset notmnist --mode train
# python ./src/main.py --nn resnet18 --mode train --train_dataset cifar10 --length 16 --n_holes 1
# python ./src/main.py --nn resnet18 --mode extract --process_datasets cifar10 cifar100 mnist fashionmnist svhn tin --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth
# python ./src/main.py --nn resnet18 --mode measure --method msp --method_args 90 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth --process_datasets cifar10 dtd
# python ./src/main.py --nn resnet18 --mode measure --method msp --method_args 90 --checkpoint resnet-18.t7 --process_datasets tin cifar100 cifar10 mnist notmnist fashionmnist dtd svhn
# python ./src/main.py --nn resnet18 --mode measure --method msp --method_args 90 --checkpoint models/resnet18_svhn/pytorch_model.bin --process_datasets svhn tin cifar100 cifar10 dtd 
# python ./src/main.py --nn densenet121 --mode measure --method msp --method_args 90 --checkpoint models/densenet121_cifar10/pytorch_model.bin --process_datasets cifar10 svhn tin cifar100 dtd 
# python ./src/main.py --nn resnet18 --mode measure --method msp --method_args 90 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth --process_datasets cifar10 notmnist
# python ./src/main.py --nn resnet18 --mode measure --method react --method_args 90 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth --process_datasets cifar10 svhn tin cifar100 dtd mnist fashionmnist notmnist

# python ./src/main.py --nn resnet18 --mode measure --method msp --method_args 50 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth --process_datasets cifar10 svhn
# python ./src/main.py --nn resnet18 --mode measure --method msp --method_args 50 --checkpoint best-cifar10-resnet18.ckpt
# python ./src/main.py --nn resnet18 --mode measure --method msp --method_args 50 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 svhn
# python ./src/main.py --nn resnet18 --mode measure --method mds --method_args 1.4e-3 mean 1 none --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 svhn
# python ./src/main.py --nn resnet18 --mode measure --method msp --checkpoint cifar10_res18_acc94.30.ckpt --process_datasets cifar10 svhn
# python ./src/main.py --nn resnet18 --mode measure --method msp --checkpoint 2_model-resnet18-epoch-169-CrossEntropyLoss-0.95970000.pth --process_datasets cifar10 svhn



#########################################################################################
python ./src/main.py --nn resnet18 --mode measure --method msp --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn resnet18 --mode measure --method mls --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn resnet18 --mode measure --method react --method_args 90 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn resnet18 --mode measure --method lof --method_args 20 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn resnet18 --mode measure --method knn --method_args 50 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none true --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none false --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 true cifar10 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 false cifar10 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365


# python ./src/main.py --nn resnet18 --mode measure --method msp --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method mls --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method react --method_args 90 --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method lof --method_args 20 --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method knn --method_args 50 --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none true --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none false --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 true mnist --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 false mnist --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365


# python ./src/main.py --nn resnet18 --mode measure+ --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
# python ./src/main.py --nn resnet18 --mode measure+ --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
#########################################################################################





# python ./src/main.py --nn resnet18 --mode measure --method knn --method_args 50
# python ./src/main.py --nn resnet18 --mode measure --method knn --method_args 50 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth
# python ./src/main.py --nn resnet18 --mode measure --process_datasets cifar10 cifar100 mnist --method odin --method_args 1000 1.4e-3 --checkpoint model-resnet18-epoch-105-CrossEntropyLoss-0.36705533.pth
# python ./src/reduceDim.py

