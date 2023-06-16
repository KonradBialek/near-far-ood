# ResNet-18
# CIFAR-10
python ./src/main.py --nn resnet18 --mode measure --method msp --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method mls --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method react --method_args 90 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method lof --method_args 20 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method knn --method_args 50 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none true --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none false --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 true cifar10 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 false cifar10 --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
########################################################################################

# MNIST
python ./src/main.py --nn resnet18 --mode measure --method msp --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method mls --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method react --method_args 90 --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method lof --method_args 20 --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method knn --method_args 50 --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none true --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method mds --method_args  1.4e-3 mean 1 none false --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 true mnist --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn resnet18 --mode measure --method odin --method_args 1000 1.4e-3 false mnist --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
########################################################################################

python ./src/main.py --nn resnet18 --mode extract --checkpoint best-cifar10-resnet18.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn resnet18 --mode extract --checkpoint best-mnist-resnet18.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
python ./src/reduceDim.py
########################################################################################

# LeNet-5
# CIFAR-10
python ./src/main.py --nn lenet --mode measure --method msp --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method mls --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method react --method_args 90 --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method lof --method_args 20 --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method knn --method_args 50 --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method mds --method_args  1.4e-3 mean 1 none true --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method mds --method_args  1.4e-3 mean 1 none false --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method odin --method_args 1000 1.4e-3 true cifar10 --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
python ./src/main.py --nn lenet --mode measure --method odin --method_args 1000 1.4e-3 false cifar10 --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 tin places365 dtd svhn mnist fashionmnist notmnist 
########################################################################################

# MNIST
python ./src/main.py --nn lenet --mode measure --method msp --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method mls --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method react --method_args 90 --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method lof --method_args 20 --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method knn --method_args 50 --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method mds --method_args  1.4e-3 mean 1 none true --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method mds --method_args  1.4e-3 mean 1 none false --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method odin --method_args 1000 1.4e-3 true mnist --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
python ./src/main.py --nn lenet --mode measure --method odin --method_args 1000 1.4e-3 false mnist --checkpoint best-mnist-lenet.ckpt --process_datasets mnist fashionmnist notmnist cifar10 cifar100 tin places365 dtd svhn 
########################################################################################

python ./src/main.py --nn lenet --mode extract --checkpoint best-cifar10-lenet.ckpt --process_datasets cifar10 cifar100 dtd svhn tin mnist fashionmnist notmnist places365
python ./src/main.py --nn lenet --mode extract --checkpoint best-mnist-lenet.ckpt --process_datasets mnist cifar100 dtd svhn tin cifar10 fashionmnist notmnist places365
python ./src/reduceDim.py

