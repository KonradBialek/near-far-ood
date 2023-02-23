import os
import shutil
from typing import List
from warnings import warn
import pandas as pd
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.transforms as transforms
import torchvision
from torchsummary import summary
from cutout import Cutout
import pandas as pd
# from cutmix.cutmix import CutMix

BATCH_SIZE = 256

def runTensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs'])
    url = tb.launch()
    print(f"TensorBoard on {url}")
    global writer
    writer = SummaryWriter()

def updateWriter(mode: str, loss: float, acc: float, epoch: int):
    writer.add_scalar(f"loss/{mode}", loss, epoch)
    writer.add_scalar(f"acc/{mode}", acc, epoch)

def dataloader(dataset: str, normalization: List[List[float]], size: List[int], rgb: bool, train: bool, ID: bool, n_holes: int, length: int):
    global trainloader, valloader
    valset = testset = None
    valloader = testloader = None
    
    if rgb and dataset in ['mnist', 'fashionmnist', 'notmnist']: # sieć musi być kolorowa więc dataset tez
        convert = transforms.Lambda(lambda x: x.repeat(3,1,1))
    if train:
        transform = transforms.Compose([
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC), # for irregular datasets
            transforms.RandomCrop(size[0], padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
            Cutout(n_holes=n_holes, length=length),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
        ])
    else:
        if rgb and dataset in ['mnist', 'fashionmnist', 'notmnist']: # todo problem z zmianą kolorów
            convert = transforms.Lambda(lambda x: x.repeat(3,1,1))
        # elif not rgb and dataset in ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin']:
        #     convert = transforms.Grayscale(num_output_channels=1)
        else:
            convert = transforms.Lambda(lambda x: x)

        transform = transforms.Compose([
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
            convert,
        ])
        transform_val = transform

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)

    elif dataset == 'dtd':
        trainset = torchvision.datasets.DTD(root='./data', download=True, transform=transform)
        valset = torchvision.datasets.DTD(root='./data', split='val', download=True, transform=transform_val)
        testset = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform_val)
    
    elif dataset == 'fashionmnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_val)
    
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_val)
    
    elif dataset == 'places365': # todo small=True - 256x256 zamiast high-resolution?
        trainset = torchvision.datasets.Places365(root='./data', download=True, transform=transform)
        valset = torchvision.datasets.Places365(root='./data', split='val', download=True, transform=transform_val)
    
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_val)
        extraset = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform_val)
        trainset = torch.utils.data.ConcatDataset([trainset, extraset])
    else:
        dataset_path = f'./data/{dataset}/'
        if dataset == 'notmnist':
            trainset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
        if dataset == 'tin':
            trainset = torchvision.datasets.ImageFolder(dataset_path + 'train', transform=transform)
            valset = processValTIN(dataset_path, transform_val)
        else:
            trainset = torchvision.datasets.ImageFolder(dataset_path + 'train', transform=transform)
            try:
                valset = torchvision.datasets.ImageFolder(dataset_path + 'val', transform=transform_val)
            except:
                warn('No validation set.')
            try:
                testset = torchvision.datasets.ImageFolder(dataset_path + 'test', transform=transform_val)
            except:
                warn('No test set.')

    if train:
        # trainset = CutMix(trainset, num_class=getNumClasses(dataset), beta=1.0, prob=0.25, num_mix=2)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        if valset is not None:
            valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
        if testset is not None:
            testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
        return trainloader, valloader, testloader
    else:
        if ID:
            pass
        elif valset is not None:
            if testset is not None:
                testset = torch.utils.data.ConcatDataset([trainset, valset, testset])
            else:
                testset = torch.utils.data.ConcatDataset([trainset, valset])
        elif testset is not None:
            testset = torch.utils.data.ConcatDataset([trainset, testset])
        else:
            testset = trainset
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
        return testloader


def processValTIN(dataset_path: str, transform_val):
    dir_list = next(os.walk(dataset_path + 'val'))
    if len(dir_list)[1] != 0 or len(dir_list)[2] != 200:
        df = pd.read_csv(dataset_path + 'val/val_annotations.txt', delimiter='\t')
        for dir_ in next(os.walk(dataset_path + 'train'))[1]:
            os.makedirs(dataset_path + 'val/' + dir_, exist_ok=True)
        for _, row in df.iterrows():
            shutil.move(f'{dataset_path}val/images/{row[0]}', f'{dataset_path}val/{row[1]}/{row[0]}')
        
        shutil.move(f'{dataset_path}val/val_annotations.txt', f'{dataset_path}val/images/val_annotations.txt')
        shutil.move(f'{dataset_path}val/images', dataset_path)  

    return torchvision.datasets.ImageFolder(dataset_path + 'val', transform=transform_val)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def isCuda():
    global use_gpu
    use_gpu = torch.cuda.is_available()
    print('GPU: '+str(use_gpu))
    return use_gpu
    # return False

def showLayers(model, shape):
    if use_gpu:
        summary(model, (shape[2], shape[0], shape[1]))    


def shapeNormalization(dataset: str, train_ID=False):
    if dataset == 'cifar10': # todo generalize
        shape = 32, 32, 3
        if train_ID:
            normalization = [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768] # mean, std
        else:
            raise NotImplementedError(f'Normalization for entire {dataset} is not known.')
    elif dataset == 'cifar10':
        shape = 32, 32, 3
        if train_ID:
            normalization = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761] # mean, std
        else:
            raise NotImplementedError(f'Normalization for entire {dataset} is not known.')
    elif dataset == 'mnist':
        shape = 28, 28, 1
        if train_ID:
            normalization = [0.13062754273414612], [0.30810779333114624] # mean, std
        else:
            raise NotImplementedError(f'Normalization for entire {dataset} is not known.')
    else:
        shape = 64, 64, 3
        if train_ID:
            normalization = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        # raise NotImplementedError(f'Normalization and shape of images in {dataset} is not known.')

    return shape, normalization

def getNumClasses(dataset: str):
    if dataset in ['cifar10', 'mnist', 'Fashionmnist', 'notmnist', 'svhn']:
        return 10
    elif dataset == 'dtd':
        return 47
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'places365':
        return 365
    elif dataset == 'tin':
        return 1000
    else:
        raise NotImplementedError(f'Number of classes in {dataset} is not known.')

def getNN(nn: str, dataset: str):
    model = torch.hub.load('pytorch/vision:v0.14.0', nn) 

    numClasses = getNumClasses(dataset)

    # if dataset in ['mnist', 'fashionmnist', 'notmnist'] and nn in ['resnet18', 'resnet34', 'renset50', 'renset101', 'renset152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2']:
    #     model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # elif dataset in ['mnist', 'fashionmnist', 'notmnist'] and nn in ['densenet121', 'densenet169', 'densenet201']:
    #     print(model.features[0])
    #     model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # error
    #     print(model.features[0])
    # elif dataset in ['mnist', 'fashionmnist', 'notmnist'] and nn == 'densenet161':
    #     model.features[0] = torch.nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False) # error

    if nn in ['resnet18', 'resnet34']:
        model.fc = torch.nn.Linear(512, numClasses)
    elif nn in ['renset50', 'renset101', 'renset152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2']:
        model.fc = torch.nn.Linear(2048, numClasses)
    elif nn == 'densenet121':
        model.fc = torch.nn.Linear(1024, numClasses)
    elif nn == 'densenet161':
        model.fc = torch.nn.Linear(2208, numClasses)
    elif nn == 'densenet169':
        model.fc = torch.nn.Linear(1664, numClasses)
    elif nn == 'densenet201':
        model.fc = torch.nn.Linear(1920, numClasses)
    else:
        raise NotImplementedError(f'Network {nn} is not known.')
    
    return model

def getNumFeatures(nn: str):
    model = torch.hub.load('pytorch/vision:v0.14.0', nn) 

    if nn in ['resnet18', 'resnet34']:
        return 512
    elif nn in ['renset50', 'renset101', 'renset152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2']:
        return 2048
    elif nn == 'densenet121':
        return 1024
    elif nn == 'densenet161':
        return 2208
    elif nn == 'densenet169':
        return 1664
    elif nn == 'densenet201':
        return 1920
    else:
        raise NotImplementedError(f'Network {nn} is not known.')
    
    return model

def touchCSV(path: str, headers):
    with open(path, 'w') as file:
        file.writelines(",".join(headers) + "\n")

def generateHeaders(num_headers: int):
    headers = []  
    for header in range(1,num_headers + 1):
        headers.append(f"{header}_feature")
    
    headers.append("class")
    return headers

def saveModel(epoch: int, model, optimizer, loss: float, checkpoints: str, nn: str, flag: int):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model,
        'optimizer_state_dict': optimizer,
        'loss': loss,
        }, f'{checkpoints}/model-{nn}-epoch-{epoch}{"-last" if flag == 2 else ""}-CrossEntropyLoss-{loss:.8f}{"-early_stop" if flag == 1 else ""}.pth')
