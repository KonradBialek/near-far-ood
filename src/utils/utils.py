import os
import shutil
from typing import List
from warnings import warn
import numpy as np
import pandas as pd
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.transforms as transforms
import torchvision
from torchsummary import summary
from .cutout import Cutout
import pandas as pd

BATCH_SIZE = 256
writer = SummaryWriter()

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)
    
    
def runTensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs'])
    url = tb.launch()
    print(f"TensorBoard on {url}")

def updateWriter(mode: str, loss: float, acc: float, epoch: int):
    writer.add_scalar(f"loss/{mode}", loss, epoch)
    writer.add_scalar(f"acc/{mode}", acc, epoch)

def dataloader(dataset: str, size = (32, 32), rgb = False, train = False, setup = False, n_holes = 1, length = 16, normalization = [[0.5], [0.5]], batch_size = BATCH_SIZE, calcNorm = False, postprocess = False):
    '''
    Load dataset.

    Args:
        dataset (str): Name of dataset to load.
        size (int, int): Size of output images.
        rgb (bool): If images should be colorful.
        train (bool): If in train mode.
        ID (bool): If in-distribution dataset.
        n_holes (int): Number of holes to cut out from image for Cutout.
        length (int): Length of the holes for Cutout.
        normalization (list[list[float]]): Mean and standard deviation for normalization.
        batch_size (bool): Size of batch of data.
        calcNorm (bool): If dataset requested from calculateNormalization.py.
        postprocess (bool): If dataset requested from main.py in measure mode.
    '''
    valloader = testloader = None
    
    if rgb and dataset in ['mnist', 'fashionmnist', 'notmnist']: # nn must be colorful so must dataset
        convert = transforms.Lambda(lambda x: x.repeat(3,1,1))
    else:
        convert = transforms.Lambda(lambda x: x)

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
    elif calcNorm or postprocess:
        transform = transform_val = transforms.Compose([
            Convert('RGB'),
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
        ])
    else:
        # if rgb and dataset in ['mnist', 'fashionmnist', 'notmnist']: 
        #     convert = transforms.Lambda(lambda x: x.repeat(3,1,1))
        # elif not rgb and dataset in ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin']:
        #     convert = transforms.Grayscale(num_output_channels=1)
        # else:
        #     convert = transforms.Lambda(lambda x: x)

        transform = transforms.Compose([
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(normalization[0], normalization[1]),
            convert,
        ])
        transform_val = transform

    trainset, valset, testset, _ = getDataset(dataset, transform, transform_val)

    if train:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        if valset is not None:
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
        if testset is not None:
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainloader, valloader, testloader
    else:
        if setup:
            testset = trainset
        elif valset is not None:
            if testset is not None:
                testset = torch.utils.data.ConcatDataset([valset, testset])
            else:
                testset = valset
        # elif testset is not None:
        #     pass
        # else:
        #     testset = trainset
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return testloader

def getDataset(dataset: str, transform = None, transform_val = None):
    valset = testset = extraset = None
    os.makedirs('./data', exist_ok=True)
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
    
    elif dataset == 'places365':
        trainset = torchvision.datasets.Places365(root='./data', download=True, transform=transform, small=True)
        valset = torchvision.datasets.Places365(root='./data', split='val', download=True, transform=transform_val, small=True)
    
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_val)
        # extraset = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform)
        # trainset = torch.utils.data.ConcatDataset([trainset, extraset]) # no extraset
    else:
        dataset_path = f'./data/{dataset}/'
        if dataset == 'notmnist':
            trainset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
        elif dataset == 'tin':
            trainset = torchvision.datasets.ImageFolder(dataset_path + 'train', transform=transform)
            valset = processValTIN(dataset_path, transform_val) # if tiny imagenet val in raw form
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

    return trainset, valset, testset, extraset

def processValTIN(dataset_path: str, transform_val):
    if len(os.listdir(dataset_path + 'val')) != 200:
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
    # use_gpu = False # if gpu is busy
    print('GPU: '+str(use_gpu))
    return use_gpu

def showLayers(model, shape):
    if use_gpu:
        summary(model, (shape[2], shape[0], shape[1]))    


normalization_dict = {'cifar10': ([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
                      'cifar100': ([0.48042983, 0.44819681, 0.39755555], [0.2764398, 0.26888656, 0.28166855]),
                      'mnist': ([0.13062754273414612], [0.30810779333114624]),
                      'fashionmnist': ([0.28604060411453247], [0.3530242443084717]),
                      'notmnist': NotImplementedError(f'Normalization and shape of images in notmnist is not known.'),
                      'dtd': NotImplementedError(f'Normalization and shape of images in dtd is not known.'),
                      'places365': NotImplementedError(f'Normalization and shape of images in places365 is not known.'),
                      'svhn': ([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614]),
                      'tin': ([0.48023694, 0.44806704, 0.39750364], [0.27643643, 0.26886328, 0.28158993]),
}

shape_dict = {'cifar10': (32, 32, 3),
              'cifar100': (32, 32, 3),
              'svhn': (32, 32, 3),
              'mnist': (28, 28, 3),
              'fashionmnist': (28, 28, 3),
              'notmnist': (28, 28, 3),
              'dtd': (300, 300, 3),
              'places365': (256, 256, 3),
              'tin': (64, 64, 3),
}

def getShapeNormalization(dataset):
    return shape_dict[dataset], normalization_dict[dataset]

num_classes_dict = {'cifar10': 10,
                    'cifar100': 100,
                    'svhn': 10,
                    'mnist': 10,
                    'fashionmnist': 10,
                    'notmnist': 10,
                    'dtd': 47,
                    'places365': 365,
                    'tin': 1000,
}

def getNN(nn: str, dataset: str):
    model = torch.hub.load('pytorch/vision:v0.14.0', nn) 
    numFetures = num_features_dict[nn]
    numClasses = num_classes_dict[dataset]
    model.fc = torch.nn.Linear(numFetures, numClasses)
    return model

num_features_dict = {'resnet18': 512,
                    'resnet34': 512,
                    'renset50': 2048,
                    'renset101': 2048,
                    'renset152': 2048,
                    'resnext50_32x4d': 2048,
                    'resnext101_32x8d': 2048,
                    'resnext101_64x4d': 2048,
                    'wide_resnet50_2': 2048,
                    'wide_resnet101_2': 2048,
                    'densenet121': 1024,
                    'densenet161': 2208,
                    'densenet169': 1664,
                    'densenet201': 1920,
}


def saveModel(epoch: int, model, optimizer, scheduler, loss: float, checkpoints: str, nn: str, flag: int):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model,
        'optimizer_state_dict': optimizer,
        'scheduler_state_dict': scheduler,
        'loss': loss,
        }, f'{checkpoints}/model-{nn}-epoch-{epoch}{"-last" if flag == 2 else ""}-CrossEntropyLoss-{loss:.8f}{"-early_stop" if flag == 1 else ""}.pth')

def loadNNWeights(nn: str, checkpoint: str):
    path = f'./checkpoints/{checkpoint}'
    ckpt = torch.load(path)
    model = torch.hub.load('pytorch/vision:v0.14.0', nn) 
    model.fc = torch.nn.Identity()
    use_gpu = isCuda()
    if use_gpu:
        model = model.cuda()
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # print(missing_keys)
    # print(unexpected_keys)
    model.eval()
    return model

def save_scores(data, labels, save_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, save_name),
                data=data,
                labels=labels)