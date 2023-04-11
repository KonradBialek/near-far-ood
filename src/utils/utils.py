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
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
from torchsummary import summary

from utils.preprocessors.test_preprocessor import TestStandardPreProcessor
from utils.preprocessors.utils import get_preprocessor

from utils.networks.utils import get_network
from .cutout import Cutout
import pandas as pd
from torch.utils.data import DataLoader
import torchvision

BATCH_SIZE = 256
writer = SummaryWriter()

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)
    

ultimate_layer_methods = ['msp', 'mls', 'odin']
penultimate_layer_methods = ['knn', 'react', 'lof']
# undefined = ['lof', 'mahalanobis']

def bothLayers(method: str):
    if method in ultimate_layer_methods:
        return False
    elif method in penultimate_layer_methods:
        return True
    else:
        raise NotImplementedError(f'Method is not known.')

def runTensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs'])
    url = tb.launch()
    print(f"TensorBoard on {url}")

def updateWriter(mode: str, loss: float, acc: float, epoch: int):
    writer.add_scalar(f"loss/{mode}", loss, epoch)
    writer.add_scalar(f"acc/{mode}", acc, epoch)

def dataloader(dataset: str or List[str], size = (32, 32), train = False, setup = False, n_holes = 1, length = 16, normalization = [[0.5], [0.5]], batch_size = BATCH_SIZE, postprocess = False):
    '''
    Load dataset.

    Args:
        dataset (str): Name of dataset to load.
        size (int, int): Size of output images.
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
    
    if train:
        transform = transforms.Compose([
            Convert('RGB'),
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC), # for irregular datasets
            transforms.RandomCrop(size[0], padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
            Cutout(n_holes=n_holes, length=length),
        ])
        transform_val = transforms.Compose([
            Convert('RGB'),
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
        ])
    elif postprocess:
        transform = transform_val = transforms.Compose([
            Convert('RGB'),
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
        ])

    if isinstance(dataset, str):
        trainset, valset, testset, _ = getDataset(dataset, transform, transform_val)
    elif isinstance(dataset, list):
        testset_ = []
        for dataset_ in dataset:
            target = -1 if dataset_ == dataset[0] else 1
            trainset, valset, testset, _ = getDataset(dataset_, transform, transform_val, target_transform=transforms.Lambda(lambda x: target))
            if dataset_ == 'dtd':
                testset = torch.utils.data.ConcatDataset([trainset, valset, testset])
            elif dataset_ == 'notmnist':
                testset = torch.utils.data.ConcatDataset([trainset, valset])
                half_size = len(testset)//2
                testset = torch.utils.data.Subset(testset, range(half_size, len(testset)))
            if testset is None:
                testset = valset
            testset_.append(testset)
        testset = torch.utils.data.ConcatDataset(testset_) # it changes -1 to 1 for ID part of dataset
        testset = tuple((testset[i][0], -1) if i < contamination_dict[dataset[0]] else testset[i] for i in range(len(testset)))

    if train:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        if valset is not None:
            valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        if testset is not None:
            testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainloader, valloader, testloader
    else:
        if setup:
            testset = trainset
        elif valset is not None:
            if testset is not None:
                testset = torch.utils.data.ConcatDataset([valset, testset])
            else:
                testset = valset
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        return testloader

def getDataset(dataset: str, transform = None, transform_val = None, target_transform = None):
    valset = testset = extraset = None
    os.makedirs('./data', exist_ok=True)
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val, target_transform=target_transform)
    
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
        valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val, target_transform=target_transform)

    elif dataset == 'dtd':
        trainset = torchvision.datasets.DTD(root='./data', download=True, transform=transform, target_transform=target_transform)
        valset = torchvision.datasets.DTD(root='./data', split='val', download=True, transform=transform_val, target_transform=target_transform)
        testset = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform_val, target_transform=target_transform)
    
    elif dataset == 'fashionmnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
        valset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_val, target_transform=target_transform)
    
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
        valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_val, target_transform=target_transform)
    
    elif dataset == 'places365':
        trainset = torchvision.datasets.Places365(root='./data', download=True, transform=transform, target_transform=target_transform, small=True)
        valset = torchvision.datasets.Places365(root='./data', split='val', download=True, transform=transform_val, target_transform=target_transform, small=True)
    
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', download=True, transform=transform, target_transform=target_transform)
        valset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_val, target_transform=target_transform)
        # extraset = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform, target_transform=target_transform)
        # trainset = torch.utils.data.ConcatDataset([trainset, extraset]) # no extraset
    else:
        dataset_path = f'./data/{dataset}/'
        if dataset == 'notmnist':
            trainset = torchvision.datasets.ImageFolder(dataset_path + 'train', transform=transform, target_transform=target_transform)
            valset = torchvision.datasets.ImageFolder(dataset_path + 'val', transform=transform_val, target_transform=target_transform)
        elif dataset == 'tin':
            trainset = torchvision.datasets.ImageFolder(dataset_path + 'train', transform=transform, target_transform=target_transform)
            valset = processValTIN(dataset_path=dataset_path, transform_val=transform_val, target_transform=target_transform) # if tiny imagenet val in raw form
        else:
            trainset = torchvision.datasets.ImageFolder(dataset_path + 'train', transform=transform, target_transform=target_transform)
            try:
                valset = torchvision.datasets.ImageFolder(dataset_path + 'val', transform=transform_val, target_transform=target_transform)
            except:
                warn('No validation set.')
            try:
                testset = torchvision.datasets.ImageFolder(dataset_path + 'test', transform=transform_val, target_transform=target_transform)
            except:
                warn('No test set.')

    return trainset, valset, testset, extraset

def processValTIN(dataset_path: str, transform_val, target_transform):
    if len(os.listdir(dataset_path + 'val')) != 200:
        df = pd.read_csv(dataset_path + 'val/val_annotations.txt', delimiter='\t')
        for dir_ in next(os.walk(dataset_path + 'train'))[1]:
            os.makedirs(dataset_path + 'val/' + dir_, exist_ok=True)
        for _, row in df.iterrows():
            shutil.move(f'{dataset_path}val/images/{row[0]}', f'{dataset_path}val/{row[1]}/{row[0]}')
        
        shutil.move(f'{dataset_path}val/val_annotations.txt', f'{dataset_path}val/images/val_annotations.txt')
        shutil.move(f'{dataset_path}val/images', dataset_path)  

    return torchvision.datasets.ImageFolder(dataset_path + 'val', transform=transform_val, target_transform=target_transform)


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
                      'mnist': ([0.13062754273414612, 0.13062754273414612, 0.13062754273414612], [0.30810779333114624, 0.30810779333114624, 0.30810779333114624]),
                      'fashionmnist': ([0.28604060411453247, 0.28604060411453247, 0.28604060411453247], [0.3530242443084717, 0.3530242443084717, 0.3530242443084717]),
                      'notmnist': ([0.4239663035214087, 0.4239663035214087, 0.4239663035214087], [0.4583350861943875, 0.4583350861943875, 0.4583350861943875]),
                      'dtd': ([0.52875836, 0.4730212, 0.4247069], [0.26853561, 0.25950334, 0.26667375]),
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

def getShape(dataset):
    return shape_dict[dataset]

def getNormalization(dataset):
    return normalization_dict[dataset]

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

contamination_dict = {'cifar10': 10000,
                    'cifar100': 10000,
                    'svhn': 13700,
                    'mnist': 10000,
                    'fashionmnist': 10000,
                    'notmnist': 9362,
                    'dtd': 5640,
                    'places365': 36500,
                    'tin': 10000,
}

def getNN(nn: str, dataset: str):
    if dataset != 'tin':
        model = torch.hub.load('pytorch/vision:v0.14.0', nn) 
        numFetures = num_features_dict[nn]
        numClasses = num_classes_dict[dataset]
        if nn.startswith('resnet'):
            model.fc = torch.nn.Linear(numFetures, numClasses)
        elif nn.startswith('densenet'):
            model.classifier = torch.nn.Linear(numFetures, numClasses)
    else:
        if nn.startswith('resnet18'):
            model = torch.hub.load('pytorch/vision:v0.14.0', nn, weights='ResNet18_Weights.IMAGENET1K_V1') 
        elif nn.startswith('densenet121'):
            model = torch.hub.load('pytorch/vision:v0.14.0', nn, weights='DenseNet121_Weights.IMAGENET1K_V1') 
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

def loadNNWeights(nn: str, checkpoint: str, both_layers: bool, dataset: str, use_gpu: bool):
    path = f'./checkpoints/{checkpoint}'
    if dataset in ['cifar10', 'mnist']:
        model = get_network(num_classes=num_classes_dict[dataset], name=nn, use_gpu=use_gpu, checkpoint=path)
    else:
        ckpt = torch.load(path)
        model = getNN(nn, dataset)
    if both_layers:
        pass
    #     if nn.startswith('resnet'):
    #         # model = create_feature_extractor(model, ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']) # maybe for mds
    #         model = create_feature_extractor(model, ['avgpool', 'fc'])
    #     elif nn.startswith('densenet'):
    #         model = create_feature_extractor(model, ['features', 'classifier'])
    #     else:
    #         raise NotImplementedError("Not known.")
    if dataset in ['cifar10', 'mnist']:
        return model

    if use_gpu:
        model = model.cuda()
    if dataset != 'tin':
        if nn.startswith('resnet'):
            if 'model_state_dict' in ckpt.keys():
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
                except:
                    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            elif 'conv1' in ckpt.keys():
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
                except:
                    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        elif nn.startswith('densenet'):
            if 'model_state_dict' in ckpt.keys():
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
                except:
                    # correct layers
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            elif 'features.conv0.weight' in ckpt.keys():
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
                except:
                    # correct layers
                    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    
    # print(missing_keys, len(missing_keys))
    # print(unexpected_keys, len(unexpected_keys))
    model.eval()
    return model

def save_scores(fetures, logits, labels, save_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, save_name),
                fetures=fetures,
                logits=logits,
                labels=labels)
    
def save_scores_(fetures, labels, save_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, save_name),
                fetures=fetures,
                labels=labels)
    
def getLastLayers(model, data):
    return model(data, return_feature=True)

def get_dataloader(dataset_config, preprocessor_args):
    # prepare a dataloader dictionary
    dataloader_dict = {}
    for split in dataset_config['split_names']:
        preprocessor = get_preprocessor(preprocessor_args, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(preprocessor_args)
        train = True if split == 'train' else False
        transform = preprocessor if split == 'train' else data_aux_preprocessor
        if dataset_config['name'] == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=dataset_config['data_dir'], train=train, download=True, transform=transform)

        elif dataset_config['name'] == 'mnist':
            dataset = torchvision.datasets.MNIST(root=dataset_config['data_dir'], train=train, download=True, transform=transform)
        else:
            raise NotImplementedError
        
        sampler = None

        dataloader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True if split == 'train' else False,
                                num_workers=4,
                                sampler=sampler)

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_ood_dataloader(ood_config, preprocessor_args):
    dataloader_dict = {}
    for split in ood_config['split_names']:
        preprocessor = get_preprocessor(preprocessor_args, split)
        data_aux_preprocessor = TestStandardPreProcessor(preprocessor_args)
        if split == 'val':
            # validation set
            train = True if split == 'train' else False
            transform = preprocessor if split == 'train' else data_aux_preprocessor
            if ood_config['name'] == 'cifar10_ood':
                dataset = torchvision.datasets.CIFAR10(root=ood_config['data_dir'], train=train, download=True, transform=transform)

            elif ood_config['name'] == 'mnist_ood':
                dataset = torchvision.datasets.MNIST(root=ood_config['data_dir'], train=train, download=True, transform=transform)
            else:
                raise NotImplementedError
            dataloader = DataLoader(dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True if split == 'train' else False,
                                    num_workers=4)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in ood_config[split]:
                train = True if split == 'train' else False
                transform = preprocessor if split == 'train' else data_aux_preprocessor
                if ood_config['name'] == 'cifar10_ood':
                    dataset = torchvision.datasets.CIFAR10(root=ood_config['data_dir'], train=train, download=True, transform=transform)

                elif ood_config['name'] == 'mnist_ood':
                    dataset = torchvision.datasets.MNIST(root=ood_config['data_dir'], train=train, download=True, transform=transform)
                else:
                    raise NotImplementedError
                dataloader = DataLoader(dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True if split == 'train' else False,
                                        num_workers=4)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict
