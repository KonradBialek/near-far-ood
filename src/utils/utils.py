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
# from cutmix.cutmix import CutMix

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

def dataloader(dataset: str, size = (32, 32), rgb = False, train = False, ID = False, n_holes = 1, length = 16, normalization = [[0.5], [0.5]], batch_size = BATCH_SIZE, calcNorm = False, postprocess = False):
    valloader = testloader = None
    
    if rgb and dataset in ['mnist', 'fashionmnist', 'notmnist']: # sieć musi być kolorowa więc dataset tez
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
        # if rgb and dataset in ['mnist', 'fashionmnist', 'notmnist']: # todo problem z zmianą kolorów
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

    if train or postprocess:
        # trainset = CutMix(trainset, num_class=getNumClasses(dataset), beta=1.0, prob=0.25, num_mix=2)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        if valset is not None:
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
        if testset is not None:
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainloader, valloader, testloader
    else:
        if ID:
            testset = trainset
        elif valset is not None:
            if testset is not None:
                testset = torch.utils.data.ConcatDataset([trainset, valset, testset])
            else:
                testset = torch.utils.data.ConcatDataset([trainset, valset])
        elif testset is not None:
            testset = torch.utils.data.ConcatDataset([trainset, testset])
        else:
            testset = trainset
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return testloader

def getDataset(dataset: str, transform = None, transform_val = None):
    valset = testset = extraset = None
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
        # extraset = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform_val)
        # trainset = torch.utils.data.ConcatDataset([trainset, extraset])
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
    # use_gpu = False # if gpu is busy
    use_gpu = torch.cuda.is_available()
    print('GPU: '+str(use_gpu))
    return use_gpu

def showLayers(model, shape):
    if use_gpu:
        summary(model, (shape[2], shape[0], shape[1]))    


def getNormalization(dataset: str, train_ID=False):
    if dataset == 'cifar10': # todo generalize
        if train_ID:
            normalization = [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]
        else:
            normalization = [0.49186878, 0.48265391, 0.44717728], [0.24697121, 0.24338894, 0.26159259]
    elif dataset == 'cifar100':
        if train_ID:
            normalization = [0.48042983, 0.44819681, 0.39755555], [0.2764398, 0.26888656, 0.28166855]
        else:
            normalization = [0.50736203, 0.48668956, 0.44108857], [0.26748815, 0.2565931, 0.27630851]
    elif dataset == 'mnist':
        if train_ID:
            normalization = [0.13062754273414612], [0.30810779333114624]
        else:
            normalization = [0.13092535192648502], [0.3084485240270358]
    elif dataset == 'fashionmnist':
        if train_ID:
            normalization = [0.28604060411453247], [0.3530242443084717]
        else:
            normalization = [0.2861561232350083], [0.3529415461508495]
    elif dataset == 'notmnist':
        if train_ID:
            raise NotImplementedError(f'Normalization and shape of images in {dataset} is not known.')
        else:
            raise NotImplementedError(f'Normalization and shape of images in {dataset} is not known.')
    elif dataset == 'dtd':
        if train_ID:
            raise NotImplementedError(f'Normalization and shape of images in {dataset} is not known.')
        else:
            raise NotImplementedError(f'Normalization and shape of images in {dataset} is not known.')
    elif dataset == 'places365':
        if train_ID:
            raise NotImplementedError(f'Normalization and shape of images in {dataset} is not known.')
        else:
            raise NotImplementedError(f'Normalization and shape of images in {dataset} is not known.')
    elif dataset == 'svhn':
        if train_ID:
            normalization = [0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614]
        else:
            normalization = [0.44154697, 0.44605756, 0.47180097], [0.20396256, 0.20805474, 0.20576004]
    elif dataset == 'tin':
        if train_ID:
            normalization = [0.48023694, 0.44806704, 0.39750364], [0.27643643, 0.26886328, 0.28158993]
        else:
            normalization = [0.48042983, 0.44819681, 0.39755555], [0.2764398, 0.26888656, 0.28166855]
    else:
        normalization = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    return normalization

def getShape(dataset: str):
    if dataset in ['cifar10', 'cifar100', 'svhn']: # todo generalize
        shape = 32, 32, 3
    elif dataset in ['mnist', 'fashionmnist', 'notmnist']:
        shape = 28, 28, 3
    elif dataset == 'dtd':
        shape = 300, 300, 3
    elif dataset == 'places365':
        shape = 256, 256, 3
    elif dataset == 'tin':
        shape = 64, 64, 3

    return shape

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
    numFetures = getNumFeatures(nn)
    numClasses = getNumClasses(dataset)
    model.fc = torch.nn.Linear(numFetures, numClasses)
    return model

def getNumFeatures(nn: str):
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

def touchCSV(path: str, headers):
    with open(path, 'w') as file:
        file.writelines(",".join(headers) + "\n")

def generateHeaders(num_headers: int):
    headers = []  
    for header in range(1,num_headers + 1):
        headers.append(f"{header}_feature")
    
    headers.append("class")
    return headers

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


def save_scores(conf, gt, save_name):
    save_dir = os.path.join('./results', 'scores1')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, save_name),
                conf=conf,
                label=gt)