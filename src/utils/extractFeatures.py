import numpy as np
import pandas as pd
import torch
from torchvision.models.resnet import ResNet
from torchvision.models.densenet import DenseNet
import os
from .utils import dataloader, getShapeNormalization, isCuda, loadNNWeights, save_scores,  showLayers

def extract(model: ResNet or DenseNet, testloader: torch.utils.data.DataLoader, use_gpu: bool, i: int, save_name = None):
    '''
    Extracts features from dataset.

    Args:
        model (ResNet or DenseNet): Model of network.
        testloader (DataLoader): DataLoader.
        use_gpu (bool): If use GPU.
        ID (bool) If in-distribution dataset.
        i (int): Id to set labels for OoD dataset.
    '''
    outputs = []
    for images, _ in testloader:
        if use_gpu: 
            images = images.cuda()
        outputs.append(model(images).cpu().detach())

    outputs = torch.cat(outputs)
    outputs = outputs.numpy()
    labels = i * np.ones(len(outputs))
        
    if save_name is not None:
        save_scores(outputs, labels, save_name=save_name, save_dir='./features')
    else:
        outputs_.append(outputs)
        labels_.append(labels)


def extractFeatures(nn: str, datasets: list, checkpoint: str):
    '''
    Manage extracting features from datasets.

    Args:
        nn (str): Name of used network.
        dataset (list): List of dataset names.
        checkpoint (str): Name of pretrained network checkpoint file.
    '''
    global outputs_, labels_
    outputs_, labels_ = [], []
    use_gpu = isCuda()
    model = loadNNWeights(nn, checkpoint)
    os.makedirs('./features/', exist_ok=True)

    save_name = nn
    shape, normalization = getShapeNormalization(datasets[0])
    for i, dataset in enumerate(datasets):
        print(f'extracting {dataset}')
        save_name += f'_{dataset}'
        if i > 0:
            testloader = dataloader(dataset, size=shape[:2], train=False, setup=False, normalization=normalization)
            extract(model, testloader, use_gpu, i)
        else:
            showLayers(model, shape) 
            trainloader = dataloader(dataset, size=shape[:2], train=False, setup=True, normalization=normalization)
            extract(model, trainloader, use_gpu, i, f'{nn}_{dataset}_setup')
            testloader = dataloader(dataset, size=shape[:2], train=False, setup=False, normalization=normalization)
            extract(model, testloader, use_gpu, i)
    
    outputs_, labels_ = np.concatenate(outputs_, axis=0), np.concatenate(labels_, axis=0)
    save_scores(outputs_, labels_, save_name=save_name, save_dir='./features')
            
