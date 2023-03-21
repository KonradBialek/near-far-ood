import numpy as np
import pandas as pd
import torch
from torchvision.models.resnet import ResNet
from torchvision.models.densenet import DenseNet
import os
from .utils import dataloader, getNormalization, getShape, isCuda, loadNNWeights, save_scores,  showLayers

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
    features, logits = [], []
    for images, _ in testloader:
        if use_gpu: 
            images = images.cuda()
        output = model(images)
        features.append(output.get('avgpool').cpu().detach())
        logits.append(output.get('fc').cpu().detach())

    features = torch.cat(features)
    logits = torch.cat(logits)
    features = features.numpy()
    logits = logits.numpy()
    labels = i * np.ones(len(features))
        
    if save_name is not None:
        save_scores(features, logits, labels, save_name=save_name, save_dir='./features')
    else:
        features_.append(features)
        logits_.append(logits)
        labels_.append(labels)


def extractFeatures(nn: str, datasets: list, checkpoint: str):
    '''
    Manage extracting features from datasets.

    Args:
        nn (str): Name of used network.
        dataset (list): List of dataset names.
        checkpoint (str): Name of pretrained network checkpoint file.
    '''
    global features_, logits_, labels_
    features_, logits_, labels_ = [], [], []
    use_gpu = isCuda()
    model = loadNNWeights(nn, checkpoint, last_layer=False, dataset=datasets[0])
    os.makedirs('./features/', exist_ok=True)

    save_name = nn
    shape = getShape(datasets[0])
    normalization = getNormalization(datasets[0])
    showLayers(model, shape) 
    for i, dataset in enumerate(datasets):
        print(f'extracting {dataset}')
        save_name += f'_{dataset}'
        if i == 0:
            trainloader = dataloader(dataset, size=shape[:2], train=False, setup=True, normalization=normalization, postprocess=True)
            extract(model, trainloader, use_gpu, i, f'{nn}_{dataset}_setup')

        testloader = dataloader(dataset, size=shape[:2], train=False, setup=False, normalization=normalization, postprocess=True)
        extract(model, testloader, use_gpu, i)
    
    features_, logits_, labels_ = np.concatenate(features_, axis=0), np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0)
    save_scores(features_, logits_, labels_, save_name=save_name, save_dir='./features')
            
