import numpy as np
import pandas as pd
import torch
import os
from .utils import dataloader, getNumFeatures, getShape, getNormalization, isCuda, loadNNWeights, save_scores,  showLayers

def extract(model, testloader: torch.utils.data.DataLoader, save_name: str, use_gpu: bool, ID: bool, i: int):
    outputs = []
    labels = []
    for images, classes in testloader:
        if use_gpu: 
            images = images.cuda()
        outputs.append(model(images).cpu().detach())
        labels.append(classes.cpu().detach())

    # np
    outputs = torch.cat(outputs)
    outputs = outputs.numpy()
    if ID:
        labels = torch.cat(labels)
        labels = labels.numpy()
    else:
        labels = -i * np.ones(len(outputs))
    save_scores(outputs, labels, save_name=save_name, save_dir='./features')

        # pd
        # outputs = pd.DataFrame(outputs.data.cpu()).astype("float")
        # labels = pd.Series(labels.data.cpu()).astype("int")
        # outputs[len(outputs.columns)] = labels if ID else -i
        # outputs.to_csv(path, mode='a', index=False, header=False)


def extractFeatures(nn: str, datasets: list, checkpoint: str):
    use_gpu = isCuda()
    model = loadNNWeights(nn, checkpoint)
    
    num_features = getNumFeatures(nn)
    rgb = True
    os.makedirs('./features/', exist_ok=True)

    for i, dataset in enumerate(datasets):
        print(f'extracting {dataset}')
        shape = getShape(dataset)
        normalization = getNormalization(dataset, True)
        if i > 0:
            testloader = dataloader(dataset, shape[:2], rgb, False, False, 1, 16, normalization)
            save_name = f'{dataset}_{nn}_OoD'
            extract(model, testloader, save_name, use_gpu, False, i)
        else:
            showLayers(model, shape) 
            testloader = dataloader(dataset, shape[:2], rgb, False, True, 1, 16, normalization)
            save_name = f'{dataset}_{nn}_ID'
            extract(model, testloader, save_name, use_gpu, True, i)
            
