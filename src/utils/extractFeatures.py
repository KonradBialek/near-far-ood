import pandas as pd
import torch
import os
from .utils import dataloader, getNumFeatures, getShape, getNormalization, isCuda, loadNNWeights,  showLayers, touchCSV, generateHeaders

def extract(model, testloader: torch.utils.data.DataLoader, path: str, use_gpu: bool, ID: bool):
    for images, labels in testloader:
        if use_gpu: 
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)

        outputs = pd.DataFrame(outputs.data.cpu()).astype("float")
        labels = pd.Series(labels.data.cpu()).astype("int")
        outputs[len(outputs.columns)] = labels if ID else -1
        outputs.to_csv(path, mode='a', index=False, header=False)


def extractFeatures(nn: str, datasets: list, checkpoint: str):
    use_gpu = isCuda()
    model = loadNNWeights(nn, checkpoint)
    
    num_features = getNumFeatures(nn)
    rgb = True
    headers = generateHeaders(num_features)
    os.makedirs('./features/', exist_ok=True)
    print(f'extracting {datasets[0]}')
    touchCSV(path, headers)
    shape = getShape(datasets[0])
    normalization = getNormalization(datasets[0], True)
    showLayers(model, shape) 
    testloader = dataloader(datasets[0], shape[:2], rgb, False, True, 1, 16, normalization)
    path = f'./features/{datasets[0]}_{nn}_ID.csv'
    extract(model, testloader, path, use_gpu)
    

    for i, dataset in enumerate(datasets):
        print(f'extracting {dataset}')
        touchCSV(path, headers)
        shape = getShape(dataset)
        normalization = getNormalization(dataset, True)
        if i > 0:
            testloader = dataloader(dataset, shape[:2], rgb, False, False, 1, 16, normalization)
            path = f'./features/{dataset}_{nn}_OoD.csv'
            extract(model, testloader, path, use_gpu, False)
        else:
            testloader = dataloader(dataset, shape[:2], rgb, False, True, 1, 16, normalization)
            path = f'./features/{dataset}_{nn}_ID.csv'
            extract(model, testloader, path, use_gpu, True)
            
