import pandas as pd
import torch
import os
from .utils import dataloader, getNumFeatures, getShape, getNormalization, isCuda, loadNNWeights,  showLayers, touchCSV, generateHeaders

def extract(model, testloader: torch.utils.data.DataLoader, path: str, use_gpu: bool):
    for images, labels in testloader:
        if use_gpu: 
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)

        outputs = pd.DataFrame(outputs.data.cpu()).astype("float")
        labels = pd.Series(labels.data.cpu()).astype("int")
        outputs[len(outputs.columns)] = labels
        outputs.to_csv(path, mode='a', index=False, header=False)


def extractFeatures(nn: str, in_dataset: str, ood_datasets: list, checkpoint: str):
    use_gpu = isCuda()
    model = loadNNWeights(nn, checkpoint)
    # try:
    #     model(torch.rand(*(torch.tensor((128, 3, 32, 32), device=torch.device('cuda'))))).data.shape[0]
    #     num_features = getNumFeatures(nn)
    #     rgb = True
    # except:
    #     model(torch.rand(*(torch.tensor((128, 1, 32, 32), device=torch.device('cuda'))))).data.shape[0]
    #     num_features = getNumFeatures(nn)
    #     rgb = False
    
    num_features = getNumFeatures(nn)
    rgb = True
    headers = generateHeaders(num_features)
    os.makedirs('./features/', exist_ok=True)
    shape = getShape(in_dataset)
    normalization = getNormalization(in_dataset, True)
    showLayers(model, shape) 
    testloader = dataloader(in_dataset, shape[:2], rgb, False, True, 1, 16, normalization)
    path = f'./features/{in_dataset}_{nn}_ID.csv'
    touchCSV(path, headers)
    print(f'extracting {in_dataset}')
    extract(model, testloader, path, use_gpu)
    

    for ood_dataset in ood_datasets:
        shape = getShape(ood_dataset)
        normalization = getNormalization(ood_dataset, True)
        testloader = dataloader(ood_dataset, shape[:2], rgb, False, False, 1, 16, normalization)
        path = f'./features/{ood_dataset}_{nn}_OoD.csv'
        touchCSV(path, headers)
        print(f'extracting {ood_dataset}')
        extract(model, testloader, path, use_gpu)
