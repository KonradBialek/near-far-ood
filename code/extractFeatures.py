import pandas as pd
import torch
import os
from utils import dataloader, getNumFeatures, isCuda, shapeNormalization,  showLayers, touchCSV, generateHeaders

def extract(model, testloader: torch.utils.data.DataLoader, path: str):
    for images, labels in testloader:
        if use_gpu: 
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)

        outputs = pd.DataFrame(outputs.data.cpu()).astype("float")
        labels = pd.Series(labels.data.cpu()).astype("int")
        outputs[len(outputs.columns)] = labels
        outputs.to_csv(path, mode='a', index=False, header=False)


def extractFeatures(nn: str, in_dataset: str, ood_datasets: list, checkpoint: str):
    global use_gpu
    path = f'./checkpoints/{checkpoint}'
    ckpt = torch.load(path)
    model = torch.hub.load('pytorch/vision:v0.14.0', nn) 
    model.fc = torch.nn.Identity()
    try:
        model(torch.rand(*(torch.tensor((128, 3, 32, 32), device=torch.device('cuda'))))).data.shape[0]
        num_features = getNumFeatures(nn)
        rgb = True
    except:
        model(torch.rand(*(torch.tensor((128, 1, 32, 32), device=torch.device('cuda'))))).data.shape[0]
        num_features = getNumFeatures(nn)
        rgb = False
    
    headers = generateHeaders(num_features)
    use_gpu = isCuda()
    if use_gpu:
        model = model.cuda()

    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # print(missing_keys)
    # print(unexpected_keys)
    model.eval()

    os.makedirs('./features/', exist_ok=True)
    # shape, normalization = shapeNormalization(in_dataset, True)
    # showLayers(model, shape) 
    # testloader = dataloader(in_dataset, normalization, shape[:2], rgb, False, True, 1, 16)
    # path = f'./features/{in_dataset}_{nn}_ID_test'
    # touchCSV(path, headers)
    # print(f'extracting {in_dataset}')
    # if testloader is None:
    #     # extract(model, valloader, num_features,  True)
    #     pass
    # else:   
    #     extract(model, testloader, path)
    

    for ood_dataset in ood_datasets:
        shape, normalization = shapeNormalization(ood_dataset, True)
        testloader = dataloader(ood_dataset, normalization, shape[:2], rgb, False, False, 1, 16)
        path = f'./features/{ood_dataset}_{nn}_OoD'
        touchCSV(path, headers)
        print(f'extracting {ood_dataset}')
        extract(model, testloader, path)
