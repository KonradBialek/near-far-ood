import os
import torch
import pandas as pd
import numpy as np
from .utils import dataloader, getNN, isCuda, getShape, getNormalization, loadNNWeights, save_scores,  showLayers
import faiss
from torch.utils.data import DataLoader

criterion = torch.nn.CrossEntropyLoss()
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def KNN(data: np.ndarray, method_args: list):
    # setup
    K = int(method_args[0])
    activation_log = normalizer(data.reshape(
                    data.shape[0], data.shape[1], -1).mean(2))

    index = faiss.IndexFlatL2(data.shape[1])
    activation_log = np.ascontiguousarray(activation_log)
    index.add(np.float32(activation_log))

    # process
    feature_normed = np.float32(np.ascontiguousarray(normalizer(data)))
    D, _ = index.search(feature_normed, K)
    return torch.from_numpy(D[:, -1])
    
def ODIN(data, model, method_args: list):
    # setup
    temperature = int(method_args[0])
    noise = float(method_args[1])

    # process
    data.requires_grad = True
    output = model(data)
    criterion = torch.nn.CrossEntropyLoss()
    labels = output.detach().argmax(axis=1)

    # Using temperature scaling
    output = output / temperature

    loss = criterion(output, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(data.grad.detach(), 0)
    gradient = (gradient.float() - 0.5) * 2

    # Scaling values taken from original code
    gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
    gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
    gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)

    # Adding small perturbations to images
    tempInputs = torch.add(data.detach(), gradient, alpha=-noise)
    output = model(tempInputs)
    output = output / temperature

    # Calculating the confidence after adding perturbations
    nnOutput = output.detach()
    nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
    nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

    return nnOutput.max(dim=1)[0]

def MSP(data, model = None):
    if isinstance(data, np.ndarray):
        score = torch.softmax(torch.tensor(data), dim=1)
    else:
        output = model(data)
        score = torch.softmax(output, dim=1)
    return torch.max(score, dim=1)[0]


def MDS(data: DataLoader, model, method_args: list):
    pass

# def MLS(df: pd.DataFrame):
#     pass

def measure(nn: str, method: str, datasets: list, method_args: list):
    for file in os.listdir('./features/'):
        if file.endswith('.npz'):
            path = f'./features/{file}'
            data = np.load(path)['data']
            label = np.load(path)['labels']

            if method == 'knn':
                output = KNN(data, method_args)
            if method == 'msp':
                output = MSP(data)
            # if method == 'mls':
            #     raise NotImplementedError(f"{method} not implenented")
            #     MLS(df)
            save_scores(output, label, file[:-4]+'_'+method, './features')
            # np.save(path[:-4]+'_'+method, output)


def measure_(nn: str, method: str, datasets: list, method_args: list, checkpoint = None):
    outputs, labels = [], []
    model = loadNNWeights(nn, checkpoint)

    datasetLoaders = []
    for dataset in datasets:
        _, valloader, _ = dataloader(dataset, postprocess=True)
        datasetLoaders.append(valloader)

    save_name = nn
    for dataset, loader in enumerate(datasetLoaders):
        conf, gt = inference(model, loader, method, method_args)
        save_name += f'_{datasets[dataset]}'
        if dataset > 0:
            gt = -dataset * np.ones_like(gt)
        outputs.append(conf)
        labels.append(gt)
        # np.save('./features/'+save_name+'_'+method, output)

    outputs, labels = np.concatenate(outputs, axis=0), np.concatenate(labels, axis=0)
    save_scores(conf, gt, save_name+'_'+method, './features')


def inference(model: torch.nn.Module, data_loader: DataLoader, method: str, method_args: list):
    conf_list, label_list = [], []
    for batch in data_loader:
        data = batch[0].cuda()
        label = batch[1].cuda()
        if method == 'odin':
            conf = ODIN(data, model, method_args)
        elif method == 'mds':
            conf = MDS(data, model, method_args)
        elif method == 'msp':
            conf = MSP(data, model)
        for idx in range(len(data)):
            conf_list.append(conf[idx].cpu().tolist())
            label_list.append(label[idx].cpu().tolist())

    conf_list = np.array(conf_list)
    label_list = np.array(label_list, dtype=int)

    return conf_list, label_list