import torch
import pandas as pd
import numpy as np
from .utils import dataloader, getNN, isCuda, getShape, getNormalization, loadNNWeights, save_scores,  showLayers
import faiss
from torch.utils.data import DataLoader

criterion = torch.nn.CrossEntropyLoss()
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def KNN(data: pd.DataFrame, method_args: list):
    K = int(method_args[0])
    data = data.to_numpy()
    activation_log = normalizer(data.reshape(
                    data.shape[0], data.shape[1], -1).mean(2))

    index = faiss.IndexFlatL2(data.shape[1])
    activation_log = np.ascontiguousarray(activation_log)
    index.add(np.float32(activation_log))

    feature_normed = np.float32(np.ascontiguousarray(normalizer(data)))
    D, _ = index.search(feature_normed, K)
    return torch.from_numpy(D[:, -1])
    
def ODIN(data, model, method_args: list):
    temperature = int(method_args[0])
    noise = float(method_args[1])

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

def MSP(data: pd.DataFrame, method_args: list):
    score = torch.softmax(torch.tensor(data.values), dim=1)
    return torch.max(score, dim=1)[0]


def MDS(data: DataLoader, model, method_args: list):
    pass

# def MLS(df: pd.DataFrame):
#     pass

def measure(nn: str, method: str, datasets: list, method_args: list):
    for i, dataset in enumerate(datasets):
        if i == 0:
            file = f'{dataset}_{nn}_ID.csv'
        else:
            file = f'{dataset}_{nn}_OoD.csv'
        path = f'./features/{file}'
        data = pd.read_csv(path)
        label = data.pop('class')

        if method == 'knn':
            output = KNN(data, method_args)
        if method == 'msp':
            output = MSP(data, method_args)
        # if method == 'mls':
        #     raise NotImplementedError(f"{method} not implenented")
        #     MLS(df)

        data["class"] = label
        data[method] = output
        data.to_csv(path[:-4]+'_'+method+'.csv', mode='w', index=False, header=True)


def measure_(nn: str, method: str, datasets: list, method_args: list, checkpoint):
    model = loadNNWeights(nn, checkpoint)

    datasetLoaders = {}
    for dataset in datasets:
        loaders = dataloader(dataset, postprocess=True)
        datasetLoaders[dataset] = {'train': loaders[0], 'val': loaders[1], 'test': loaders[2]}
    
    datasetLoaders2 = {}
    for i, split in enumerate(datasetLoaders[datasets[0]]):
        datasetLoaders2[split] = {}
        for dataset in datasetLoaders:
            datasetLoaders2[split][dataset] = loaders[i]

    datasetLoaders2['nearood'] = datasetLoaders['cifar100']
    datasetLoaders2['farood'] = datasetLoaders['mnist']

    for dataset, (_, loader) in enumerate(datasetLoaders.items()):
        conf, gt = inference(model, loader['val'], method, method_args)
        save_name = datasets[dataset]
        if dataset > 0:
            gt = -1 * np.ones_like(gt)  # hard set to -1 as ood
            save_name += '_OoD'
        save_scores(conf, gt, save_name)


def inference(model: torch.nn.Module, data_loader: DataLoader, method: str, method_args: list):
    conf_list, label_list = [], []
    for batch in data_loader:
        data = batch[0].cuda()
        label = batch[1].cuda()
        if method == 'odin':
            conf = ODIN(data, model, method_args)
        elif method == 'mds':
            conf = MDS(data, model, method_args)
        for idx in range(len(data)):
            conf_list.append(conf[idx].cpu().tolist())
            label_list.append(label[idx].cpu().tolist())

    # convert values into numpy array
    conf_list = np.array(conf_list)
    label_list = np.array(label_list, dtype=int)

    return conf_list, label_list