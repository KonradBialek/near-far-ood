import os
import torch
import pandas as pd
import numpy as np

from utils.evaluators.utils import get_evaluator
from utils.postprocessors.utils import get_postprocessor
from .utils import dataloader, getLastLayers, getNormalization, getShape, bothLayers, loadNNWeights, save_scores_
import faiss
from torch.utils.data import DataLoader

criterion = torch.nn.CrossEntropyLoss()
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def KNN(data: np.ndarray, method_args: list, data_: np.array):
    '''
    Measure distance with KNN.

    Args:
        data (np.ndarray): Features to process.
        method_args (list): List of method's aruments.
    '''

    # setup
    K = int(method_args[0])
    activation_log = normalizer(data_.reshape(
                    data_.shape[0], data_.shape[1], -1).mean(2))

    index = faiss.IndexFlatL2(data_.shape[1])
    activation_log = np.ascontiguousarray(activation_log)
    index.add(np.float32(activation_log))

    # process
    feature_normed = np.float32(np.ascontiguousarray(normalizer(data)))
    D, _ = index.search(feature_normed, K)
    return torch.from_numpy(D[:, -1])
    
def ODIN(data, model, method_args: list):
    '''
    Measure distance with ODIN.

    Args:
        data (Tensor): Features to process.
        model (ResNet or DenseNet): Model of network.
        method_args (list): List of method's aruments.
    '''
    
    # setup
    temperature = int(method_args[0])
    noise = float(method_args[1])

    # process
    data.requires_grad = True
    output = model(data)
    # output = getLastLayers(model, data)[1]
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
    # output = getLastLayers(model, tempInputs)[1]
    output = output / temperature

    # Calculating the confidence after adding perturbations
    nnOutput = output.detach()
    nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
    nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

    return nnOutput.max(dim=1)[0]

def MSP(data, model = None):
    '''
    Measure distance with MSP.

    Args:
        data (np.ndarray or Tensor): Features to process.
        model (None, ResNet or DenseNet): Model of network.
    '''
    
    if isinstance(data, np.ndarray):
        score = torch.softmax(torch.tensor(data), dim=1)
    else:
        output = model(data)
        # output = getLastLayers(model, data)[1]
        score = torch.softmax(output, dim=1)
    return torch.max(score, dim=1)[0]



def MLS(data, model = None):
    '''
    Measure distance with MLS.

    Args:
        data (np.ndarray or Tensor): Features to process.
        model (ResNet or DenseNet): Model of network.
    '''
    if isinstance(data, np.ndarray):
        output = torch.tensor(data)
    else:
        output = model(data)
        # output = getLastLayers(model, data)[1]
    return torch.max(output, dim=1)[0]

def measure(method: str, method_args: list):
    '''
    Loop over .npz files in ./features directory. Measure distance with requested method.

    Args:
        method (str): Requested method.
        method_args (list): List of method's aruments.
    '''
    both_layers = bothLayers(method=method)
    for file in os.listdir('./features/'):
        if file.endswith('setup.npz'):
            path = f'./features/{file}'
            if both_layers:
                data_ = np.load(path)['features']
            else:
                data_ = np.load(path)['logits']

    for file in os.listdir('./features/'):
        if file.endswith('.npz'):
            path = f'./features/{file}'
            if both_layers:
                data = np.load(path)['features']
            else:
                data = np.load(path)['logits']
            label = np.load(path)['labels']
            if data.ndim > 1 and not file.endswith('setup.npz'):
                if data.shape[1] > 2:
                    if method == 'knn':
                        output = KNN(data, method_args, data_)
                    if method == 'msp':
                        output = MSP(data)
                    save_scores_(output, label, file[:-4]+'_'+method, './features')
                else:
                    print("Data have invalid shape.")
            else:
                print("Data have invalid shape.")



def measure_(nn: str, method: str, datasets: list, method_args: list, checkpoint = None):
    model = loadNNWeights(nn, checkpoint, both_layers=bothLayers(method=method), dataset=datasets[0])
    evaluator = get_evaluator(eval='ood', eval_args=[])
    postprocessor = get_postprocessor(method=method, method_args=method_args)
    shape = getShape(datasets[0])
    normalization = getNormalization(datasets[0])

    trainloader = dataloader(datasets[0], size=shape[:2], train=False, setup=True, normalization=normalization, postprocess=True)
    postprocessor.setup(net=model, trainloader=trainloader)

    idLoader, oodLoaders = {}, {}
    for dataset in datasets:
        testloader = dataloader(dataset, size=shape[:2], train=False, setup=False, normalization=normalization, postprocess=True)
        if dataset == datasets[0]:
            idLoader[dataset] = testloader
        else:
            oodLoaders[dataset] = testloader

    # start calculating accuracy
    print('\nStart evaluation...', flush=True)
    acc_metrics = evaluator.eval_acc(model, idLoader[datasets[0]],
                                    postprocessor)
    print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
            flush=True)
    print(u'\u2500' * 70, flush=True)

    # start evaluating ood detection methods
    evaluator.eval_ood(model, idLoader, oodLoaders, postprocessor)
    print('Completed!', flush=True)

    # testloader = dataloader(datasets, size=shape[:2], train=False, setup=False, normalization=normalization, postprocess=True)
    # print('\nOOD...', flush=True)
    # acc_metrics = evaluator.eval_acc(model, testloader,
    #                                 postprocessor)
    # print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
    #         flush=True)
    # print(u'\u2500' * 70, flush=True)