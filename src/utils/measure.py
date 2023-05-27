import os
import torch
import pandas as pd
import numpy as np

from utils.evaluators.utils import get_evaluator
from utils.postprocessors.utils import get_postprocessor
from .utils import dataloader, get_dataloader, get_ood_dataloader, getLastLayers, getNormalization, getShape, bothLayers, isCuda, loadNNWeights, save_scores, save_scores_, num_classes_dict, shape_dict, normalization_dict
import faiss
from torch.utils.data import DataLoader


def extract(model, testloader: torch.utils.data.DataLoader, use_gpu: bool, i: int, save_name = None):
    '''
    Extracts features from dataset.

    Args:
        model: Model of network.
        testloader (DataLoader): DataLoader.
        use_gpu (bool): If use GPU.
        ID (bool) If in-distribution dataset.
        i (int): Id to set labels for OoD dataset.
    '''
    features, logits, labels = [], [], []
    for images, labels__ in testloader:
        if use_gpu: 
            images = images.cuda()
        features__, logits__ = getLastLayers(model, images)
        features__ = features__.cpu().detach().numpy()
        logits__ = logits__.cpu().detach().numpy()
        features.append(features__)
        labels.append(labels__)
        logits.append(logits__)

    features = np.concatenate(features, axis=0)
    logits = np.concatenate(logits, axis=0)
    if i > 0:
        labels = -i * np.ones(len(features))
    else:
        labels = np.concatenate(labels, axis=0)
        
    if save_name is not None:
        save_scores(features, logits, labels, save_name=save_name, save_dir='./features')
    else:
        features_.append(features)
        logits_.append(logits)
        labels_.append(labels)

def measure(nn: str, method: str, datasets: list, method_args: list, checkpoint = None, mode='measure'):
    use_gpu = isCuda()
    model = loadNNWeights(nn, checkpoint, both_layers=bothLayers(method=method), dataset=datasets[0], use_gpu=use_gpu)

    if mode == 'measure':
        print(method)
        if method == 'mds':
            method_args.append(num_classes_dict[datasets[0]])
        method_args.append(use_gpu)
        evaluator = get_evaluator(eval='ood', eval_args=[use_gpu])
        postprocessor = get_postprocessor(method=method, method_args=method_args)

        dataloader_args = {'split_names':  ['train', 'val', 'test'], 'name': datasets[0], 'num_classes': num_classes_dict[datasets[0]], 'data_dir': './data/images_classic'}
        preprocessor_args = {'name': 'base', 'image_size': getShape(datasets[0])[0], 'interpolation': 'bilinear', 'normalization_type': datasets[0]}
        id_loader = get_dataloader(dataloader_args, preprocessor_args)
        postprocessor.setup(net=model, trainloader=id_loader)

        dataloader_args = {'split_names': datasets, 'name': datasets[0], 'num_classes': num_classes_dict[datasets[0]], 'data_dir': './data/images_classic'}
        ood_loader = get_ood_dataloader(dataloader_args, preprocessor_args, lof=True if method == 'lof' else False)

        print('\nOOD...', flush=True)
        evaluator.eval_ood_(model, ood_loader,
                            postprocessor)

    if mode == 'extract':
        global features_, logits_, labels_
        features_, logits_, labels_ = [], [], []
        os.makedirs('./features/', exist_ok=True)

        save_name = nn
        dataloader_args = {'split_names':  ['test'], 'num_classes': num_classes_dict[datasets[0]], 'data_dir': './data/images_classic'}
        preprocessor_args = {'name': 'base', 'image_size': getShape(datasets[0])[0], 'interpolation': 'bilinear', 'normalization_type': datasets[0]}
        for i, dataset in enumerate(datasets):
            print(f'extracting {dataset}')
            dataloader_args['name'] = datasets[i]
            save_name += f'_{dataset}'
            testloader = get_dataloader(dataloader_args, preprocessor_args)
            extract(model, testloader['test'], use_gpu, i)
        
        features_, logits_, labels_ = np.concatenate(features_, axis=0), np.concatenate(logits_, axis=0), np.concatenate(labels_, axis=0)
        save_scores(features_, logits_, labels_, save_name=save_name, save_dir='./features')