import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.postprocessors import BasePostprocessor
from utils.utils import getLastLayers


def to_np(x):
    return x.data.cpu().numpy()


class BaseEvaluator:
    def __init__(self, eval_args):
        pass

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                data = batch[0].cuda()
                target = batch[1].cuda()

                # forward
                try: 
                    output = getLastLayers(net, data)[1]
                except:
                    output = net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def extract(self, net: nn.Module, data_loader: DataLoader):
        net.eval()
        feat_list, label_list = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Feature Extracting: ',
                              position=0,
                              leave=True):
                data = batch[0].cuda()
                label = batch[1]

                feat = getLastLayers(net, data)[0]
                feat_list.extend(to_np(feat))
                label_list.extend(to_np(label))

        feat_list = np.array(feat_list)
        label_list = np.array(label_list)

        save_dir = 'features'
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'feature'),
                 feat_list=feat_list,
                 label_list=label_list)

    def save_metrics(self, value):
        all_values = [value]
        temp = 0
        for i in all_values:
            temp = temp + i
        # total_value = np.add([x for x in all_values])s

        return temp
