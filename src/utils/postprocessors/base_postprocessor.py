import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import getLastLayers


class BasePostprocessor:
    def __init__(self, method_args):
        self.device = 'cuda' if method_args[-1] else 'cpu'

    def setup(self, net: nn.Module, trainloader):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data):
        '''
        Measure distance with MSP.

        Args:
            data (np.ndarray or Tensor): Features to process.
            net (None, ResNet or DenseNet): Model of network.
        '''
        
        if isinstance(data, np.ndarray):
            score = torch.softmax(torch.tensor(data), dim=1)
        else:
            try:
                output = net(data)
            except:
                net.cuda()
                output = net(data)
            # output = getLastLayers(net, data)[1]
            score = torch.softmax(output, dim=1)
        return torch.max(score, dim=1)

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        for batch in data_loader:
            data = batch[0].to(device=self.device)
            label = batch[1].to(device=self.device)
            conf, pred = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
