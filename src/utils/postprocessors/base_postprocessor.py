import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BasePostprocessor:
    def __init__(self, method_args):
        pass

    def setup(self, model: nn.Module, trainloader):
        pass

    @torch.no_grad()
    def postprocess(self, model: nn.Module, data):
        '''
        Measure distance with MSP.

        Args:
            data (np.ndarray or Tensor): Features to process.
            model (None, ResNet or DenseNet): Model of network.
        '''
        
        if isinstance(data, np.ndarray):
            score = torch.softmax(torch.tensor(data), dim=1)
        else:
            output = model(data).get('fc')
            score = torch.softmax(output, dim=1)
        return torch.max(score, dim=1)

    def inference(self, model: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        for batch in data_loader:
            data = batch[0].cuda()
            label = batch[1].cuda()
            pred, conf = self.postprocess(model, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
