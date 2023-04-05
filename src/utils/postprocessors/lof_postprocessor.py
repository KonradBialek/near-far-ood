import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor

from utils.utils import getLastLayers


class LocalOutlierFactorPostprocessor(BasePostprocessor):
    def __init__(self, method_args):
        super().__init__(method_args)
        self.n_neighbors = int(method_args[0])

    def setup(self, net: nn.Module, trainloader):
        activation_log = []
        net.eval()
        with torch.no_grad():
            for batch in tqdm(trainloader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch[0].cuda()
                data = data.float()

                batch_size = data.shape[0]

                feature = getLastLayers(net, data)[0]

                dim = feature.shape[1]
                activation_log.append(feature.data.cpu().numpy().reshape(
                    batch_size, dim, -1).mean(2))

        activation_log = np.concatenate(activation_log, axis=0)
        self.clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True, n_jobs=-1)

        self.clf.fit(activation_log)

        
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data):
        '''
        Measure distance with MSP.

        Args:
            data (np.ndarray or Tensor): Features to process.
            net (None, ResNet or DenseNet): Model of network.
        '''
        if not isinstance(data, np.ndarray):
            data = getLastLayers(net, data)[0]
            data = data.data.cpu().numpy().reshape(
                    data.shape[0], data.shape[1], -1).mean(2)

        pred = torch.Tensor(self.clf.predict(data))
        conf = torch.Tensor(self.clf.score_samples(data))
        return conf, pred
        
