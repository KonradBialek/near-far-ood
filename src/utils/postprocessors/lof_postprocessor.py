import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor

from utils.utils import getLastLayers

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)+1e-10

class LocalOutlierFactorPostprocessor(BasePostprocessor):
    def __init__(self, method_args):
        super().__init__(method_args)
        self.n_neighbors = int(method_args[0])

    def setup(self, net, trainloader):
        activation_log = []
        net.eval()
        with torch.no_grad(): # prepare data to fit classifier
            for batch in tqdm(trainloader['train'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch[0].to(device=self.device)
                data = data.float()

                batch_size = data.shape[0]
                feature = getLastLayers(net, data)[0]

                dim = feature.shape[1]
                activation_log.append(
                    normalizer(feature.data.cpu().numpy().reshape(
                    batch_size, dim, -1).mean(2)))

        self.activation_log = np.concatenate(activation_log, axis=0)
        self.clf = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, 
            novelty=True, n_jobs=-1)
        self.clf.fit(self.activation_log) # fit lof

        
    @torch.no_grad()
    def postprocess(self, net, data):
        # process data
        data = getLastLayers(net, data)[0]
        data = normalizer(data.data.cpu().numpy().reshape(
               data.shape[0], data.shape[1], -1).mean(2))

        pred = torch.Tensor(self.clf.predict(data))
        conf = torch.Tensor(self.clf.score_samples(data))
        return conf, pred