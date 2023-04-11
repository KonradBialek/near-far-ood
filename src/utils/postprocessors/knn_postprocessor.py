from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.utils import getLastLayers

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNNPostprocessor(BasePostprocessor):
    def __init__(self, method_args):
        super(KNNPostprocessor, self).__init__(method_args)
        self.K = int(method_args[0])

    def setup(self, net: nn.Module, trainloader):
        activation_log = []
        net.eval()
        with torch.no_grad():
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
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(self.activation_log)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        feature, output = getLastLayers(net, data)
        feature_normed = normalizer(feature.data.cpu().numpy().reshape(
                    feature.shape[0], feature.shape[1], -1).mean(2))
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return torch.from_numpy(kth_dist), pred
