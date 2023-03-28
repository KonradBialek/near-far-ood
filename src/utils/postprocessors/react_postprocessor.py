from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.utils import getLastLayers

from .base_postprocessor import BasePostprocessor


class ReactPostprocessor(BasePostprocessor):
    def __init__(self, mthod_args):
        super(ReactPostprocessor, self).__init__(mthod_args)
        self.percentile = int(mthod_args[0])

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
        self.threshold = np.percentile(activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = self.forward_threshold(data, net)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return energyconf, pred
    
    def forward_threshold(self, data: Any, net):
        net.eval()
        if hasattr(net, 'fc'):
            feature = net(data).get("avgpool")
            feature = feature.clip(max=self.threshold)
            feature = feature.view(feature.size(0), -1)
            logits_cls = net.fc(feature)
        elif hasattr(net, 'classsifier'):
            feature = net(data).get("features")
            feature = feature.clip(max=self.threshold)
            feature = feature.view(feature.size(0), -1)
            logits_cls = net.classifier(feature)

        return logits_cls
