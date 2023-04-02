"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.utils import getLastLayers, getNormalization

from .base_postprocessor import BasePostprocessor


class ODINPostprocessor(BasePostprocessor):
    def __init__(self, method_args):
        super().__init__(method_args)

        self.temperature = int(method_args[0])
        self.noise = float(method_args[1])
        self.preprocessing = method_args[2].lower() in ['true', '1', 't', 'y', 'yes']
        self.normalization_dataset = getNormalization(method_args[3])[1]

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        data.grad = torch.gradient(data)[0]
        output = net(data)
        # output = getLastLayers(net, data)[1]

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss = Variable(loss, requires_grad=True)
        loss.backward()

        if self.preprocessing:
            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(data.grad.detach(), 0)
            gradient = (gradient.float() - 0.5) * 2

            # Scaling values taken from original code
            gradient[:, 0] = (gradient[:, 0]) / self.normalization_dataset[0]
            gradient[:, 1] = (gradient[:, 1]) / self.normalization_dataset[1]
            gradient[:, 2] = (gradient[:, 2]) / self.normalization_dataset[2]

            # Adding small perturbations to images
            tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
            output = net(tempInputs)
            # output = getLastLayers(net, tempInputs)[1]
            output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        return nnOutput.max(dim=1)
