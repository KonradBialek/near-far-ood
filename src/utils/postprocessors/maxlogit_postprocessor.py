from typing import Any

import torch
import torch.nn as nn

from utils.utils import getLastLayers

from .base_postprocessor import BasePostprocessor


class MaxLogitPostprocessor(BasePostprocessor):
    def __init__(self, method_args):
        super().__init__(method_args)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = getLastLayers(net, data)[1]
        return torch.max(output, dim=1)
