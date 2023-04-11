from types import MethodType

import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from mmcls.apis import init_model

from .densenet import DenseNet3
from .resnet18_32x32 import ResNet18_32x32


def get_network(num_classes, name, use_gpu, checkpoint = None):

    if name == 'resnet18':
        net = ResNet18_32x32(num_classes=num_classes)

    elif name == 'densenet121':
        net = DenseNet3(depth=100,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0.0,
                        num_classes=num_classes)
    else:
        raise Exception('Unexpected Network Architecture!')

    if checkpoint is not None:
        if type(net) is dict:
            for subnet, checkpoint in zip(net.values(),
                                          checkpoint):
                if checkpoint is not None:
                    if checkpoint != 'none':
                        subnet.load_state_dict(torch.load(checkpoint),
                                               strict=False)
        else:
            try:
                net.load_state_dict(torch.load(checkpoint),
                                    strict=False)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(checkpoint)
                loaded_pth.pop('fc.weight')
                loaded_pth.pop('fc.bias')
                net.load_state_dict(loaded_pth, strict=False)
        print('Model Loading {} Completed!'.format(name))

    if use_gpu:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()
        torch.cuda.manual_seed(1)
        np.random.seed(1)
    cudnn.benchmark = True
    return net
