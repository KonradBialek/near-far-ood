# from types import MethodType

# import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from mmcls.apis import init_model

import openood.utils.comm as comm

from .bit import KNOWN_MODELS
from .conf_branch_net import ConfBranchNet
from .csi_net import CSINet
from .de_resnet18_256x256 import AttnBasicBlock, BN_layer, De_ResNet18_256x256
from .densenet import DenseNet3
from .draem_net import DiscriminativeSubNetwork, ReconstructiveSubNetwork
from .dropout_net import DropoutNet
from .dsvdd_net import build_network
from .godin_net import GodinNet
from .lenet import LeNet
from .mcd_net import MCDNet
from .openmax_net import OpenMax
from .patchcore_net import PatchcoreNet
from .projection_net import ProjectionNet
from .react_net import ReactNet
from .resnet18_32x32 import ResNet18_32x32
from .resnet18_64x64 import ResNet18_64x64
from .resnet18_224x224 import ResNet18_224x224
from .resnet18_256x256 import ResNet18_256x256
from .resnet50 import ResNet50
from .udg_net import UDGNet
from .wrn import WideResNet


def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)

    elif network_config.name == 'lenet':
        net = LeNet(num_classes=num_classes, num_channel=3)


    elif network_config.name == 'conf_branch_net':

        backbone = get_network(network_config.backbone)
        net = ConfBranchNet(backbone=backbone, num_classes=num_classes)

    elif network_config.name == 'dsvdd':
        net = build_network(network_config.type)

    elif network_config.name == 'projectionNet':
        backbone = get_network(network_config.backbone)
        net = ProjectionNet(backbone=backbone, num_classes=2)

    elif network_config.name == 'dropout_net':
        backbone = get_network(network_config.backbone)
        net = DropoutNet(backbone=backbone, dropout_p=network_config.dropout_p)

    elif network_config.name == 'simclr_net':
        # backbone = get_network(network_config.backbone)
        # net = SimClrNet(backbone, out_dim=128)
        from .temp import SSLResNet
        net = SSLResNet()
        net.encoder = nn.DataParallel(net.encoder).cuda()

    elif network_config.name == 'rd4ad_net':
        encoder = get_network(network_config.backbone)
        bn = BN_layer(AttnBasicBlock, 2)
        decoder = De_ResNet18_256x256()
        net = {'encoder': encoder, 'bn': bn, 'decoder': decoder}
    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        if type(net) is dict:
            for subnet, checkpoint in zip(net.values(),
                                          network_config.checkpoint):
                if checkpoint is not None:
                    if checkpoint != 'none':
                        subnet.load_state_dict(torch.load(checkpoint),
                                               strict=False)
        elif network_config.name == 'bit' and not network_config.normal_load:
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == 'vit':
            pass
        else:
            try:
                net.load_state_dict(torch.load(network_config.checkpoint),
                                    strict=False)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(network_config.checkpoint)
                loaded_pth.pop('fc.weight')
                loaded_pth.pop('fc.bias')
                net.load_state_dict(loaded_pth, strict=False)
        print('Model Loading {} Completed!'.format(network_config.name))
    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()
        torch.cuda.manual_seed(1)
        np.random.seed(1)
    cudnn.benchmark = True
    return net
