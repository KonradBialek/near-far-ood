import torch
from numpy import load
from torch.utils.data import DataLoader
import torchvision

from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config

from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset
from .udg_dataset import UDGDataset


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config)
        train = True if split == 'train' else False
        transform = preprocessor if split == 'train' else data_aux_preprocessor
        if dataset_config.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root='./data/images_classic', train=train, download=True, transform=transform)

        elif dataset_config.name == 'mnist':
            dataset = torchvision.datasets.MNIST(root='./data/images_classic', train=train, download=True, transform=transform)
        else:
            raise NotImplementedError

        sampler = None
        if dataset_config.num_gpus * dataset_config.num_machines > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            split_config.shuffle = False

        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers,
                                sampler=sampler)

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    ood_config = config.ood_dataset
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            # validation set
            train = True if split == 'train' else False
            transform = preprocessor if split == 'train' else data_aux_preprocessor
            if ood_config.name == 'cifar10_ood':
                dataset = torchvision.datasets.CIFAR10(root='./data/images_classic', train=train, download=True, transform=transform)

            elif ood_config.name == 'mnist_ood':
                dataset = torchvision.datasets.MNIST(root='./data/images_classic', train=train, download=True, transform=transform)
            else:
                raise NotImplementedError
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                train = True if split == 'train' else False
                transform = preprocessor if split == 'train' else data_aux_preprocessor
                print(dataset_name)
                if dataset_name == 'svhn' and split in ['val', 'nearood', 'farood']:
                    split = 'test'
                elif dataset_name == 'place365' and split in ['test', 'nearood', 'farood']:
                    split = 'val'

                if dataset_name == 'cifar10':
                    dataset = torchvision.datasets.CIFAR10(root='./data/images_classic', train=train, download=True, transform=transform)

                elif dataset_name == 'mnist':
                    dataset = torchvision.datasets.MNIST(root='./data/images_classic', train=train, download=True, transform=transform)

                elif dataset_name == 'tin':
                    dataset = torchvision.datasets.MNIST(root='./data/images_classic', train=train, download=True, transform=transform)

                elif dataset_name == 'svhn':
                    dataset = torchvision.datasets.SVHN(root='./data/images_classic', split=split, download=True, transform=transform)

                elif dataset_name == 'texture':
                    dataset = torchvision.datasets.DTD(root='./data/images_classic', split=split, download=True, transform=transform)

                elif dataset_name == 'cifar100':
                    dataset = torchvision.datasets.CIFAR100(root='./data/images_classic', train=train, download=True, transform=transform)

                elif dataset_name == 'place365':
                    dataset = torchvision.datasets.MNIST(root='./data/images_classic', train=train, download=True, transform=transform)
                    # dataset = torchvision.datasets.Places365(root='./data/images_classic', split=split, download=True, transform=transform)
                else:
                    raise NotImplementedError
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = DataLoader(dataset,
                            batch_size=split_config.batch_size,
                            shuffle=split_config.shuffle,
                            num_workers=dataset_config.num_workers)

    return dataloader
