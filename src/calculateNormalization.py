import numpy as np
import argparse
from PIL import Image
from utils.utils import getDataset, getShape


'''
Calculates mean and standard deviation of pixels in images.

Args as in python ./scr/calculateNormalization.py --help.
'''

dataset_options = ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin', 'mnist', 'fashionmnist', 'notmnist']
builtin_datasets = ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'mnist', 'fashionmnist']
mode_options = ['train', 'all']
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="cifar10", type=str, choices=dataset_options,
                    help='(str) dataset (name in torchvision.models or folder name in ./data)')
parser.add_argument('-m', '--mode', default="train", type=str, choices=mode_options,
                    help='(str) calculate normalization for train subset or all images in dataset')

args = parser.parse_args()

trainset, valset, testset, _ = getDataset(args.dataset)

if args.dataset in builtin_datasets and args.dataset != 'dtd':
    if args.mode == 'all': # if use all images - concatenate datasets
        if valset is not None:
            if testset is not None:
                data = np.concatenate((trainset.data, valset.data, testset.data), axis=0)
            else:
                data = np.concatenate((trainset.data, valset.data), axis=0)
        elif testset is not None:
            data = np.concatenate((trainset.data, testset.data), axis=0)
    else:
        data = trainset.data
    data = data / 255

else:
    if args.dataset == 'dtd':
        if args.mode == 'all': 
            raise NotImplementedError("'all' mode not implemented")
        else:
            filelist = []
            with open('data/dtd/dtd/labels/train1.txt') as file:
                for line in file:
                    filelist.append('data/dtd/dtd/images/'+line[:-1])
    else:
        filelist = [trainset.imgs[x][0] for x in range(len(trainset.imgs))]
        if args.mode == 'all':
            if valset is not None:
                vallist = [valset.imgs[x][0] for x in range(len(valset.imgs))]
                if testset is not None:
                    testlist = [testset.imgs[x][0] for x in range(len(testset.imgs))]
                    filelist = np.concatenate((filelist, vallist, testlist), axis=0)
                else:
                    filelist = np.concatenate((filelist, vallist), axis=0)
            elif testset is not None:
                testlist = [testset.imgs[x][0] for x in range(len(testset.imgs))]
                filelist = np.concatenate((filelist, testlist), axis=0)

    imgs = []
    for fname in filelist:
        img = Image.open(fname)
        if img.mode != "RGB":
            img.convert("RGB")

        img = img.resize(getShape(args.dataset)[:-1])
        img_arr = np.array(img)

        imgs.append(img_arr)
    data = np.array(imgs)
    data = data / 255

if args.dataset == 'svhn':
    data = np.transpose(data, (0, 2, 3, 1))

mean = data.mean(axis = (0,1,2)) 
std = data.std(axis = (0,1,2))
print(f"{mean}, {std}")
