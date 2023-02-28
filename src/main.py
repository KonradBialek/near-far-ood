import argparse
from warnings import warn
from utils.extractFeatures import extractFeatures
from utils.train import train
from utils.measure import measure


model_options = ['resnet18', 'resnet34', 'renset50', 'renset101', 'renset152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
OOD_options = ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin', 'mnist', 'fashionmnist', 'notmnist']
ID_options = ['cifar10', 'cifar100', 'places365', 'svhn', 'mnist', 'fashionmnist']
method_options = ['knn', 'odin', 'msp', 'mds', 'mls']
mode_options = ['train', 'extract', 'measure']

parser = argparse.ArgumentParser()
# common args
parser.add_argument('-n', '--nn', default="resnet18", type=str, choices=model_options,
                    help='neural network (name in pytorch/vision:v0.14.0)')
parser.add_argument('-M', '--mode', default="measure", type=str, choices=mode_options,
                    help='"train", "measure" or "extract')

# train args
parser.add_argument('-t', '--train_dataset', default="cifar10", type=str, choices=ID_options,
                    help='train dataset (name in torchvision.models or folder name in ./data)')
# parser.add_argument('-s', '--la_steps', default=5, type=int,
#                     help='steps for Lookahead')
# parser.add_argument('-a', '--la_alpha', default=0.5, type=float,
#                     help='alpha for Lookahead')
parser.add_argument('-N', '--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('-l', '--length', type=int, default=16,
                    help='length of the holes')

# extract args
parser.add_argument('-i', '--in_dataset', default="cifar10", type=str, choices=ID_options,
                    help='in-distribution dataset (name in torchvision.models or folder name in ./data)')
parser.add_argument('-o', '--ood_datasets', nargs='+', default=["mnist", "cifar100"], type=str, choices=OOD_options,
                    help='out-of-distribution datasets (names in torchvision.models or folder names in ./data)')
# required in extract, optional in train:
parser.add_argument('-c', '--checkpoint', type=str,
                    help='checkpoint file for resuming training or extracting features (path/name in ./checkpoints)')

# measure args
parser.add_argument('-m', '--method', default="knn", type=str, choices=method_options,
                    help='out-of-distribution method - lowercase')
parser.add_argument('-a', '--method_args', nargs='+', default=["50"], type=str,
                    help='out-of-distribution method arguments')
parser.add_argument('-f', '--feature_datasets', nargs='+', default=["mnist", "cifar10", "cifar100"], type=str, choices=OOD_options,
                    help='datasets to extract features starting with id dataset (names in torchvision.models or folder names in ./data)')
parser.set_defaults(argument=True)

def main():
    global args
    args = parser.parse_args()
    if args.mode == 'train':
        # train(args.nn, args.train_dataset, args.checkpoint, args.la_steps, args.la_alpha, args.n_holes, args.length)
        train(nn=args.nn, dataset=args.train_dataset, checkpoint=args.checkpoint, n_holes=args.n_holes, length=args.length)
    elif args.mode == 'extract':
        if args.checkpoint is not None:
            extractFeatures(nn=args.nn, in_dataset=args.in_dataset, ood_datasets=args.ood_datasets, checkpoint=args.checkpoint)
        else:
            print('Provide checkpoint file.')
    elif args.mode == 'measure':
            measure(nn=args.nn, method=args.method, feature_datasets=args.feature_datasets, method_args=args.method_args)
    else:
         warn("Wrong mode.")


if __name__ == '__main__':
    main()
