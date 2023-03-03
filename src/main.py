import argparse
from warnings import warn
from utils.extractFeatures import extractFeatures
from utils.train import train
from utils.measure import measure, measure_


model_options = ['resnet18', 'resnet34', 'renset50', 'renset101', 'renset152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
OOD_options = ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin', 'mnist', 'fashionmnist', 'notmnist']
train_options = ['cifar10', 'cifar100', 'mnist']
method_options = ['knn', 'odin', 'msp', 'mds', 'mls']
mode_options = ['train', 'extract', 'measure']

parser = argparse.ArgumentParser()
# common args
parser.add_argument('-n', '--nn', default="resnet18", type=str, choices=model_options,
                    help='neural network (name in pytorch/vision:v0.14.0)')
parser.add_argument('-M', '--mode', default="measure", type=str, choices=mode_options,
                    help='"train", "measure" or "extract')
parser.add_argument('-f', '--process_datasets', nargs='+', default=["mnist", "cifar10", "cifar100"], type=str, choices=OOD_options,
                    help='datasets to extract features or measure distance starting with id dataset (names in torchvision.models or folder names in ./data)')

# train args
parser.add_argument('-t', '--train_dataset', default="cifar10", type=str, choices=train_options,
                    help='train dataset (name in torchvision.models or folder name in ./data)')
parser.add_argument('-s', '--la_steps', default=5, type=int,
                    help='steps for Lookahead')
parser.add_argument('-a', '--la_alpha', default=0.5, type=float,
                    help='alpha for Lookahead')
parser.add_argument('-N', '--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('-l', '--length', type=int, default=16,
                    help='length of the holes')

# required in extract, optional in train and measure:
parser.add_argument('-c', '--checkpoint', type=str,
                    help='checkpoint file for resuming training or extracting features (path/name in ./checkpoints)')

# measure args
parser.add_argument('-m', '--method', default="knn", type=str, choices=method_options,
                    help='out-of-distribution method - lowercase')
parser.add_argument('-A', '--method_args', nargs='+', default=["50"], type=str,
                    help='out-of-distribution method arguments')
parser.set_defaults(argument=True)

def main():
    global args
    args = parser.parse_args()
    if args.mode == 'train':
        train(nn=args.nn, dataset=args.train_dataset, checkpoint=args.checkpoint, n_holes=args.n_holes, length=args.length, la_steps=args.la_steps, la_alpha=args.la_alpha)
    elif args.mode == 'extract':
        if args.checkpoint is not None:
            extractFeatures(nn=args.nn, datasets=args.process_datasets, checkpoint=args.checkpoint)
        else:
            print('Provide checkpoint file.')
    elif args.mode == 'measure':
            if args.method in ['msp', 'knn']:
                measure(nn=args.nn, method=args.method, datasets=args.process_datasets, method_args=args.method_args)
            else:
                if args.checkpoint is not None:
                    measure_(nn=args.nn, method=args.method, datasets=args.process_datasets, method_args=args.method_args, checkpoint=args.checkpoint)
                else:
                    print('Provide checkpoint file.')
    else:
         warn("Wrong mode.")


if __name__ == '__main__':
    main()
