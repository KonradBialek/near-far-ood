import argparse
from warnings import warn
from utils.extractFeatures import extractFeatures
from utils.train import train
from utils.measure import measure


# model_options = ['resnet18', 'resnet34', 'renset50', 'renset101', 'renset152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
model_options = ['resnet18', 'densenet121']
OOD_options = ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin', 'mnist', 'fashionmnist', 'notmnist']
train_options = ['cifar10', 'cifar100', 'svhn', 'tin', 'mnist', 'fashionmnist']
method_options = ['knn', 'odin', 'msp', 'mls', 'react', 'lof', 'mds']
mode_options = ['train', 'extract', 'measure']

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
# common args
parser.add_argument('-n', '--nn', default="resnet18", type=str, choices=model_options,
                    help='(str) neural network (name in pytorch/vision:v0.14.0)')
parser.add_argument('-M', '--mode', default="measure", type=str, choices=mode_options,
                    help='(str) "train", "measure" or "extract')

# train args
parser.add_argument('-t', '--train_dataset', default="cifar10", type=str, choices=train_options,
                    help='(str) train dataset (name in torchvision.models or folder name in ./data)')
parser.add_argument('-s', '--la_steps', default=5, type=int,
                    help='(int) steps for Lookahead')
parser.add_argument('-a', '--la_alpha', default=0.5, type=float,
                    help='(float) alpha for Lookahead')
parser.add_argument('-N', '--n_holes', type=int, default=1,
                    help='(int) number of holes to cut out from image for Cutout')
parser.add_argument('-l', '--length', type=int, default=16,
                    help='(int) length of the holes for Cutout')

# measure args
parser.add_argument('-m', '--method', default="knn", type=str, choices=method_options,
                    help='(str) out-of-distribution method - lowercase')
parser.add_argument('-A', '--method_args', nargs='+', default=["50"], type=str,
                    help='''(list) out-of-distribution method arguments
order:
KNN: K
ODIN: temperature noise preprocessing normalization_dataset
preprocessing = ['true', '1', 't', 'y', 'yes'] (from this list if True, other if False)
normalization_dataset = ['cifar10', 'cifar100', 'dtd', 'svhn', 'tin', 'mnist', 'fashionmnist']
MSP: -
MLS: -
ReAct: percentile
LOF: n_neighbors
MDS: magnitude feature_type_list alpha_list reduce_dim_list preprocessing
preprocessing = ['true', '1', 't', 'y', 'yes'] (from this list if True, other if False)
examples:
KNN: 50
ODIN: 1000 1.4e-3 true cifar10
ReAct: 90
LOF: 20
MDS: 1.4e-3 mean 1 none true
''')
parser.set_defaults(argument=True)

# required in extract, optional in train and measure:
parser.add_argument('-c', '--checkpoint', type=str,
                    help='(str) checkpoint file for resuming training or extracting features (path/name in ./checkpoints)')
# required in extract and measure
parser.add_argument('-p', '--process_datasets', nargs='+', default=['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin', 'mnist', 'fashionmnist', 'notmnist'], type=str, choices=OOD_options,
                    help='(list[str]) datasets to extract features or measure distance starting with id dataset (names in torchvision.models or folder names in ./data)')


def main():
    """
    Chooses mode of program.

    Args as in python ./scr/main.py --help.
    """
    global args
    args = parser.parse_args()
    if args.mode == 'train':
        train(nn=args.nn, dataset=args.train_dataset, checkpoint=args.checkpoint, n_holes=args.n_holes, length=args.length, la_steps=args.la_steps, la_alpha=args.la_alpha)
    elif args.mode == 'extract':
        if args.checkpoint is not None:
            extractFeatures(nn=args.nn, datasets=args.process_datasets, checkpoint=args.checkpoint)
        else:
            print('Provide checkpoint file.')
    elif args.mode in ['measure', 'measure+']:
        if args.checkpoint is not None:
            measure(nn=args.nn, method=args.method, datasets=args.process_datasets, method_args=args.method_args, checkpoint=args.checkpoint, mode=args.mode)
        else:
            print('Provide checkpoint file.')
    else:
         warn("Wrong mode.")


if __name__ == '__main__':
    main()
