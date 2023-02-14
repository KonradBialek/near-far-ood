import argparse
from utils import train, measure
parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')


parser.add_argument('-n', '--nn', default="resnet18", type=str,
                    help='neural network (pytorch name)')
parser.add_argument('-c', '--checkpoint', default="", type=str,
                    help='checkpoint file (name in ./checkpoints)')
parser.add_argument('-m', '--method', default="knn", type=str,
                    help='out-of-distribution method')
parser.add_argument('-d', '--mode', default="measure", type=str,
                    help='train or measure')
parser.add_argument('-i', '--in_dataset', default="cifar10", type=str,
                    help='in-distribution dataset (folder name in ./data)')
parser.add_argument('-o', '--out_datasets', default="mnist cifar100", type=str,
                    help='out-of-distribution datasets (folder names in ./data)')
parser.set_defaults(argument=True)

def main():
    global args
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.nn, args.in_dataset, args.checkpoint)
    elif args.mode == 'measure':
        measure(args.nn, args.methd, args.in_dataset, args.out_datasets, args.checkpoint)


if __name__ == '__main__':
    main()
