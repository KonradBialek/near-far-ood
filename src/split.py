import argparse
import os
import shutil
import random
from sklearn.model_selection import train_test_split

dataset_options = ['notmnist']
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="notmnist", type=str, choices=dataset_options,
                    help='(str) dataset (name in torchvision.models or folder name in ./data)')
parser.add_argument('-s', '--split', default=.2, type=float,
                    help='(float) test ratio in train-test ratio')


args = parser.parse_args()
class_labels = os.listdir(f'data/{args.dataset}')

image_paths_by_class = {}
for label in class_labels:
    image_paths_by_class[label] = [f'data/{args.dataset}/{label}/{filename}' for filename in os.listdir(f'data/{args.dataset}/{label}')]

if args.split < 1:
    train_dir = f'data/{args.dataset.upper()}/train'
    os.makedirs(train_dir, exist_ok=True)
if args.split > 0:
    test_dir = f'data/{args.dataset.upper()}/test'
    os.makedirs(test_dir, exist_ok=True)

train_paths_by_class, test_paths_by_class = {}, {}
for label in class_labels:
    train_paths, test_paths = train_test_split(image_paths_by_class[label], test_size=args.split, random_state=42)
    train_paths_by_class[label] = train_paths
    test_paths_by_class[label] = test_paths

for label in class_labels:
    os.makedirs(f'{train_dir}/{label}', exist_ok=True)
    for image_path in train_paths_by_class[label]:
        shutil.copy(image_path, f'{train_dir}/{label}/{os.path.basename(image_path)}')

for label in class_labels:
    os.makedirs(f'{test_dir}/{label}', exist_ok=True)
    for image_path in test_paths_by_class[label]:
        shutil.copy(image_path, f'{test_dir}/{label}/{os.path.basename(image_path)}')
