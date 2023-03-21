import argparse
import os
import shutil
from sklearn.model_selection import train_test_split

dataset_options = ['notmnist']
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="notmnist", type=str, choices=dataset_options,
                    help='(str) dataset (name in torchvision.models or folder name in ./data)')
parser.add_argument('-s', '--split', default=.2, type=float,
                    help='(float) val ratio in train-val ratio')


args = parser.parse_args()
class_labels = os.listdir(f'data/{args.dataset}_')

image_paths_by_class = {}
for label in class_labels:
    image_paths_by_class[label] = [f'data/{args.dataset}_/{label}/{filename}' for filename in os.listdir(f'data/{args.dataset}_/{label}')]

if args.split < 1:
    train_dir = f'data/{args.dataset[:-1]}/train'
    os.makedirs(train_dir, exist_ok=True)
if args.split > 0:
    val_dir = f'data/{args.dataset[:-1]}/val'
    os.makedirs(val_dir, exist_ok=True)

train_paths_by_class, val_paths_by_class = {}, {}
for label in class_labels:
    train_paths, val_paths = train_test_split(image_paths_by_class[label], test_size=args.split, random_state=42)
    train_paths_by_class[label] = train_paths
    val_paths_by_class[label] = val_paths

for label in class_labels:
    os.makedirs(f'{train_dir}/{label}', exist_ok=True)
    for image_path in train_paths_by_class[label]:
        shutil.copy(image_path, f'{train_dir}/{label}/{os.path.basename(image_path)}')

for label in class_labels:
    os.makedirs(f'{val_dir}/{label}', exist_ok=True)
    for image_path in val_paths_by_class[label]:
        shutil.copy(image_path, f'{val_dir}/{label}/{os.path.basename(image_path)}')
