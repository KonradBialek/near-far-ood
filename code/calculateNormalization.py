from pathlib import Path
import numpy as np
import argparse
from PIL import Image
from utils import dataloader

OOD_options = ['cifar10', 'cifar100', 'dtd', 'places365', 'svhn', 'tin', 'mnist', 'fashionmnist', 'notmnist']
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="cifar10", type=str, choices=OOD_options,
                    help='dataset (name in torchvision.models or folder name in ./data)')

args = parser.parse_args()
imageFilesDir = Path('../datasets/dataset/train/colour')
files = list(imageFilesDir.rglob('*.jpg'))
dataloader = dataloader(args.dataset, normalization, shape[:2], None, True, None, n_holes, length)
# Since the std can't be calculated by simply finding it for each image and averaging like  
# the mean can be, to get the std we first calculate the overall mean in a first run then  
# run it again to get the std.

mean = np.array([0.,0.,0.])
stdTemp = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])

numSamples = len(files)

for i in range(numSamples):
    im = np.asarray(Image.open(str(files[i])).convert("RGB"))
    im = im / 255.
    
    for j in range(3):
        mean[j] += np.mean(im[:,:,j])

mean = (mean/numSamples)

for i in range(numSamples):
    im = np.asarray(Image.open(str(files[i])).convert("RGB"))
    im = im / 255.
    
    for j in range(3):
        stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

std = np.sqrt(stdTemp/numSamples)

print(mean)
print(std)
