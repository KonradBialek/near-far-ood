import os
from matplotlib import pyplot as plt

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.utils import save_scores


n_components = 50 

def tsne_compute(x, n_components=50):
    if n_components < x.shape[0]:
        pca = PCA(n_components=50)
        x = pca.fit_transform(x)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
    tsne_pos = tsne.fit_transform(x)

    return tsne_pos


# dataset_list = [
#     'mnist', 'svhn', 'notmnist', 'fashionmnist', 'dtd', 'cifar10',
#     'tin', 'places365', 'cifar100'
# ]

for file in os.listdir('./features/'):
    if file.endswith('.npz'):
        data = np.load('./features/'+file)['data']
        labels = np.load('./features/'+file)['labels']
        if data.ndim > 1:
            if n_components < data.shape[0]:
                data = preprocessing.normalize(data, norm='l2')
                pca = PCA(n_components=n_components, whiten=True)
                data = pca.fit_transform(data)
                data = preprocessing.normalize(data, norm='l2')
            tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
            data = tsne.fit_transform(data)
            print(len(data))
            save_scores(data, labels, save_name=file[:-4]+'_tsne', save_dir='./features')




# dataset_list = [
#     'mnist', 'cifar10', 'cifar100'
# ]
# dirname = './results/scores'
# sample_rate = 0.1

# featstat_list, idx_list = [], []
# for idx, dataset in enumerate(dataset_list):
#     file_name = os.path.join(dirname, f'{dataset}.npz')
#     featstat_sublist = np.load(file_name)['conf']
#     label_list = np.load(file_name)['label']
#     # selection:
#     num_samples = len(featstat_sublist)
#     index_list = np.arange(num_samples)
#     index_select = np.random.choice(index_list,
#                                     int(sample_rate * num_samples),
#                                     replace=False)
#     featstat_list.extend(featstat_sublist[index_select])
#     idx_list.extend(idx * np.ones(len(index_select)))

# print(featstat_list)
# featstat_list, index_list = np.array(featstat_list), np.array(idx_list)
# tsne_pos_lowfeat = tsne_compute(featstat_list)
# print(len(tsne_pos_lowfeat))
# np.save(os.path.join(dirname, 'tsne_pos_lowfeat'), tsne_pos_lowfeat)
# np.save(os.path.join(dirname, 'idx'), idx_list)








# dirname = './results/scores'
# sample_rate = 0.1

# featstat_list, idx_list = [], []
# for idx, dataset in enumerate(dataset_list):
#     file_name = os.path.join(dirname, f'{dataset}.npz')
#     featstat_sublist = np.load(file_name)['conf']
#     label_list = np.load(file_name)['label']
#     # selection:
#     num_samples = len(featstat_sublist)
#     index_list = np.arange(num_samples)
#     index_select = np.random.choice(index_list,
#                                     int(sample_rate * num_samples),
#                                     replace=False)
#     featstat_list.extend(featstat_sublist[index_select])
#     idx_list.extend(idx * np.ones(len(index_select)))

# featstat_list, index_list = np.array(featstat_list), np.array(idx_list)
# tsne_pos_lowfeat = tsne_compute(featstat_list)
# np.save(os.path.join(dirname, 'tsne_pos_lowfeat'), tsne_pos_lowfeat)
# np.save(os.path.join(dirname, 'idx'), idx_list)
