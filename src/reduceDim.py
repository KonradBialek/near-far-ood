import os
from matplotlib import pyplot as plt

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.utils import save_scores


n_components = 50 

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
            save_scores(data, labels, save_name=file[:-4]+'_tsne', save_dir='./features')
        else:
            print("Data has too much dimensions.")
