import os
from matplotlib import pyplot as plt

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.utils import save_scores

'''
Loops over .npz files in ./features directory. Reduces dimensions of extracted features of datasets with PCA to 50 dimensions, then t-SNE.
'''
n_components = 50 

for file in os.listdir('./features/'):
    if file.endswith('.npz'):
        features = np.load('./features/'+file)['features']
        logits = np.load('./features/'+file)['logits']
        labels = np.load('./features/'+file)['labels']
        if features.ndim > 1 and not file.endswith('setup.npz'):
            if features.shape[1] > 2:
                if n_components < features.shape[0]:
                    features = preprocessing.normalize(features, norm='l2')
                    pca = PCA(n_components=n_components, whiten=True)
                    features = pca.fit_transform(features)
                    features = preprocessing.normalize(features, norm='l2')

                    logits = preprocessing.normalize(logits, norm='l2')
                    logits = pca.fit_transform(logits)
                    logits = preprocessing.normalize(logits, norm='l2')

                tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
                features = tsne.fit_transform(features)
                logits = tsne.fit_transform(logits)
                save_scores(features, logits, labels, save_name=file[:-4]+'_tsne', save_dir='./features')
            else:
                print("Data have invalid shape.")
        else:
            print("Data have invalid shape.")
