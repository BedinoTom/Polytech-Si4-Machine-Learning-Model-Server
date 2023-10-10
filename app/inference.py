import numpy as np
import pickle
from sklearn import tree
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    file = open(model_path, 'rb')
    model = pickle.load(file)
    file.close()
    return model

def proccess_standard_scaler(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def proccess_t_sne(X, component):
    tsne = manifold.TSNE(n_components=component, init='pca', random_state=0)
    return tsne.fit_transform(X)

def inference(model_param, np_image):
    pass