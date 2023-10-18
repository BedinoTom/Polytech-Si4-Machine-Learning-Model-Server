import numpy as np
import pickle
import os
from sklearn import tree
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage import color
from skimage.transform import resize

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

def transform_image(np_image):
    if len(np_image.shape) == 3 and (np_image.shape[2] ==3):
        np_image = color.rgb2gray(np_image)
    if len(np_image.shape) == 3 and (np_image.shape[2] == 4):
        np_image = color.rgb2gray(color.rgba2rgb(np_image))
    if np_image.shape[0] == 8 and np_image.shape[1] == 8:
        return np_image
    return resize(np_image, (8, 8),
                       anti_aliasing=True)

def inference(model_param, np_image):
    model = load_model(model_path=os.path.join(os.getcwd(), "app", "models", model_param["file"]))
    prepare_image = transform_image(np_image).reshape((64,))
    
    if model_param["std"]:
        prepare_image = proccess_standard_scaler(prepare_image)
    if model_param["tsne"]:
        prepare_image = proccess_t_sne(prepare_image, int(model_param["tsne_component"]))
        
    try:
        result = model.predict([prepare_image])
        return str(result[0])
    except Exception as e:
        print(e)
        return "-1"