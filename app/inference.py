import numpy as np
import pickle
import os
from sklearn import datasets
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from skimage import color
from skimage.transform import resize
from PIL import Image, ImageOps

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SquareClassifier':
            from .square_algo import SquareClassifier
            return SquareClassifier
        if name == 'Square':
            from .square_algo import Square
            return Square
        return super().find_class(module, name)

def load_dataset():
    X, _ = datasets.load_digits(return_X_y=True)
    return X

def load_model(model_path):
    file = open(model_path, 'rb')
    model = CustomUnpickler(file).load()
    file.close()
    return model

def proccess_standard_scaler(dataset, X):
    scaler = StandardScaler()
    return scaler.fit_transform(np.concatenate((dataset,[X]), axis=0))[-1]

def proccess_t_sne(dataset, X, component):
    tsne = manifold.TSNE(n_components=component, init='pca', random_state=0)
    return tsne.fit_transform(np.concatenate((dataset,[X]), axis=0))[-1]

def transform_image(image):
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    inverted_image = ImageOps.invert(background)
    gray_image = inverted_image.convert('L')
    resize_image = gray_image.resize((8,8))
    return np.array(resize_image)

def inference(model_param, image):
    dataset = load_dataset()
    model = load_model(model_path=os.path.join(os.getcwd(), "app", "models", model_param["file"]))
    prepare_image = transform_image(image).reshape((64,))
    
    if model_param["std"]:
        prepare_image = proccess_standard_scaler(dataset, prepare_image)
    if model_param["tsne"]:
        prepare_image = proccess_t_sne(dataset, prepare_image, int(model_param["tsne_component"]))
        
    try:
        result = model.predict([prepare_image])
        return str(result[0])
    except Exception as e:
        print(e)
        return "-1"