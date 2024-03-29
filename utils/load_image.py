import numpy as np
import ast
import cv2

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

def get_image(img_path):
    """
    Loads image from image path. Convert PIL image to 3D tensor
    Preprocess using Keras API and return it.
    :param img_path: path to image file
    """
    img = image.load_img(img_path, target_size=(224,224)) 
    x = image.img_to_array(img)  # PIL Image to 3D tensor
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x)  # RGB -> BGR. subtract mean ImageNet pixel
    
    # return 4D preprocessed tensor
    return x
