import numpy as np
import cv2
from keras.models import Model


def get_cam(img_tensor, model, last_layer_weights):

    activ_maps, class_probs = model.predict(img_tensor)
    activ_maps = np.squeeze(activ_maps)  # ?x7x7x2048 -> 7x7x2048
    pred = np.argmax(class_probs)
    class_weights = last_layer_weights[:, pred]  # 2048x1000 -> 2048,
    mask = np.dot(activ_maps.reshape(7*7, 2048), class_weights).reshape(7,7)
    mask = cv2.resize(mask, (224,224))

    return mask

