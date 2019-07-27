from utils import *
import sys

image_path = str(input("Enter path to image: "))
print(image_path)

image_tensor = get_image(image_path)
model, last_layer_weights = get_ResNet()
mask = get_cam(image_tensor, model, last_layer_weights)


def show_results():
    global mask, image_tensor
    get_plot(mask, image_tensor)
