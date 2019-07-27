
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_plot(mask, img_tensor):
    fig, ax = plt.subplots()
    ax.imshow(img_tensor.squeeze(axis=0))
    ax.imshow(mask, alpha=0.7, cmap='jet')
    plt.show()

    print("Done!")
