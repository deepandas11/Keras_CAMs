
from keras.applications.resnet50 import ResNet50
from keras.models import Model


def get_ResNet():
    """
    Fetch a Resnet model trained on ImageNet
    Layer weights include weights from penultimate dense layer to final layer
    Model input is ?x224x224x3 image
    Model output is Activation maps and class probabilities

    """
    model = ResNet50(weights='imagenet')
    layer_weights = model.layers[-1].get_weights()[0]
    ResNetModel = Model(inputs=model.input,
                        outputs=(model.layers[-4].output, model.layers[-1].output))

    return ResNetModel, layer_weights