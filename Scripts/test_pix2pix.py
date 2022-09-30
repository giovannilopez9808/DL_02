from Modules.dataset import dataset_model
from Modules.pix2pix import pix2pix_model
from Modules.params import get_params
import matplotlib.pyplot as plt
from os.path import join
from sys import argv
from numpy import (
    expand_dims,
    array,
    min,
    max
)


def plot_image(ax: plt.subplot,
               image: array,
               label: str) -> None:
    image = image[0]
    image = (image+1)/2
    ax.set_title(label)
    ax.imshow(image)
    ax.axis("off")


params = get_params()
params["dataset"]["type"] = argv[1]
dataset = dataset_model(params)
model = pix2pix_model()
image = list(dataset.train.take(1))[0][0]
image = expand_dims(image,
                    axis=0)
predict_cat = model.generator_g(image)
predict_dog = model.generator_f(image)
# predict = model(image)
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,
                               figsize=(15, 5))
plot_image(ax1,
           image,
           "photo")
plot_image(ax2,
           predict_cat,
           "to cat")
plot_image(ax3,
           predict_dog,
           "to dog")
plt.tight_layout(pad=2)
plt.savefig(f"{argv[1]}_pix2pix.png")
