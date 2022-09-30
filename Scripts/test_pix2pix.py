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
predict = model.generator_g(image)
print(min(predict), max(predict))
# predict = model(image)
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(10, 5))
plot_image(ax1,
           image,
           "photo")
plot_image(ax2,
           predict,
           "predict")
plt.tight_layout()
plt.savefig("test_pix2pix.png")
