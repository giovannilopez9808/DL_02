from Modules.dataset import dataset_model
from Modules.params import get_params
import matplotlib.pyplot as plt
from Modules.VAE import VAE
from os.path import join
from numpy import array
from sys import argv


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
model = VAE(**params["VAE"])
filename = f"{argv[1]}_model.h5"
filename = join(params["path models"],
                filename)
model.load_weights(filename)
image = list(dataset.train.take(1))[0]
predict = model(image)
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(10, 5))
plot_image(ax1,
           image,
           "photo")
plot_image(ax2,
           predict,
           "predict")
plt.tight_layout()
plt.savefig("test_VAE.png")
