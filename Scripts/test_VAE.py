from Modules.dataset import dataset_model
from Modules.params import get_params
import matplotlib.pyplot as plt
from Modules.VAE2 import VAE2
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
model = VAE2(params)

image = list(dataset.train.take(1))[0]
print(image.shape)
pred = model.vae(image)
fig, axs = plt.subplots(1, 2,
                        figsize=(10, 5))
axs = axs.flatten()
plot_image(axs[0],
           image,
           "photo")
plot_image(axs[1],
           pred,
           "decoder dog")
plt.tight_layout(pad=2)
filename = f"VAE_{argv[1]}_train.png"
filename = join("..",
                filename)
plt.savefig(filename,
            dpi=400)

image = list(dataset.test.take(1))[0]
pred = model.vae(image)
fig, axs = plt.subplots(1, 2,
                        figsize=(10, 5))
axs = axs.flatten()
plot_image(axs[0],
           image,
           "photo")
plot_image(axs[1],
           pred,
           "decoder dog")
plt.tight_layout(pad=2)
filename = f"VAE_{argv[1]}_test.png"
filename = join("..",
                filename)
plt.savefig(filename,
            dpi=400)
