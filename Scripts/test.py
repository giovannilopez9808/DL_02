from Modules.VAE_pix2pix import VAE_pix2pix_model
from Modules.dataset import dataset_model
from Modules.params import get_params
import matplotlib.pyplot as plt
from os.path import join
from numpy import array


def plot_image(ax: plt.subplot,
               image: array,
               label: str) -> None:
    image = image[0]
    image = (image+1)/2
    ax.set_title(label)
    ax.imshow(image)
    ax.axis("off")


params = get_params()
params["dataset"]["type"] = "all"
dataset = dataset_model(params)
model = VAE_pix2pix_model(params)
dog, cat = list(dataset.train.take(1))[0]
pred_dog = model.vae_dog(dog)
gen_cat = model.generator_cat(pred_dog)
same_dog = model.generator_dog(pred_dog)
pred_cat = model.vae_cat(cat)
gen_dog = model.generator_dog(pred_cat)
same_cat = model.generator_cat(pred_cat)
fig, axs = plt.subplots(2, 4,
                        figsize=(15, 10))
axs = axs.flatten()
plot_image(axs[0],
           dog,
           "photo")
plot_image(axs[1],
           pred_dog,
           "decoder dog")
plot_image(axs[2],
           gen_cat,
           "cat generate")
plot_image(axs[3],
           same_dog,
           "same dog")
plot_image(axs[4],
            cat,
            "cat")
plot_image(axs[5],
           pred_cat,
           "decoder cat")
plot_image(axs[6],
           gen_dog,
           "dog generate")
plot_image(axs[7],
           same_cat,
           "same dog")
plt.tight_layout(pad=2)
filename = "train_cycleGAN.png"
filename = join(params["path graphics"],
                filename)
plt.savefig(filename)

dog, cat = list(dataset.test.take(1))[0]
pred_dog = model.vae_dog(dog)
gen_cat = model.generator_cat(pred_dog)
same_dog = model.generator_dog(pred_dog)
pred_cat = model.vae_cat(cat)
gen_dog = model.generator_dog(pred_cat)
same_cat = model.generator_cat(pred_cat)
fig, axs = plt.subplots(2, 4,
                        figsize=(15, 10))
axs = axs.flatten()
plot_image(axs[0],
           dog,
           "photo")
plot_image(axs[1],
           pred_dog,
           "decoder dog")
plot_image(axs[2],
           gen_cat,
           "cat generate")
plot_image(axs[3],
           same_dog,
           "same dog")
plot_image(axs[4],
            cat,
            "cat")
plot_image(axs[5],
           pred_cat,
           "decoder cat")
plot_image(axs[6],
           gen_dog,
           "dog generate")
plot_image(axs[7],
           same_cat,
           "same dog")
plt.tight_layout(pad=2)
filename = "test_cycleGAN.png"
filename = join(params["path graphics"],
                filename)
plt.savefig(filename)
