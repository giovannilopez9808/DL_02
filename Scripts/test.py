from Modules.pix2pix import VAE_pix2pix_model
from Modules.dataset import dataset_model
from Modules.params import get_params
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy import (
    linspace,
    arange,
    round,
    zeros,
)


def plot_latent_space(model: VAE_pix2pix_model,
                      input_size: tuple = (28, 28, 1),
                      n: int = 30,
                      figsize: int = 15,
                      scale: float = 1.,
                      latents_start: list = [0, 1]):
    # display a n*n 2D manifold of digits
    canvas = zeros((input_size[0]*n,
                    input_size[1]*n,
                    input_size[2]))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = linspace(-scale,
                      scale,
                      n)
    grid_y = linspace(-scale,
                      scale,
                      n)[::-1]
    z_sample = normal(0, 1,
                      (1, model.vae_x.latent_dim))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample[0][latents_start[0]] = xi
            z_sample[0][latents_start[1]] = yi
            x_decoded = model.vae_x.decoder_model(z_sample)
            img = x_decoded[0].numpy().reshape(input_size)
            pos_x_i = i*input_size[0]
            pos_x_j = (i+1)*input_size[0]
            pos_y_i = j*input_size[1]
            pos_y_j = (j+1)*input_size[1]
            canvas[pos_x_i: pos_x_j, pos_y_i: pos_y_j, :] = img

    plt.figure(figsize=(figsize, figsize))
    start_range = input_size[0] // 2
    end_range = n*input_size[0] + start_range
    pixel_range = arange(start_range,
                         end_range,
                         input_size[0])
    sample_range_x = round(grid_x,
                           1)
    sample_range_y = round(grid_y,
                           1)
    plt.xticks(pixel_range,
               sample_range_x)
    plt.yticks(pixel_range,
               sample_range_y)
    plt.xlabel(f"z[{latents_start[0]}]")
    plt.ylabel(f"z[{latents_start[1]}]")
    plt.imshow(canvas,
               cmap="Greys_r")
    plt.show()


params = get_params()
dataset = dataset_model(params)
model = VAE_pix2pix_model(**params["VAE"])
plot_latent_space(model,
                  input_size=params["VAE"]["input_dim"],
                  n=6,
                  latents_start=[20, 30],
                  scale=3)
