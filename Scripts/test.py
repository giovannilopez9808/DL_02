from dataset import dataset_model
import matplotlib.pyplot as plt
from params import get_params
from model import model
import numpy as np


def plot_latent_space(vae,
                      input_size=(28, 28, 1),
                      n=30,
                      figsize=15,
                      scale=1.,
                      latents_start=[0, 1]):
    # display a n*n 2D manifold of digits
    canvas = np.zeros((input_size[0]*n, input_size[1]*n, input_size[2]))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    z_sample = np.random.normal(0, 1, (1, vae.latent_dim))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample[0][latents_start[0]
                        ], z_sample[0][latents_start[1]] = xi, yi
            x_decoded = vae.generate(z_sample)
            img = x_decoded[0].numpy().reshape(input_size)
            canvas[i*input_size[0]: (i + 1)*input_size[0],
                   j*input_size[1]: (j + 1)*input_size[1],
                   :] = img

    plt.figure(figsize=(figsize, figsize))
    start_range = input_size[0] // 2
    end_range = n*input_size[0] + start_range
    pixel_range = np.arange(start_range, end_range, input_size[0])
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[{}]".format(latents_start[0]))
    plt.ylabel("z[{}]".format(latents_start[1]))
    plt.imshow(canvas, cmap="Greys_r")
    plt.show()


params = get_params()
dataset = dataset_model(params)
model = model(params)
model.model.load_weights("../Models/Checkpoint_model.h5")
plot_latent_space(model.model,
                  input_size=(256, 256, 3),
                  n=6,
                  latents_start=[20, 30],
                  scale=3)
