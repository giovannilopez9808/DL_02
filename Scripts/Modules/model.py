from keras.callbacks import ModelCheckpoint
from .pix2pix import VAE_pix2pix_model
from tensorflow.data import Dataset
from keras.optimizers import Adam
from tensorflow import function
from os.path import join
from time import time
from sys import exit


class CycleGAN_model:
    def __init__(self,
                 params: dict) -> None:
        self.params = params
        self._get_model(params)

    def _get_model(self,
                   params: dict) -> None:
        self.model = VAE_pix2pix_model(
            **params["VAE"]
        )

    # @function
    def run(self,
            dog_dataset: Dataset,
            cat_dataset: Dataset) -> None:
        for epoch in range(1,
                           self.params["epochs"]+1):
            start = time()
            print(f"Epoch {epoch}")
            for i, (image_x, image_y) in enumerate(Dataset.zip((dog_dataset,
                                                                cat_dataset))):
                history = self.model.train_step(image_x,
                                                image_y)
                print(history["loss_vae_x"].numpy())
                exit(1)
                if i % 100 == 0:
                    print('.', end='')
            # Using a consistent image (sample_horse)
            # so that the progress of the model
            # is clearly visible.
            # generate_images(generator_g, sample_horse)
            if (epoch + 1) % 5 == 0:
                _ = self.model.checkpoint.save()
                print(f'Saving checkpoint  epoch {epoch+1}')
            final_time = time()-start
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               final_time))
