from keras.callbacks import ModelCheckpoint
from .pix2pix import VAE_pix2pix_model
from tensorflow.data import Dataset
from keras.optimizers import Adam
from os.path import join
from time import time


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

    def run(self,
            dog_dataset: Dataset,
            cat_dataset: Dataset) -> None:
        for epoch in range(1,
                           self.params["epochs"]+1):
            start = time()
            for i, (image_x, image_y) in enumerate(Dataset.zip((dog_dataset,
                                                                cat_dataset))):
                self.model.train_step(image_x,
                                      image_y)
                if i % 10 == 0:
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
