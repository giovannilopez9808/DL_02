from .pix2pix import VAE_pix2pix_model
from tensorflow.data import Dataset
from pandas import (DataFrame,
                    concat)
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
            dataset: Dataset) -> DataFrame:
        history_all = DataFrame()
        for epoch in range(1,
                           self.params["epochs"]+1):
            start = time()
            print(f"Epoch {epoch}")
            for i, (image_x, image_y) in dataset.shuffle(2022).take(10).enumerate():
                history = self.model.train_step(image_x,
                                                image_y)
                values = map(lambda loss: loss.numpy(),
                             history.values())
                history = DataFrame(values,
                                    index=history.keys())
                history = history.T
                history.index = [i]
                history_all = concat([history_all,
                                      history])
                if i % 1 == 0:
                    print(history)
                    print('.', end='')
            if (epoch + 1) % 5 == 0:
                _ = self.model.checkpoint.save()
                print(f'\nSaving checkpoint  epoch {epoch+1}')
            final_time = time()-start
            print('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                 final_time))
        return history_all
