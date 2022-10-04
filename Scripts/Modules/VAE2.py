from keras.losses import BinaryCrossentropy
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from tabulate import tabulate
from tensorflow.train import (
    CheckpointManager,
    Checkpoint
)
from os.path import join
from tensorflow import (
    GradientTape,
    reduce_mean,
    reduce_sum,
    function,
    square,
    exp,
)
from time import time
from .VAE import VAE
from pandas import (
    DataFrame,
    concat
)
from numpy import array
from keras import Model
loss_obj = BinaryCrossentropy(from_logits=True)
OUTPUT_CHANNELS = 3
LAMBDA = 10


def plot_image(ax: plt.subplot,
               image: array,
               label: str) -> None:
    image = image[0]
    image = (image+1)/2
    ax.set_title(label)
    ax.imshow(image)
    ax.axis("off")


class VAE2(Model):
    def __init__(self,
                 params: dict,
                 image_type:str="dog",
                 **kwargs) -> None:
        super(VAE2,
              self).__init__(**kwargs)
        self.params = params
        self.vae = VAE(**params["VAE"])
        self._create_checkpoint(image_type)

    def _create_checkpoint(self,
                           image_type:str) -> None:
        checkpoint_path = join(self.params["path checkpoint"],
                               image_type)
        ckpt = Checkpoint(
            vae=self.vae,
        )
        self.checkpoint = CheckpointManager(
            ckpt,
            checkpoint_path,
            max_to_keep=1
        )
        # if a checkpoint exists, restore the latest checkpoint.
        if self.checkpoint.latest_checkpoint:
            ckpt.restore(self.checkpoint.latest_checkpoint)
            print(f'Model VAE {image_type}')

    @function
    def train_step(self,
                   dog: array) -> dict:
        '''
        '''
        with GradientTape(persistent=True) as tape:
            # predict
            pred = self.vae.encoder_model(dog)
            # predict
            zd, zd_mean, zd_log_var = self.vae.sampler_model(pred)
            pred = self.vae.decoder_model(zd)
            # loss
            rd_loss = self.vae.r_loss_factor * self.vae.mae(
                dog,
                pred
            )
            kld_loss = 1 + zd_log_var - square(zd_mean) - exp(zd_log_var)
            kld_loss = -0.5*kld_loss
            kld_loss = reduce_mean(reduce_sum(kld_loss,
                                              axis=1))
            total_loss = rd_loss + kld_loss
        # Calculate the gradients for generator and discriminator
        vae_gradients = tape.gradient(total_loss,
                                      self.vae.trainable_weights)
        self.vae.optimizer.apply_gradients(
            zip(vae_gradients,
                self.vae.trainable_weights))
        self.vae.total_loss_tracker.update_state(total_loss)
        self.vae.reconstruction_loss_tracker.update_state(rd_loss)
        self.vae.kl_loss_tracker.update_state(kld_loss)
        loss_history = {
            "kl": self.vae.kl_loss_tracker.result(),
        }
        return loss_history

    def fit(self,
            dataset: Dataset,
            epochs: int) -> DataFrame:
        history_all = DataFrame()
        image_test = list(dataset.test.take(1))[0]
        for epoch in range(1,
                           epochs+1):
            start = time()
            print(f"Epoch {epoch}")
            history_epoch = DataFrame()
            for i, image in dataset.train.enumerate():
                i = i.numpy()
                history = self.train_step(image)
                values = map(lambda loss: loss.numpy(),
                             history.values())
                history = DataFrame(values,
                                    index=history.keys())
                history = history.T
                history.index = [i]
                history_epoch = concat([history_epoch,
                                        history])
            history_epoch = DataFrame(history_epoch.mean())
            history_epoch.index=[epoch]
            history_all = concat([history_all,
                                  history_epoch])
            print(tabulate(history_epoch,
                           headers=history_epoch.columns))
            decoder = self.vae(image_test)
            fig, axs = plt.subplots(1, 2,
                                    figsize=(10, 5))
            axs = axs.flatten()
            plot_image(axs[0],
                       image_test,
                       "image")
            plot_image(axs[1],
                       decoder,
                       "decoder image")
            plt.tight_layout(pad=2)
            filename = str(epoch).zfill(5)
            filename = f"Test_{filename}"
            filename = join(self.params["path graphics"],
                            self.params["dataset"]["type"],
                            filename)
            plt.savefig(filename,
                        dpi=400)
            plt.close()
            if epoch % 50 == 0:
                _ = self.checkpoint.save()
                print(f'\nSaving checkpoint  epoch {epoch+1}')
            final_time = time()-start
            print('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                 final_time))
        return history_all
