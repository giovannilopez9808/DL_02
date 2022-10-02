from tensorflow_examples.models.pix2pix import pix2pix
from keras.losses import BinaryCrossentropy
from tensorflow.data import Dataset
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tabulate import tabulate
from keras.losses import Loss
from tensorflow.train import (
    CheckpointManager,
    Checkpoint
)
from os.path import join
from tensorflow import (
    GradientTape,
    reduce_mean,
    reduce_sum,
    zeros_like,
    ones_like,
    function,
    square,
    exp,
    abs
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


class VAE_pix2pix_model(Model):
    def __init__(self,
                 params: dict,
                 **kwargs) -> None:
        super(VAE_pix2pix_model,
              self).__init__(**kwargs)
        self.params = params
        self.vae_dog = VAE(**params["VAE"])
        self.vae_cat = VAE(**params["VAE"])
        self.generator_cat = pix2pix.unet_generator(
            OUTPUT_CHANNELS,
            norm_type='instancenorm',
        )
        self.generator_dog = pix2pix.unet_generator(
            OUTPUT_CHANNELS,
            norm_type='instancenorm'
        )
        self.discriminator_cat = pix2pix.discriminator(
            norm_type='instancenorm',
            target=False
        )
        self.discriminator_dog = pix2pix.discriminator(
            norm_type='instancenorm',
            target=False
        )
        self.generator_dog_optimizer = Adam(
            2e-4,
            beta_1=0.5
        )
        self.generator_cat_optimizer = Adam(
            2e-4,
            beta_1=0.5
        )
        self.discriminator_dog_optimizer = Adam(
            2e-4,
            beta_1=0.5)
        self.discriminator_cat_optimizer = Adam(
            2e-4,
            beta_1=0.5
        )
        self._create_checkpoint()

    def _create_checkpoint(self) -> None:
        checkpoint_path = "../Checkpoint"
        ckpt = Checkpoint(
            vae_dog=self.vae_dog,
            vae_cat=self.vae_cat,
            generator_cat=self.generator_cat,
            generator_dog=self.generator_dog,
            discriminator_cat=self.discriminator_cat,
            discriminator_dog=self.discriminator_dog,
            generator_cat_optimizer=self.generator_cat_optimizer,
            generator_dog_optimizer=self.generator_dog_optimizer,
            discriminator_cat_optimizer=self.discriminator_cat_optimizer,
            discriminator_dog_optimizer=self.discriminator_dog_optimizer
        )
        self.checkpoint = CheckpointManager(
            ckpt,
            checkpoint_path,
            max_to_keep=1
        )
        # if a checkpoint exists, restore the latest checkpoint.
        if self.checkpoint.latest_checkpoint:
            ckpt.restore(self.checkpoint.latest_checkpoint)
            print('Latest checkpoint restored!!')

    @function
    def train_step(self,
                   dog: array,
                   cat: array) -> dict:
        '''
        '''
        with GradientTape(persistent=True) as tape:
            # predict
            pred_dog = self.vae_dog.encoder_model(dog)
            # predict
            zd, zd_mean, zd_log_var = self.vae_dog.sampler_model(pred_dog)
            pred_dog = self.vae_dog.decoder_model(zd)
            # loss
            rd_loss = self.vae_dog.r_loss_factor * self.vae_dog.mae(
                dog,
                pred_dog
            )
            kld_loss = 1 + zd_log_var - square(zd_mean) - exp(zd_log_var)
            kld_loss = -0.5*kld_loss
            kld_loss = reduce_mean(reduce_sum(kld_loss,
                                              axis=1))
            total_loss_dog = rd_loss + kld_loss

            pred_cat = self.vae_cat.encoder_model(cat)
            # predict
            zc, zc_mean, zc_log_var = self.vae_cat.sampler_model(pred_cat)
            pred_cat = self.vae_cat.decoder_model(zc)
            # loss
            rc_loss = self.vae_cat.r_loss_factor * self.vae_cat.mae(
                cat,
                pred_cat
            )
            klc_loss = 1 + zc_log_var - square(zc_mean) - exp(zc_log_var)
            klc_loss = -0.5*klc_loss
            klc_loss = reduce_mean(reduce_sum(klc_loss,
                                              axis=1))
            total_loss_cat = rc_loss + klc_loss

            fake_cat = self.generator_cat(
                pred_dog,
                training=True
            )
            cycled_dog = self.generator_dog(
                fake_cat,
                training=True
            )
            fake_dog = self.generator_dog(
                pred_cat,
                training=True
            )
            cycled_cat = self.generator_cat(
                fake_dog,
                training=True
            )
            # same_x and same_y are used for identity loss.
            same_dog = self.generator_dog(
                dog,
                training=True
            )
            same_cat = self.generator_cat(
                cat,
                training=True
            )
            disc_real_cat = self.discriminator_cat(
                cat,
                training=True
            )
            disc_real_dog = self.discriminator_dog(
                dog,
                training=True
            )
            disc_fake_cat = self.discriminator_cat(
                fake_cat,
                training=True
            )
            disc_fake_dog = self.discriminator_dog(
                fake_dog,
                training=True
            )
            # calculate the loss
            gen_dog_loss = generator_loss(disc_fake_dog)
            gen_cat_loss = generator_loss(disc_fake_cat)
            cycle_loss_dog = calc_cycle_loss()(
                dog,
                cycled_dog
            )
            cycle_loss_cat = calc_cycle_loss()(
                cat,
                cycled_cat
            )
            cycle_loss = cycle_loss_cat +cycle_loss_dog
            # Total generator loss = adversarial loss + cycle loss
            total_gen_cat_loss = gen_cat_loss
            total_gen_cat_loss += cycle_loss
            total_gen_cat_loss += identity_loss()(cat,
                                                  same_cat)
            total_gen_dog_loss = gen_dog_loss
            total_gen_dog_loss += cycle_loss
            total_gen_dog_loss += identity_loss()(dog,
                                                  same_dog)
            disc_dog_loss = discriminator_loss()(disc_real_dog,
                                                 disc_fake_dog)
            disc_cat_loss = discriminator_loss()(disc_real_cat,
                                                 disc_fake_cat)
        # Calculate the gradients for generator and discriminator
        vae_dog_gradients = tape.gradient(total_loss_dog,
                                          self.vae_dog.trainable_weights)
        vae_cat_gradients = tape.gradient(total_loss_cat,
                                          self.vae_cat.trainable_weights)
        generator_cat_gradients = tape.gradient(
            total_gen_cat_loss,
            self.generator_cat.trainable_variables
        )
        generator_dog_gradients = tape.gradient(
            total_gen_dog_loss,
            self.generator_dog.trainable_variables
        )
        discriminator_cat_gradients = tape.gradient(
            disc_cat_loss,
            self.discriminator_cat.trainable_variables
        )
        discriminator_dog_gradients = tape.gradient(
            disc_dog_loss,
            self.discriminator_dog.trainable_variables
        )
        # Apply the gradients to the optimizer
        self.vae_dog.optimizer.apply_gradients(
            zip(vae_dog_gradients,
                self.vae_dog.trainable_weights))
        self.vae_cat.optimizer.apply_gradients(
            zip(vae_cat_gradients,
                self.vae_cat.trainable_weights))
        self.generator_cat_optimizer.apply_gradients(
            zip(generator_cat_gradients,
                self.generator_cat.trainable_variables)
        )
        self.generator_dog_optimizer.apply_gradients(
            zip(generator_dog_gradients,
                self.generator_dog.trainable_variables)
        )
        self.discriminator_cat_optimizer.apply_gradients(
            zip(discriminator_cat_gradients,
                self.discriminator_cat.trainable_variables)
        )
        self.discriminator_dog_optimizer.apply_gradients(
            zip(discriminator_dog_gradients,
                self.discriminator_dog.trainable_variables)
        )
        self.vae_dog.total_loss_tracker.update_state(total_loss_dog)
        self.vae_dog.reconstruction_loss_tracker.update_state(rd_loss)
        self.vae_dog.kl_loss_tracker.update_state(kld_loss)
        self.vae_cat.total_loss_tracker.update_state(total_loss_cat)
        self.vae_cat.reconstruction_loss_tracker.update_state(rc_loss)
        self.vae_cat.kl_loss_tracker.update_state(klc_loss)
        loss_history = {
            "kl_cat": self.vae_cat.kl_loss_tracker.result(),
            "kl_dog": self.vae_dog.kl_loss_tracker.result(),
            "loss_cat": total_gen_cat_loss,
            "loss_dog": total_gen_dog_loss,
        }
        return loss_history

    def fit(self,
            dataset: Dataset,
            epochs: int) -> DataFrame:
        history_all = DataFrame()
        dog_test, cat_test = list(dataset.test.take(1))[0]
        for epoch in range(1,
                           epochs+1):
            start = time()
            print(f"Epoch {epoch}")
            for i, (dog, cat) in dataset.train.take(1).enumerate():
                i = i.numpy()
                history = self.train_step(dog,
                                          cat)
                values = map(lambda loss: loss.numpy(),
                             history.values())
                history = DataFrame(values,
                                    index=history.keys())
                history = history.T
                history.index = [i]
                history_all = concat([history_all,
                                      history])
                print(tabulate(history,
                               headers=history.columns))
            if (epoch + 1) % 50 == 0:
                decoder_dog = self.vae_dog(dog_test)
                gen_cat = self.generator_cat(decoder_dog)
                decoder_cat = self.vae_cat(cat_test)
                gen_dog = self.generator_dog(decoder_cat)
                fig, axs = plt.subplots(2, 3,
                                        figsize=(15, 10))
                axs = axs.flatten()
                plot_image(axs[0],
                           dog_test,
                           "dog")
                plot_image(axs[1],
                           decoder_dog,
                           "decoder dog")
                plot_image(axs[2],
                           gen_cat,
                           "cat generate")
                plot_image(axs[3],
                           cat_test,
                           "cat")
                plot_image(axs[4],
                           decoder_cat,
                           "decoder cat")
                plot_image(axs[5],
                           gen_dog,
                           "dog generate")
                plt.tight_layout(pad=2)
                filename = str(epoch).zfill(5)
                filename = f"Test_{filename}"
                filename = join(self.params["path graphics"],
                                filename)
                plt.savefig(filename,
                            dpi=400)
                plt.close()
                _ = self.checkpoint.save()
                print(f'\nSaving checkpoint  epoch {epoch+1}')
            final_time = time()-start
            print('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                 final_time))
        return history_all


class discriminator_loss(Loss):
    def __init__(self):
        super().__init__()

    def call(self,
             real,
             generated):
        real_loss = loss_obj(ones_like(real),
                             real)
        generated_loss = loss_obj(zeros_like(generated),
                                  generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5


def generator_loss(generated):
    loss = loss_obj(ones_like(generated),
                    generated)
    return loss


class calc_cycle_loss(Loss):
    def __init__(self):
        super().__init__()

    def call(self,
             real_image,
             cycled_image):
        loss = reduce_mean(abs(real_image - cycled_image))
        return LAMBDA * loss


class identity_loss(Loss):
    def __init__(self):
        super().__init__()

    def call(self,
             real_image,
             same_image):
        loss = reduce_mean(abs(real_image - same_image))
        return LAMBDA * 1.5 * loss
