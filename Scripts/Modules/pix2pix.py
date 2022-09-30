from tensorflow_examples.models.pix2pix import pix2pix
from keras.losses import BinaryCrossentropy
from tensorflow.data import Dataset
from keras.optimizers import Adam
from tabulate import tabulate
from keras.losses import Loss
from tensorflow.train import (
    CheckpointManager,
    Checkpoint
)
from tensorflow import (
    GradientTape,
    reduce_mean,
    zeros_like,
    ones_like,
    function,
    abs
)
from time import time
from pandas import (
    DataFrame,
    concat
)
from numpy import (
    array,
    prod
)
from keras import Model
loss_obj = BinaryCrossentropy(from_logits=True)
OUTPUT_CHANNELS = 3
LAMBDA = 10


class pix2pix_model(Model):
    def __init__(self,
                 **kwargs) -> None:
        super(pix2pix_model,
              self).__init__(**kwargs)
        self.generator_g = pix2pix.unet_generator(
            OUTPUT_CHANNELS,
            norm_type='instancenorm'
        )
        self.generator_f = pix2pix.unet_generator(
            OUTPUT_CHANNELS,
            norm_type='instancenorm'
        )
        self.discriminator_x = pix2pix.discriminator(
            norm_type='instancenorm',
            target=False
        )
        self.discriminator_y = pix2pix.discriminator(
            norm_type='instancenorm',
            target=False
        )
        self.generator_g_optimizer = Adam(
            2e-4,
            beta_1=0.5
        )
        self.generator_f_optimizer = Adam(
            2e-4,
            beta_1=0.5
        )
        self.discriminator_x_optimizer = Adam(
            2e-4,
            beta_1=0.5)
        self.discriminator_y_optimizer = Adam(
            2e-4,
            beta_1=0.5
        )
        self._create_checkpoint()

    def _create_checkpoint(self) -> None:
        checkpoint_path = "../Checkpoint"
        ckpt = Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            generator_g_optimizer=self.generator_g_optimizer,
            generator_f_optimizer=self.generator_f_optimizer,
            discriminator_x_optimizer=self.discriminator_x_optimizer,
            discriminator_y_optimizer=self.discriminator_y_optimizer
        )
        self.checkpoint = CheckpointManager(
            ckpt,
            checkpoint_path,
            max_to_keep=5
        )
        # if a checkpoint exists, restore the latest checkpoint.
        if self.checkpoint.latest_checkpoint:
            ckpt.restore(self.checkpoint.latest_checkpoint)
            print('Latest checkpoint restored!!')

    @function
    def train_step(self,
                   real_x: array,
                   real_y: array) -> dict:
        '''
        '''
        with GradientTape(persistent=True) as tape:
            # predict
            fake_y = self.generator_g(
                real_x,
                training=True
            )
            cycled_x = self.generator_f(
                fake_y,
                training=True
            )
            fake_x = self.generator_f(
                real_y,
                training=True
            )
            cycled_y = self.generator_g(
                fake_x,
                training=True
            )
            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(
                real_x,
                training=True
            )
            same_y = self.generator_g(
                real_x,
                training=True
            )
            disc_real_x = self.discriminator_x(
                real_x,
                training=True
            )
            disc_real_y = self.discriminator_y(
                real_y,
                training=True
            )
            disc_fake_x = self.discriminator_x(
                fake_x,
                training=True
            )
            disc_fake_y = self.discriminator_y(
                fake_y,
                training=True
            )
            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)
            cycle_loss_x = calc_cycle_loss()(
                real_x,
                cycled_x
            )
            cycle_loss_y = calc_cycle_loss()(
                real_y,
                cycled_y
            )
            total_cycle_loss = cycle_loss_x+cycle_loss_y

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss
            total_gen_g_loss += cycle_loss_x
            # total_gen_g_loss += vae_loss_x
            # total_gen_g_loss += total_cycle_loss
            total_gen_g_loss += identity_loss()(real_y,
                                                same_y)
            total_gen_f_loss = gen_f_loss
            total_gen_g_loss += cycle_loss_y
            # total_gen_f_loss += vae_loss_y
            # total_gen_f_loss += total_cycle_loss
            total_gen_g_loss += identity_loss()(real_x,
                                                same_x)
            disc_x_loss = discriminator_loss()(disc_real_x,
                                               disc_fake_x)
            disc_y_loss = discriminator_loss()(disc_real_y,
                                               disc_fake_y)
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(
            total_gen_g_loss,
            self.generator_g.trainable_variables
        )
        generator_f_gradients = tape.gradient(
            total_gen_f_loss,
            self.generator_f.trainable_variables
        )
        discriminator_x_gradients = tape.gradient(
            disc_x_loss,
            self.discriminator_x.trainable_variables
        )
        discriminator_y_gradients = tape.gradient(
            disc_y_loss,
            self.discriminator_y.trainable_variables
        )
        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(
            zip(generator_g_gradients,
                self.generator_g.trainable_variables)
        )
        self.generator_f_optimizer.apply_gradients(
            zip(generator_f_gradients,
                self.generator_f.trainable_variables)
        )
        self.discriminator_x_optimizer.apply_gradients(
            zip(discriminator_x_gradients,
                self.discriminator_x.trainable_variables)
        )
        self.discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients,
                self.discriminator_y.trainable_variables)
        )

        loss_history = {
            "loss_g": total_gen_g_loss,
            "loss_f": total_gen_f_loss,
        }
        return loss_history

    def fit(self,
            dataset: Dataset,
            epochs: int) -> DataFrame:
        history_all = DataFrame()
        for epoch in range(1,
                           epochs+1):
            start = time()
            print(f"Epoch {epoch}")
            for i, (image_x, image_y) in dataset.take(1).enumerate():
                i = i.numpy()
                history = self.train_step(image_x,
                                          image_y)
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
            if (epoch + 1) % 100 == 0:
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
