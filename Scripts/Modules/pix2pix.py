from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.train import (CheckpointManager,
                              Checkpoint)
from keras.layers import (BatchNormalization,
                          Conv2DTranspose,
                          Activation,
                          LeakyReLU,
                          Reshape,
                          Dropout,
                          Flatten,
                          Conv2D,
                          Input,
                          Dense)
from keras.losses import (BinaryCrossentropy,
                          MeanAbsoluteError)
from tensorflow.compat.v1 import Session
from tensorflow import (GradientTape,
                        reduce_mean,
                        zeros_like,
                        reduce_sum,
                        ones_like,
                        function,
                        square,
                        compat,
                        shape,
                        exp,
                        abs)
from keras.backend import random_normal
from keras.optimizers import Adam
from keras.metrics import Mean
from keras.losses import Loss
from numpy import (array,
                   prod)
from keras import Model
loss_obj = BinaryCrossentropy(from_logits=True)
OUTPUT_CHANNELS = 3
LAMBDA = 10


class VAE_pix2pix_model(Model):
    def __init__(self,
                 latent_dim: tuple,
                 input_dim: tuple,
                 r_loss_factor: int = 1,
                 summary: bool = False,
                 **kwargs) -> None:
        super(VAE_pix2pix_model,
              self).__init__(**kwargs)
        self.vae_x = VAE(
            latent_dim=latent_dim,
            input_dim=input_dim,
            r_loss_factor=r_loss_factor,
            summary=summary,
            **kwargs
        )
        self.vae_y = VAE(
            latent_dim=latent_dim,
            input_dim=input_dim,
            r_loss_factor=r_loss_factor,
            summary=summary,
            **kwargs
        )
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
            vae_x=self.vae_x,
            vae_y=self.vae_y,
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
            vae_x = self.vae_x.encoder_model(real_x)
            z, z_mean, z_log_var = self.vae_x.sampler_model(vae_x)
            pred_x = self.vae_x.decoder_model(z)
            # loss
            r_loss_x = self.vae_x.r_loss_factor * self.vae_x.mae(real_x,
                                                                 pred_x)
            kl_loss_x = -0.5 * \
                (1 + z_log_var - square(z_mean) - exp(z_log_var))
            kl_loss_x = reduce_mean(reduce_sum(kl_loss_x,
                                               axis=1))
            vae_loss_x = r_loss_x + kl_loss_x

            vae_y = self.vae_y.encoder_model(real_y)
            z, z_mean, z_log_var = self.vae_y.sampler_model(vae_y)
            pred_y = self.vae_y.decoder_model(z)
            # loss
            r_loss_y = self.vae_y.r_loss_factor * self.vae_y.mae(real_y,
                                                                 pred_y)
            kl_loss_y = -0.5 * \
                (1 + z_log_var - square(z_mean) - exp(z_log_var))
            kl_loss_y = reduce_mean(reduce_sum(kl_loss_y,
                                               axis=1))
            vae_loss_y = r_loss_y + kl_loss_y
            fake_y = self.generator_g(
                pred_x,
                training=True
            )
            cycled_x = self.generator_f(
                fake_y,
                training=True
            )
            fake_x = self.generator_f(
                pred_y,
                training=True
            )
            cycled_y = self.generator_g(
                fake_x,
                training=True
            )
            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(
                pred_x,
                training=True
            )
            same_y = self.generator_g(
                pred_y,
                training=True
            )
            disc_real_x = self.discriminator_x(
                pred_x,
                training=True
            )
            disc_real_y = self.discriminator_y(
                pred_y,
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
                pred_x,
                cycled_x
            )
            cycle_loss_y = calc_cycle_loss()(
                pred_y,
                cycled_y
            )
            total_cycle_loss = cycle_loss_x+cycle_loss_y

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss
            total_gen_g_loss += total_cycle_loss
            total_gen_g_loss += identity_loss()(pred_y,
                                                same_y)
            total_gen_f_loss = gen_f_loss
            total_gen_f_loss += total_cycle_loss
            total_gen_g_loss += identity_loss()(pred_x,
                                                same_x)
            disc_x_loss = discriminator_loss(disc_real_x,
                                             disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y,
                                             disc_fake_y)
        # gradient
        vae_grads_x = tape.gradient(
            vae_loss_x,
            self.vae_x.trainable_weights
        )
        vae_grad_y = tape.gradient(
            vae_loss_y,
            self.vae_y.trainable_weights
        )
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
        # train step
        self.vae_x.optimizer.apply_gradients(
            zip(vae_grads_x,
                self.vae_x.trainable_weights)
        )
        self.vae_y.optimizer.apply_gradients(
            zip(vae_grad_y,
                self.vae_y.trainable_weights)
        )
        # compute progress
        self.vae_x.total_loss_tracker.update_state(vae_loss_x)
        self.vae_x.reconstruction_loss_tracker.update_state(r_loss_x)
        self.vae_x.kl_loss_tracker.update_state(kl_loss_x)

        self.vae_y.total_loss_tracker.update_state(vae_loss_y)
        self.vae_y.reconstruction_loss_tracker.update_state(r_loss_y)
        self.vae_y.kl_loss_tracker.update_state(kl_loss_y)
        loss_history = {
            "loss_vae_x": self.vae_x.total_loss_tracker.result(),
            "re_vae_x_loss": self.vae_x.reconstruction_loss_tracker.result(),
            "kl_vae_x_loss": self.vae_x.kl_loss_tracker.result(),
            "loss_vae_y": self.vae_y.total_loss_tracker.result(),
            "re_vae_y_loss": self.vae_y.reconstruction_loss_tracker.result(),
            "kl_vae_y_loss": self.vae_y.kl_loss_tracker.result(),
            "loss_g": total_gen_g_loss,
            "loss_f": total_gen_f_loss,
        }
        return loss_history


def discriminator_loss(real,
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
        return LAMBDA * 0.5 * loss


class VAE(Model):
    def __init__(self,
                 latent_dim: tuple,
                 input_dim: tuple,
                 r_loss_factor: int = 1,
                 summary: bool = False,
                 **kwargs) -> None:
        super(VAE, self).__init__(**kwargs)
        self.r_loss_factor = r_loss_factor
        # Architecture
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_conv_filters = [64, 64, 64, 64]
        self.encoder_conv_kernel_size = [3, 3, 3, 3]
        self.encoder_conv_strides = [2, 2, 2, 2]
        self.n_layers_encoder = len(self.encoder_conv_filters)
        self.decoder_conv_t_filters = [64, 64, 64, 3]
        self.decoder_conv_t_kernel_size = [3, 3, 3, 3]
        self.decoder_conv_t_strides = [2, 2, 2, 2]
        self.n_layers_decoder = len(self.decoder_conv_t_filters)
        self.use_batch_norm = True
        self.use_dropout = True
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.mae = MeanAbsoluteError()
        self.compile(Adam())
        self.built = True
        # Encoder
        self.encoder_model = Encoder(
            input_dim=self.input_dim,
            output_dim=self.latent_dim,
            encoder_conv_filters=self.encoder_conv_filters,
            encoder_conv_kernel_size=self.encoder_conv_kernel_size,
            encoder_conv_strides=self.encoder_conv_strides,
            use_batch_norm=self.use_batch_norm,
            use_dropout=self.use_dropout
        )
        self.encoder_conv_size = self.encoder_model.last_conv_size
        self.sampler_model = Sampler(latent_dim=self.latent_dim)
        # Decoder
        self.decoder_model = Decoder(
            input_dim=self.latent_dim,
            input_conv_dim=self.encoder_conv_size,
            decoder_conv_t_filters=self.decoder_conv_t_filters,
            decoder_conv_t_kernel_size=self.decoder_conv_t_kernel_size,
            decoder_conv_t_strides=self.decoder_conv_t_strides,
            use_batch_norm=self.use_batch_norm,
            use_dropout=self.use_dropout
        )
        if summary:
            self.encoder_model.summary()
            self.sampler_model.summary()
            self.decoder_model.summary()

    @property
    def metrics(self) -> list:
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker, ]

    @function
    def generate(self, z_sample) -> list:
        '''
        We use the sample of the N(0,I) directly as
        input of the deterministic generator.
        '''
        return self.decoder_model(z_sample)

    @function
    def codify(self, images) -> tuple:
        '''
        For an input image we obtain its particular distribution:
        its mean, its variance (unvertaintly) and a sample z of such
        distribution.
        '''
        x = self.encoder_model.predict(images)
        z, z_mean, z_log_var = self.sampler_model(x)
        return z, z_mean, z_log_var

    # implement the call method
    @function
    def call(self,
             inputs,
             training: bool = False) -> list:
        '''
        '''
        tmp1 = self.encoder_model.use_Dropout
        tmp2 = self.decoder_model.use_Dropout
        if not training:
            self.encoder_model.use_Dropout = False
            self.decoder_model.use_Dropout = False
        x = self.encoder_model(inputs)
        z, _, _ = self.sampler_model(x)
        pred = self.decoder_model(z)
        self.encoder_model.use_Dropout = tmp1
        self.decoder_model.use_Dropout = tmp2
        return pred


class Encoder(Model):
    def __init__(self,
                 input_dim: tuple,
                 output_dim: tuple,
                 encoder_conv_filters: list,
                 encoder_conv_kernel_size: list,
                 encoder_conv_strides: list,
                 use_batch_norm: bool = True,
                 use_dropout: bool = True,
                 **kwargs) -> None:
        '''
        '''
        super(Encoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.n_layers_encoder = len(self.encoder_conv_filters)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.model = self.encoder_model()
        self.built = True

    def get_config(self) -> dict:
        config = super(Encoder, self).get_config()
        config.update({
            "units": self.units
        })
        return config

    def encoder_model(self) -> Model:
        '''
        '''
        encoder_input = Input(shape=self.input_dim,
                              name='encoder')
        x = encoder_input
        for i in range(self.n_layers_encoder):
            x = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                name=f'encoder_conv_{i}')(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)
        self.last_conv_size = x.shape[1:]
        x = Flatten()(x)
        encoder_output = Dense(self.output_dim)(x)
        model = Model(encoder_input,
                      encoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)


class Decoder(Model):
    def __init__(self,
                 input_dim: tuple,
                 input_conv_dim: tuple,
                 decoder_conv_t_filters: list,
                 decoder_conv_t_kernel_size: list,
                 decoder_conv_t_strides: list,
                 use_batch_norm: bool = True,
                 use_dropout: bool = True,
                 **kwargs):
        '''
        '''
        super(Decoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.input_conv_dim = input_conv_dim
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.n_layers_decoder = len(self.decoder_conv_t_filters)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.model = self.decoder_model()
        self.built = True

    def get_config(self) -> dict:
        config = super(Decoder,
                       self).get_config()
        config.update({
            "units": self.units
        })
        return config

    def decoder_model(self) -> Model:
        '''
        '''
        decoder_input = Input(shape=self.input_dim,
                              name='decoder')
        x = Dense(prod(self.input_conv_dim))(decoder_input)
        x = Reshape(self.input_conv_dim)(x)
        for i in range(self.n_layers_decoder):
            x = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same',
                name=f'decoder_conv_t_{i}')(x)
            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)
        decoder_output = x
        model = Model(decoder_input,
                      decoder_output)
        return model

    def call(self, inputs) -> list:
        '''
        '''
        return self.model(inputs)


class Sampler(Model):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self,
                 latent_dim: int,
                 **kwargs) -> None:
        super(Sampler, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.model = self.sampler_model()
        self.built = True

    def get_config(self) -> dict:
        config = super(Sampler, self).get_config()
        config.update({"units": self.units})
        return config

    def sampler_model(self) -> Model:
        '''
        input_dim is a vector in the latent (codified) space
        '''
        input_data = Input(shape=self.latent_dim)
        z_mean = Dense(self.latent_dim, name="z_mean")(input_data)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(input_data)
        self.batch = shape(z_mean)[0]
        self.dim = shape(z_mean)[1]
        epsilon = random_normal(shape=(self.batch,
                                       self.dim))
        z = z_mean + exp(0.5 * z_log_var) * epsilon
        model = Model(input_data,
                      [z,
                       z_mean,
                       z_log_var])
        return model

    def call(self, inputs) -> list:
        '''
        '''
        return self.model(inputs)
