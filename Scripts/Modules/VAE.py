from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2DTranspose,
    Activation,
    LeakyReLU,
    Reshape,
    Dropout,
    Flatten,
    Conv2D,
    Input,
    Dense
)
from numpy import prod
from tensorflow import (
    GradientTape,
    reduce_mean,
    reduce_sum,
    function,
    square,
    shape,
    exp
)
<<<<<<< HEAD
from time import time
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
=======
>>>>>>> parent of af843d34 (Delete useless VAE)


class VAE(Model):
    def __init__(self,
                 r_loss_factor: int,
                 input_dim: int,
                 latent_dim: int,
                 summary: bool = False,
                 **kwargs):
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
        self.compile(Adam(2e-4,
                          beta_1=0.5))
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
        # Sampler
        self.sampler_model = Sampler(latent_dim=self.latent_dim)
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

        self.built = True

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker, ]

    @function
    def train_step(self, data):
        '''
        '''
        with GradientTape(persistent=True) as tape:

            # predict
            x = self.encoder_model(data)
            z, z_mean, z_log_var = self.sampler_model(x)
            pred = self.decoder_model(z)

            # loss
            r_loss = self.r_loss_factor * self.mae(data, pred)
            kl_loss = 1 + z_log_var - square(z_mean) - exp(z_log_var)
            kl_loss = -0.5*kl_loss
            kl_loss = reduce_mean(reduce_sum(kl_loss, axis=1))
            total_loss = r_loss + kl_loss

        # gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        # train step
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # compute progress
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        history = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        return history

    @function
    def generate(self,
                 z_sample):
        '''
        We use the sample of the N(0,I) directly as  
        input of the deterministic generator. 
        '''
        return self.decoder_model(z_sample)

    @function
    def codify(self,
               images):
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
             training=False):
        '''
        '''
        tmp1 = self.encoder_model.use_dropout
        tmp2 = self.decoder_model.use_dropout
        if not training:
            self.encoder_model.use_dropout = False
            self.decoder_model.use_Dropout = False
        x = self.encoder_model(inputs)
        z, _, _ = self.sampler_model(x)
        pred = self.decoder_model(z)
        self.encoder_model.use_dropout = tmp1
        self.decoder_model.use_dropout = tmp2
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
                x = Activation("tanh")(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('tanh')(x)
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
        config = super().get_config()
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
