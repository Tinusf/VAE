import tensorflow.keras
import tensorflow.keras.layers as layers
import config
from tensorflow.keras import backend as K


class VAEEncoderModel(tensorflow.keras.Model):
    def __init__(self, image_shape):
        print("image_shape", image_shape)
        super(VAEEncoderModel, self).__init__()
        self.conv1 = layers.Conv2D(
            filters=28, kernel_size=(5, 5), activation="relu", input_shape=image_shape,
            padding="same",
            strides=(2, 2)
        )
        self.conv2 = layers.Conv2D(
            filters=56, kernel_size=(5, 5), activation="relu",
            padding="same",
            strides=(2, 2)
        )
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation='relu')
        self.mu_z = layers.Dense(config.VAE_LATENT_SIZE)
        # Use log(var(z)) instead of var(z) because then we make sure that var(z) is positive.
        self.log_var_z = layers.Dense(config.VAE_LATENT_SIZE)
        # Randomly sample in order to get the approximation p(z|x)
        self.output_layer = layers.Lambda(self.sample, output_shape=(config.VAE_LATENT_SIZE,))
        self.batchnorm0 = layers.BatchNormalization()
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.batchnorm3 = layers.BatchNormalization()
        self.batchnorm4 = layers.BatchNormalization()
        self.batchnorm6 = layers.BatchNormalization()
        self.batchnorm5 = layers.BatchNormalization()

    def sample(self, mu_and_log_var):
        # Sample using Reparameterization trick
        # z = mean_value + standard deviation * epsilon
        # mu is the expected value (forventningsverdi) and log_var_z is the log of the var of z.
        # (varians)
        mu_z, log_var_z = mu_and_log_var
        batch_size = K.shape(mu_z)[0]
        dimension = K.int_shape(mu_z)[1]
        random_eps = K.random_normal(shape=(batch_size, dimension))
        # e^log(x) = x
        # And because of that we can get rid of the log. And we know for sure that var(z) is
        # positive.
        var_z = K.exp(log_var_z)
        # return K.random_normal(mean=mu_z, stddev=K.sqrt(var_z), shape=(batch, dimension))
        return mu_z + K.sqrt(var_z) * random_eps

    def call(self, x):
        if config.VAE_BATCH_NORM:
            x = self.batchnorm0(x)
        x = self.conv1(x)
        if config.VAE_BATCH_NORM:
            x = self.batchnorm1(x)
        x = self.conv2(x)
        if config.VAE_BATCH_NORM:
            x = self.batchnorm2(x)
        x = self.flatten(x)
        x = self.dense(x)
        if config.VAE_BATCH_NORM:
            x = self.batchnorm6(x)
        mu_z = self.mu_z(x)
        log_var_z = self.log_var_z(mu_z)
        if config.VAE_BATCH_NORM:
            mu_z = self.batchnorm3(mu_z)
        if config.VAE_BATCH_NORM:
            log_var_z = self.batchnorm4(log_var_z)
        # Set the expected value MU, and log(var) so it can be used to calculate loss later.
        self.prev_mu_z = mu_z
        self.prev_log_var_z = log_var_z
        return self.output_layer([mu_z, log_var_z])
