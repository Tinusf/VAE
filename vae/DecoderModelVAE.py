import tensorflow.keras
import tensorflow.keras.layers as layers
import numpy as np
import config


class VAEDecoderModel(tensorflow.keras.Model):
    def __init__(self, image_shape):
        print("image_shape", image_shape)
        super(VAEDecoderModel, self).__init__()
        flatten_nodes = np.product(image_shape)
        self.latent_inputs = layers.Dense(flatten_nodes, activation="relu")
        self.dense1 = layers.Dense(512, activation="relu")
        self.dense2 = layers.Dense(512, activation="relu")
        self.dense3 = layers.Dense(flatten_nodes, activation="sigmoid")
        self.output_layer = layers.Reshape(image_shape, input_shape=(flatten_nodes,))
        self.batchnorm0 = layers.BatchNormalization()
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.batchnorm3 = layers.BatchNormalization()

    def call(self, x):
        if config.VAE_BATCH_NORM:
            x = self.batchnorm0(x)
        x = self.latent_inputs(x)
        if config.VAE_BATCH_NORM:
            x = self.batchnorm1(x)
        x = self.dense1(x)
        if config.VAE_BATCH_NORM:
            x = self.batchnorm2(x)
        x = self.dense2(x)
        if config.VAE_BATCH_NORM:
            x = self.batchnorm3(x)
        x = self.dense3(x)
        return self.output_layer(x)
