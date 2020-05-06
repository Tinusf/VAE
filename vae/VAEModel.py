import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from vae.EncoderModelVAE import VAEEncoderModel
from vae.DecoderModelVAE import VAEDecoderModel


class VAEModel(keras.Model):
    def __init__(self, image_shape):
        super(VAEModel, self).__init__()
        flatten_size = np.product(image_shape)

        # Uses the encoder and decoder model.
        self.encoder_model = VAEEncoderModel(image_shape)
        self.decoder_model = VAEDecoderModel(image_shape)

        self.flatten_size = flatten_size
        optimizer = keras.optimizers.Adam(lr=0.001)
        # Use a custom loss function here.
        self.compile(optimizer=optimizer, loss=self.loss_function)

    def call(self, inputs):
        # First call the encoder and then put that output in the decoder.
        x = self.encoder_model.call(inputs)
        return self.decoder_model.call(x)

    def loss_function(self, y_true, y_pred):
        # Custom loss function which uses both reconstruction loss and the KL divergence loss.
        reconstruction_loss = (binary_crossentropy(y_true, y_pred) * self.flatten_size)
        mu_z = self.encoder_model.prev_mu_z
        log_var_z = self.encoder_model.prev_log_var_z

        # Math is from https://arxiv.org/pdf/1312.6114.pdf Appendix B
        # Exp is the opposite of log therefore it is needed here.
        # And then i take the mean of that.
        kl_divergence = -0.5 * K.sum(1 + log_var_z - K.square(mu_z) - K.exp(log_var_z), axis=-1)

        loss = K.mean(reconstruction_loss) + K.mean(kl_divergence)
        return loss
