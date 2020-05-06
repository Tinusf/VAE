from stacked_mnist import DataMode

#
# VAE (Variational Autoencoder Config
#

# The number of nodes in the latent.
VAE_LATENT_SIZE = 48
# If the VAE should be loaded from file or retrained.
LOAD_VAE = False
# Which dataset to use for the VAE for the generative and reconstruction purposes.
VAE_GEN_DATAMODE = DataMode.COLOR_BINARY_COMPLETE
# Which dataset to use for the VAE for the anomaly detector.
VAE_ANOM_DATAMODE = DataMode.COLOR_BINARY_MISSING
# Batch size for the VAE
VAE_BATCH_SIZE = 128
# How many epochs should the VAE learn.
VAE_EPOCHS = 20
# Should the VAE use batch normalization between each layer
VAE_BATCH_NORM = True

#
# Util
#
# If the verification net should be loaded from file or retrained.
LOAD_VERIFICATION_NET_MODEL = True
