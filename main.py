import os
from tensorflow.keras import backend as K
import util
import draw
from stacked_mnist import StackedMNISTData
import config
import numpy as np
from tensorflow.keras import losses
from vae.VAEModel import VAEModel
import tensorflow as tf

# This is a wierd fix for my GPU to get it to always work.
gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
    print("Setting memory growth on GPU 0")
    tf.config.experimental.set_memory_growth(gpu[0], True)


def get_vae(datamode, generator):
    # Either load the VAE from file or create a new model and train it.
    x_train, y_train = generator.get_full_data_set(training=True)

    image_shape = x_train.shape[1:]
    x_train = x_train.astype(np.float64)

    vae_model = VAEModel(image_shape)

    path = f"./models/VAE_{datamode.name}/"
    filename = path + "model.tf"
    if not os.path.exists(path):
        os.mkdir(path)

    if config.LOAD_VAE:
        vae_model.load_weights(filename)
    else:
        # Train with the x data both as input to the neural net and as labels.
        vae_model.fit(x_train,
                      x_train,
                      epochs=config.VAE_EPOCHS,
                      batch_size=config.VAE_BATCH_SIZE)
        # Have to save the weights instead of the model because custom loss function and
        # everything is custom.
        vae_model.save_weights(filename)
    return vae_model


def main():
    batch_size = 16
    datamode = config.VAE_GEN_DATAMODE
    generator = StackedMNISTData(mode=datamode, default_batch_size=2048)

    # Get the VAE model.
    vae_model = get_vae(datamode, generator)

    x_test, y_test = generator.get_full_data_set(training=False)
    x_test = x_test.astype(np.float64)

    # Create reconstructions of the testing images.
    reconstructed_images = vae_model.predict(x_test)

    draw.draw_images(np.array(x_test[0:16]), np.array(y_test[0:16]), mult_255=True)
    draw.draw_images(np.array(reconstructed_images[0:16]), np.array(y_test[0:16]))

    verification_model = util.get_verification_model(datamode, generator)

    cov = verification_model.check_class_coverage(data=reconstructed_images, tolerance=.8)
    pred, acc = verification_model.check_predictability(data=reconstructed_images,
                                                        correct_labels=y_test)
    print(f"VAE - Reconstructed images - Coverage: {100 * cov:.2f}%")
    print(f"VAE - Reconstructed images - Predictability: {100 * pred:.2f}%")
    print(f"VAE - Reconstructed images - Accuracy: {100 * acc:.2f}%")
    print("---------------------------------------------")

    # Check quality of generated images
    latents = np.random.randn(batch_size, config.VAE_LATENT_SIZE)
    generated_images = vae_model.decoder_model.predict(latents, batch_size=batch_size)
    draw.draw_images(generated_images, mult_255=False)

    cov = verification_model.check_class_coverage(data=generated_images, tolerance=.8)
    pred, _ = verification_model.check_predictability(data=generated_images)
    print(f"VAE - Generated images - Coverage: {100 * cov:.2f}%")
    print(f"VAE - Generated images - Predictability: {100 * pred:.2f}%")
    print("---------------------------------------------")

    datamode = config.VAE_ANOM_DATAMODE
    generator = StackedMNISTData(mode=datamode, default_batch_size=2048)

    x_test, y_test = generator.get_full_data_set(training=False)
    x_test = x_test.astype(np.float64)

    vae_model = get_vae(datamode, generator)

    loss = []
    # N is how many random encodings should each image check their loss against.
    # 5000 is kind of slow, takes a couple of minutes.
    N = 1000
    print("Calculating loss between original images and randomly created images, this will take "
          "a while.")
    # Create random latent encoding.
    latents = np.random.randn(N, config.VAE_LATENT_SIZE)
    # Generate images for these latents.
    generated = vae_model.decoder_model.predict(latents, batch_size=batch_size)
    # x_test_length = len(x_test)
    for i in range(len(x_test)):
        # print(f"Iteration {i}/{x_test_length}")
        # Clone the real image N times in order to simplify the crossentropy.
        original_clones = np.array([x_test[i] * N])
        # Get the average loss of crossentropy between the random generated images and the
        # current real image.
        cur_loss = float(K.mean(losses.categorical_crossentropy(original_clones, generated)))
        loss.append(cur_loss)

    # Get the top 16 loss indexes.
    top_loss = np.array(loss).argsort()[-16:][::-1]

    top_16 = []
    top_16_labels = []
    for i in top_loss:
        top_16.append(x_test[i])
        top_16_labels.append(str(y_test[i]))

    # Conclusion: With a VAE you can generate new images easily.
    # Also anomalies are quite good aswell, some thick images and some 8's.

    draw.draw_images(np.array(top_16), labels=top_16_labels, mult_255=False)


if __name__ == '__main__':
    main()
