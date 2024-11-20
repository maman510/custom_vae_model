import keras
import keras.layers
import keras.losses
import keras.optimizers
import tensorflow as tf
import numpy as np
import time
from keras.datasets import cifar10
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)
tf.config.run_functions_eagerly(True)

class DynamicVAE(keras.Model):

    def __init__(self, encoder, decoder, training_epoch_count, kl_beta_initial=1.0):
        super(DynamicVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_beta = kl_beta_initial
        #define total epochs for training
        self.training_epoch_count = training_epoch_count
        self.current_epoch = 0
        self.current_recon_level = None

    
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        
        return reconstructed, z_mean, z_log_var
    
    def sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def compute_kl_divergence(self, z_mean, z_log_var):
        kl_div = -0.5 * tf.reduce_sum(1 + z_log_var - 
                        tf.square(z_mean) - 
                        tf.exp(z_log_var), axis=-1)
        return tf.reduce_mean(kl_div)
    
    def compute_reconstruction_error(self, inputs, reconstructed):
        return tf.reduce_mean(tf.square(inputs - reconstructed))
    

    def train_step(self, data):
        self.current_epoch += 1
        with tf.GradientTape() as tape:
            #forward pass:
            reconstructed, z_mean, z_log_var = self(data)

            #calculate kl_div and recon error
            kl_divergence = self.compute_kl_divergence(z_mean, z_log_var)
            reconstruction_error = self.compute_reconstruction_error(data, reconstructed)

            #adjust kl_beta
            self.kl_beta = self.adjust_kl_beta(reconstruction_error, kl_divergence)
            total_loss = reconstruction_error + (kl_divergence * self.kl_beta)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

     

        return {"loss": total_loss, "kl_divergence": kl_divergence, "reconstruction_error": reconstruction_error, "kl_beta": self.kl_beta}
    

    def _check_update(self, recon_error):
      
        if recon_error < 0.2:
            current_recon_level = "very_low"
        elif recon_error >= 0.2 and recon_error < 0.4:
            current_recon_level = "low"
        elif recon_error >= 0.4 and recon_error < 0.6:
            current_recon_level = "medium_low"
        elif recon_error >= 0.6 and recon_error < 0.8:
            current_recon_level = "medium"
        else:
            current_recon_level = "high"
        return current_recon_level


     
   
      
    def adjust_kl_beta(self, reconstruction_error, kl_divergence):
        #find what stage of training loop is
        recon_level = self._check_update(reconstruction_error)
        if self.current_recon_level != recon_level:
            print(f"\n\nUPDATING RECON LEVEL...\n\n")
            time.sleep(2)
            self.current_recon_level = recon_level

        beta_schedule = {
                    "high": 0.1,          
                    "medium": 0.2,
                    "medium_low": 0.4,
                    "low": 0.6,     
                    "very_low": 1.0              
                }
        
        return beta_schedule[self.current_recon_level]
        




def build_encoder(input_shape, latent_dim):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", strides=2, padding="same")(inputs)
    x = keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
    x = keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
    #start bottleneck
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    z_mean = keras.layers.Dense(latent_dim)(x)
    z_log_var = keras.layers.Dense(latent_dim)(x)

    return keras.Model(inputs, [z_mean, z_log_var], name="encoder")


def build_decoder(latent_dim, original_shape):
    reshape_layer_shape = (original_shape[0]//4, original_shape[1]//4,64)
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(128, activation="relu")(latent_inputs)
    x = keras.layers.Dense(np.prod(reshape_layer_shape), activation="relu")(x)
    x = keras.layers.Reshape(reshape_layer_shape)(x)
    #begin upsampling
    x = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
    x = keras.layers.Conv2DTranspose(32, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
    outputs = keras.layers.Conv2DTranspose(original_shape[2], kernel_size=(3,3), activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

input_shape = (32,32,3)
latent_dim = 2

encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)

epochs=10

vae = DynamicVAE(encoder, decoder, training_epoch_count=epochs)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, 
    decay_steps=100000, 
    decay_rate=0.96, 
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

vae.compile(optimizer=optimizer)


vae.fit(x_train,  epochs=epochs, batch_size=64)

        
# kl_beta_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.1 * (epoch + 1))

# Testing with one image
original_image = x_test[0:1]  # Take the first image from the test set


# Reconstruct the image using the VAE
reconstructed_image = vae.predict(original_image)

# The model returns a tuple (outputs, loss) from the VAE. We need to get the output (reconstructed image).
reconstructed_image = reconstructed_image[0]  # This extracts the first element of the tuple, which is the reconstructed image.

# Display the original and reconstructed image side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Original Image
axes[0].imshow(original_image.squeeze())
axes[0].set_title("Original Image")
axes[0].axis("off")

# Reconstructed Image
axes[1].imshow(reconstructed_image.squeeze())
axes[1].set_title("Reconstructed Image")
axes[1].axis("off")

plt.show()