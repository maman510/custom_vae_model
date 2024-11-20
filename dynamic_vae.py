import keras
import keras.layers
import keras.losses
import keras.optimizers
import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)

class DynamicVAE(keras.Model):

    def __init__(self, encoder, decoder, kl_beta_initial=1.0):
        super(DynamicVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_beta = kl_beta_initial
    
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
        with tf.GradientTape() as tape:
            #forward pass:
            reconstructed, z_mean, z_log_var = self(data)

            #calculate kl_div and recon error
            kl_divergence = self.compute_kl_divergence(z_mean, z_log_var)
            reconstruction_error = self.compute_reconstruction_error(data, reconstructed)

            #adjust kl_beta
            self.kl_beta = self.adjust_kl_beta(kl_divergence, reconstruction_error)
            total_loss = reconstruction_error + kl_divergence * self.kl_beta

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": total_loss, "kl_divergence": kl_divergence, "reconstruction_error": reconstruction_error, "kl_beta": self.kl_beta}
    
    def adjust_kl_beta(self, reconstruction_error, kl_divergence):
        
        if reconstruction_error > 0.05:
            updated_kl = tf.maximum(self.kl_beta * 0.9, 0.1)
            print(f"\n\nUPDATING KL_BETA: current: {self.kl_beta} to {updated_kl}\n\n")
            return updated_kl
        
        elif kl_divergence < 0.1:
            #decay kl_beta
            updated_kl = tf.minimum(self.kl_beta * 1.05, 10.0)
            print(f"\n\nUPDATING KL_BETA: current: {self.kl_beta} to {updated_kl}\n\n")
            return updated_kl
        else:
            
            return self.kl_beta
        


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



input_shape = (64,64,3)
latent_dim = 16

encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)


vae = DynamicVAE(encoder, decoder)

optimizer = keras.optimizers.Adam()
vae.compile(optimizer=optimizer)

x_train = tf.random.normal((100, 64,64,3))

vae.fit(x_train, epochs=10, batch_size=16)

        
    
