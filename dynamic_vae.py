import keras
import keras.applications
import keras.applications.vgg16
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
        
        #=== vgg and ssim
        self.vgg_model = self._build_vgg_model()

        
        
        #define total epochs for training
        self.training_epoch_count = training_epoch_count
        self.current_epoch = 0
        self.current_recon_level = None
        
    def _build_vgg_model(self):
        vgg = keras.applications.VGG16(weights="imagenet", include_top=False)
      #  'block1_conv2', 'block2_conv2' ,'block5_conv2', 'block5_conv3'
        layer_names =    ['block3_conv2', 'block3_conv3', 'block4_conv2','block4_conv3']
        vgg_outputs = [vgg.get_layer(name).output for name in layer_names]
        vgg_model = keras.Model(inputs=vgg.input, outputs=vgg_outputs)
        vgg_model.trainable = True
        return vgg_model
 
    def _vgg_loss(self, original, reconstructed):
        original_preprocessed = keras.applications.vgg16.preprocess_input(original * 255.0)
        reconstructed_preprocessed = keras.applications.vgg16.preprocess_input(reconstructed * 255.0)
        original_features = self.vgg_model(original_preprocessed)
        reconstructed_features = self.vgg_model(reconstructed_preprocessed)
        loss = 0
        for original_feat, recon_feat in zip(original_features, reconstructed_features):
            loss += tf.reduce_mean(tf.square(original_feat -  recon_feat))
        
        return loss

    def _ssim_loss(self, original, reconstructed):
        return 1 - tf.reduce_mean(tf.image.ssim(original, reconstructed, max_val=1.0))
    

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _compute_kl_divergence(self, z_mean, z_log_var):
        kl_div = -0.5 * tf.reduce_sum(1 + z_log_var - 
                        tf.square(z_mean) - 
                        tf.exp(z_log_var), axis=-1)
        
        return tf.reduce_mean(kl_div)
    
    def _compute_reconstruction_error(self, input_images, reconstructed, delta=1.0):
        huber_loss = tf.keras.losses.Huber(delta=delta)
        return huber_loss(input_images, reconstructed)
    

    def combined_loss(self, original, reconstructed, z_mean, z_log_var):
        reconstruction_loss = self._compute_reconstruction_error(original, reconstructed)
        perceptual_loss = self._vgg_loss(original, reconstructed)
        ssim_loss = self._ssim_loss(original, reconstructed)
        kl_loss = self._compute_kl_divergence(z_mean, z_log_var)

        #set weights for reconstruction, vgg, and ssim losses

        alpha = 1.0 #for reconstruction loss
        beta = 0.1 #for vgg_loss
        gamma = 0.1 #for ssim_loss

        total_loss = (alpha * reconstruction_loss) + (beta * perceptual_loss) + (gamma * ssim_loss) + kl_loss
        return kl_loss, reconstruction_loss, perceptual_loss, ssim_loss, total_loss
    
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self._sampling(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        
        return reconstructed, z_mean, z_log_var
    
 
    def train_step(self, original_images):
        self.current_epoch += 1
        with tf.GradientTape() as tape:
            #forward pass:
            reconstructed, z_mean, z_log_var = self(original_images)

            #calculate kl_div and recon error

            kl_divergence, reconstruction_loss, perceptual_loss, ssim_loss, total_loss = self.combined_loss(original_images, reconstructed, z_mean, z_log_var)

            #adjust kl_beta
           # self.kl_beta = self.adjust_kl_beta(reconstruction_error, kl_divergence)
           # total_loss = reconstruction_error + (kl_divergence * self.kl_beta)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

     

        return {"loss": total_loss, "vgg_loss": perceptual_loss , "ssim_loss": ssim_loss, "reconstruction_error": reconstruction_loss, "kl_divergence": kl_divergence}
    

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
    
    def reconstruct_image(self, original_image):
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

latent_dim = 32

encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)

epochs=100

vae = DynamicVAE(encoder, decoder, training_epoch_count=epochs)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, 
    decay_steps=100000, 
    decay_rate=0.96, 
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

vae.compile(optimizer=optimizer)


vae.fit(x_train,  epochs=epochs, batch_size=32)

# Testing with one image
original_image = x_test[0:1]  # Take the first image from the test set

vae.reconstruct_image(original_image)
