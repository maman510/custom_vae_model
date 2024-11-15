import keras.layers
import keras.losses
import keras.optimizers
import tensorflow as tf
import keras
import tool_box
import numpy as np
import os
import sys
import PIL
from encoder import Encoder
from decoder import Decoder
from custom_loss import ReconstructionLossMetric, KLDivergenceMetric, VAELoss
from keras.datasets import cifar10
import time


class VAE(keras.Model):
    def __init__(self, input_shape, latent_dims, image_resize_target=None):  
        super(VAE, self).__init__()
        self.encoder_filters = self._get_encoder_filters(input_shape)
        self.decoder_filters = self.encoder_filters[::-1]
        self.latent_dims = latent_dims
        self.image_resize_target = image_resize_target

        self.encoder = Encoder(self.latent_dims, self.encoder_filters, target_image_size=self.image_resize_target)
        
        self.decoder = Decoder(latent_dims=self.latent_dims, filters=self.decoder_filters)
        self._input_shape = input_shape
  


 


    def call(self, inputs):
     
     
        self.z_log_var, self.z_mean, self.z = self.encoder(inputs)     
        self.reconstructed_image = self.decoder(self.z)
        return self.reconstructed_image
    

    def _get_encoder_filters(self,input_shape):
        layer_count = 1
        filters = []
        while True:
            if input_shape[0]/(2**len(filters)) == 1:
                break
            else:
                filters.append(input_shape[0]*layer_count)
                layer_count *= 2

        return filters
 
            
def resize_dataset(image, labels):
    image = tf.image.resize(image, TARGET_SIZE, method="lanczos5")
    image = tf.cast(image, dtype=tf.float32)
    image = image/255.0
    return image, image


# # Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

scale_factor = 2
TARGET_SIZE =   (64, 64)


latent_dims = 2
vae = VAE(input_shape=x_train.shape[1:], latent_dims=latent_dims)

# Compile the VAE model with the custom loss function and metrics
vae.compile(optimizer = "adam", loss=VAELoss(vae.encoder),metrics=[ReconstructionLossMetric(), KLDivergenceMetric()])
vae.fit(x_train, x_train, batch_size=2048, epochs=500, validation_data=(x_test, x_test))

# 782/782 [==============================] - 9s 8ms/step - loss: 0.6406 - reconstruction_loss: 500.9775 - kl_divergence: 171.6104 - val_loss: 0.6355 - val_reconstruction_loss: 99.7816 - val_kl_divergence: 35.6470
# Epoch 2/500
# 782/782 [==============================] - 5s 7ms/step - loss: 0.6345 - reconstruction_loss: 496.2100 - kl_divergence: 172.7031 - val_loss: 0.6350 - val_reconstruction_loss: 99.7015 - val_kl_divergence: 35.6941
# Epoch 3/500
# 782/782 [==============================] - 5s 6ms/step - loss: 0.6339 - reconstruction_loss: 495.6848 - kl_divergence: 172.8136 - val_loss: 0.6335 - val_reconstruction_loss: 99.4704 - val_kl_divergence: 34.6676