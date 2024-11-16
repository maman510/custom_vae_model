import keras.callbacks
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
        if image_resize_target != None:
            input_shape = (*image_resize_target, 3)

        self.encoder_filters = self._get_encoder_filters(input_shape)
        self.decoder_filters = self.encoder_filters[::-1]
        self.latent_dims = latent_dims
        self.image_resize_target = image_resize_target

        self.encoder = Encoder(self.latent_dims, self.encoder_filters, image_resize_target=self.image_resize_target)
        self.decoder = Decoder(latent_dims=self.latent_dims, filters=self.decoder_filters)
        self._input_shape = input_shape
  
    def call(self, inputs):
        self.z_log_var, self.z_mean, self.z = self.encoder(inputs)     
        self.reconstructed_image = self.decoder(self.z)
        return self.reconstructed_image
    
    #override compile to access instance components during training (e.g. self.optimizer in fit during training)
    def compile(self,optimizer, loss, metrics, **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.metics = metrics

        return super(VAE, self).compile(optimizer, loss, metrics, **kwargs)
    
    #override fit to reshape validation data
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_data=None, shuffle=True, initial_epoch=0, steps_per_epoch=None, validation_steps=None, **kwargs):
            
            # You can access validation data here before starting the training
            if validation_data:
                val_x, val_y = validation_data
                if self.image_resize_target != None:
                    val_x = tf.image.resize(val_x, self.image_resize_target)
                    val_x = val_x / 255.0
                print(tool_box.color_string('yellow', f"\n\nValidation data shape: {val_x.shape}\n\n"))
             
            # Call the base fit method
            return super(VAE, self).fit(
                x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                callbacks=callbacks, validation_data=validation_data, shuffle=shuffle, 
                initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps, **kwargs
            )

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



# # Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

scale_factor = 2
latent_dims = 128
TARGET_SIZE =   (64, 64)
vae = VAE(input_shape=x_train.shape[1:], latent_dims=latent_dims, image_resize_target=TARGET_SIZE)
vae.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss=VAELoss(encoder=vae.encoder),metrics=[ReconstructionLossMetric(), KLDivergenceMetric()])
vae.fit(x=x_train, y=x_train, batch_size=128, epochs=500, validation_data=(x_test, x_test))





#===================== PCA EXAMPLE =============================================
'''
from sklearn.decomposition import PCA

# Assume 'image_data' is a batch of images as numpy array (e.g., shape (batch_size, 28, 28, 1))

# Step 1: Apply PCA to reduce dimensionality
pca = PCA(n_components=50)
image_data_flattened = image_data.reshape(-1, 28*28)  # Flatten the images
image_data_pca = pca.fit_transform(image_data_flattened)

# Step 2: Define the VAE architecture
latent_dim = 2  # Dimension of the latent space (could be higher for more complex models)



'''



'''
- base

BASELINE KL (dims = 256, beta = 1, batch = 32)
==============================================

1563/1563 [==============================] - 18s 11ms/step - loss: 0.6455 - reconstruction_loss: 955.8806 - kl_divergence: 351.4878 - val_loss: 0.7259 - val_reconstruction_loss: 220.5565 - val_kl_divergence: 63.3376
Epoch 2/500
1563/1563 [==============================] - 14s 9ms/step - loss: 1.3140 - reconstruction_loss: 1009.2990 - kl_divergence: 341.9273 - val_loss: 3.0440 - val_reconstruction_loss: 203.9532 - val_kl_divergence: 65.7909
Epoch 3/500
1563/1563 [==============================] - 16s 10ms/step - loss: nan - reconstruction_loss: nan - kl_divergence: nan - val_loss: nan - val_reconstruction_loss: nan - val_kl_divergence: nan

BASELINE KL (dims = 256, beta = 4, batch = 32)
================================================

'''
