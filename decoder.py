import tensorflow as tf
import keras.layers
import keras
import tool_box
import numpy as np
import time

class Decoder(keras.Model):

    def __init__(self, latent_dims, filters, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.latent_dims = latent_dims
        self.filters = filters
 
    
        self.reshape_layer_shape = (self.filters[-1] // (2 ** len(self.filters)),  # height
                                    self.filters[-1] // (2 ** len(self.filters)),  # width
                                    self.filters[0] // 2)  # depth

        # Decoder layers
        self.dense_layer = keras.layers.Dense(np.prod(self.reshape_layer_shape), activation='relu')
        self.reshape_layer = keras.layers.Reshape(self.reshape_layer_shape)

        self.upsampling_layers = []
        for i in range(len(self.filters)):
            self.upsampling_layers.append(keras.layers.UpSampling2D(size=(2, 2), name=f"decoder_upsampling_{i}"))
            self.upsampling_layers.append(keras.layers.Conv2D(self.filters[i], kernel_size=(3, 3), activation='relu', padding='same'))

        # Output layer to reconstruct the image
        self.output_layer = keras.layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same', name="decoder_output")


    def call(self, inputs):
        x = inputs
        print(tool_box.color_string('blue', f"\n\nDECODER FILTERS: {self.filters}\n\nDECODER INPUT SHAPE: {x.shape}\n\n"))
       # time.sleep(3)
        x = self.dense_layer(x)
        x = self.reshape_layer(x)
        for i in range(len(self.upsampling_layers)):
            x = self.upsampling_layers[i](x)

        return self.output_layer(x)
    

    
