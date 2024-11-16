import keras
import keras.layers
import tensorflow as tf
import numpy as np
from custom_loss import VAELoss, KLDivergenceMetric, ReconstructionLossMetric
from keras.datasets import cifar10
from tabulate import tabulate
import tool_box
from custom_encoder import Encoder
class Decoder(keras.Model):

    def __init__(self, latent_dims, reconstruction_shape):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.reconstruction_shape = reconstruction_shape
        self.filters = self._get_decoder_filters(self.reconstruction_shape)
        self.reshape_layer_shape = (2,2, self.filters[0]//2)
        self.dense_layer_hidden_units = np.prod(self.reshape_layer_shape)
    
    def build(self, input_shape):
        self.dense_layer = keras.layers.Dense(self.dense_layer_hidden_units, activation="relu", name=f"decoder_dense_layer")
        self.reshape_layer = keras.layers.Reshape(self.reshape_layer_shape)
        self.upsample_layers = []
        for i in range(1,len(self.filters)):
          
            self.upsample_layers.append(keras.layers.UpSampling2D(size=(2,2), name=f"decoder_upsample_{i+1}"))
            self.upsample_layers.append(keras.layers.Conv2D(self.filters[i], kernel_size=(3,3), padding="same", activation="relu", name=f"decoder_conv_{i+1}"))
        
        self.reconstruction_layer = keras.layers.Conv2D(3, kernel_size=(3,3), padding="same", activation="sigmoid", name=f"decoder_output_layer")
        super(Decoder, self).build(input_shape)

    def call(self, inputs):
        x = self.dense_layer(inputs)
        x = self.reshape_layer(x)
        for layer in self.upsample_layers:
            x = layer(x)
        self.reconstruction = self.reconstruction_layer(x)
        return self.reconstruction
    
    def summary(self):
        table = []
        headers = ["Layer Name", "Output Shape", "Model"]
        dense_output = self.dense_layer.compute_output_shape((None, self.latent_dims))
        x = self.reshape_layer.compute_output_shape(dense_output)
        table.append([f"{self.dense_layer.name}", dense_output, "Decoder"])
        table.append([f"{self.reshape_layer.name}",x, "Decoder"])
      
        for layer in self.upsample_layers:
            x = layer.compute_output_shape(x)
            table.append([f"{layer.name}", x, "Decoder"])
        
        reconstruction_shape = self.reconstruction_layer.compute_output_shape(x)
        table.append([f"{self.reconstruction_layer.name}", reconstruction_shape, "Decoder"])
        self.summary_table = tool_box.color_string('green', tabulate(table, headers, tablefmt="fancy_grid"))
        print("\n\n\t")
        print(self.summary_table)
        print("\n\n")

    def _get_decoder_filters(self,input_shape):
        layer_count = 1
        filters = []
        while True:
            if input_shape[0]/(2**len(filters)) == 1:
                filters.append(input_shape[0]*layer_count)
                break
            else:
                filters.append(input_shape[0]*layer_count)
                layer_count *= 2

        return filters[::-1][1:]

