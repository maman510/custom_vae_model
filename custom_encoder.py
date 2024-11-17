import keras
import keras.layers
import tensorflow as tf
import numpy as np
from custom_loss import VAELoss, KLDivergenceMetric, ReconstructionLossMetric
from keras.datasets import cifar10
from tabulate import tabulate
import tool_box

class Encoder(keras.Model):

    def __init__(self, latent_dims, input_dims):
        super(Encoder, self).__init__()
        self.latent_dims = latent_dims
        self.input_dims = input_dims
        self.filters = self._get_filters(self.input_dims)


    def build(self, input_shape):
        self.downsample_layers = []
        for i in range(len(self.filters)):
            self.downsample_layers.append(keras.layers.Conv2D(self.filters[i], kernel_size=(3,3), activation="relu", padding="same", name=f"encoder_conv_{i+1}"))
            self.downsample_layers.append(keras.layers.MaxPool2D(pool_size=(2,2), padding="same", name=f"encoder_pool_{i+1}"))
        
        self.flatten_layer = keras.layers.Flatten(name=f"encoder_flatten")
        self.z_log_var_layer = keras.layers.Dense(self.latent_dims, activation="relu", name="encoder_log_var_layer")
        self.z_mean_layer = keras.layers.Dense(self.latent_dims, activation="relu", name="encoder_z_mean_layer")
        self.z_layer = keras.layers.Lambda(self._get_random_sampling, name="encoder_output_layer")
        super(Encoder, self).build(input_shape)
    
    def call(self, inputs):
        x = self.downsample_layers[0](inputs)
        for layer in self.downsample_layers[1:]:
            x = layer(x)

        x = self.flatten_layer(x)
        self.z_log_var = self.z_log_var_layer(x)
        self.z_mean = self.z_mean_layer(x)
        self.z = self.z_layer([self.z_mean, self.z_log_var])
        return self.z_log_var, self.z_mean, self.z

    def summary(self):
        
        x = self.downsample_layers[0].compute_output_shape((None, *self.input_dims))
        table = []
        headers = ["Layer Name", "Output Shape", "Model"]
        for layer in self.downsample_layers[1:]:
            row = [f"{layer.name}", str(x), "Encoder"]
            table.append(row)
    
            x = layer.compute_output_shape(x)
        
        flatten_shape = self.flatten_layer.compute_output_shape(x)
        table.append([f"{self.flatten_layer.name}", str(flatten_shape), "Encoder"])
        log_var_shape = self.z_log_var_layer.compute_output_shape(flatten_shape)
        table.append([f"{self.z_log_var_layer.name}", log_var_shape, "Encoder"])
        z_mean_shape = self.z_mean_layer.compute_output_shape(log_var_shape)
        table.append([f"{self.z_mean_layer.name}", z_mean_shape, "Encoder"])

        table.append([f"{self.z_layer.name}", (None, self.latent_dims), "Encoder"])
        
        print("\n\n\t")
        print(tool_box.color_string('yellow', tabulate(table, headers, tablefmt="fancy_grid")))
        print("\n\n")

    def _get_random_sampling(self,args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

   
    def _get_filters(self,input_shape):
        layer_count = 1
        filters = []
        while True:
            if input_shape[0]/(2**len(filters)) == 1:
                filters.append(input_shape[0]*layer_count)
                break
            else:
                filters.append(input_shape[0]*layer_count)
                layer_count *= 2

        return filters
    
