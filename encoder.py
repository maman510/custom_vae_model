
import tensorflow as tf
import keras.layers
import keras
import tool_box
import numpy as np


class Encoder(keras.Model):

    def __init__(self, latent_dims, filters, target_image_size=None):
        super(Encoder, self).__init__(name="encoder")
        self.latent_dims = latent_dims
        self.filters = filters
        self.target_image_size = target_image_size


    def build(self, input_shape):
        #combine conv and pool into single array for correct layer stack order (e.g. [conv2d, pool, conv2d, pool...])
        self.transformation_layers = []
        for i in range(len(self.filters)):
            self.transformation_layers.append(keras.layers.Conv2D(self.filters[i], 
                                            kernel_size=(3,3), 
                                            padding="same", 
                                            activation="relu",
                                            name=f"encoder_conv_layer_{i+1}"
                                            )
            )
            self.transformation_layers.append(keras.layers.BatchNormalization())
            
            self.transformation_layers.append(keras.layers.AveragePooling2D(pool_size=(2,2),
                                                                 padding="same",
                                                                 name=f"encoder_pooling_layer_{i+1}"
                                                                 )
            )
            
       
        
        self._input_layer = self.transformation_layers[0]
        # Latent space representation (bottleneck)

        self.flatten_layer = keras.layers.Flatten()
        self.z_mean_layer = keras.layers.Dense(self.latent_dims, name="z_mean")
        self.z_log_var_layer = keras.layers.Dense(self.latent_dims, name="z_log_var")
        self.output_layer = keras.layers.Lambda(self._sampling, name="encoder_output_layer")
        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        if self.target_image_size != None:
            inputs = tf.image.resize(inputs, self.target_image_size, method="lanczos5")
            inputs = tf.cast(inputs, dtype=tf.float32)  
        x = inputs
        for i in range(len(self.transformation_layers)):
            x = self.transformation_layers[i](x)
    
        self.shape_before_bottleneck = x.shape[1:]
        x = self.flatten_layer(x)
        self.z_mean = self.z_mean_layer(x)
        self.z_log_var = self.z_log_var_layer(x)
        self.z = self.output_layer([self.z_mean,self.z_log_var])
        return self.z_log_var, self.z_mean, self.z


    def _sampling(self, args):
        z_log_var, z_mean = args
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon






