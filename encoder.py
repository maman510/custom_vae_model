
import tensorflow as tf
import keras.layers
import keras
import tool_box
import numpy as np
import time
# from sklearn.decomposition import PCA

# # Assume 'image_data' is a batch of images as numpy array (e.g., shape (batch_size, 28, 28, 1))

# # Step 1: Apply PCA to reduce dimensionality
# pca = PCA(n_components=50)
# image_data_flattened = image_data.reshape(-1, 28*28)  # Flatten the images
# image_data_pca = pca.fit_transform(image_data_flattened)

# # Step 2: Define the VAE architecture
# latent_dim = 2  # Dimension of the latent space (could be higher for more complex models)


class Encoder(keras.Model):

    def __init__(self, latent_dims, filters, image_resize_target=None):
        super(Encoder, self).__init__(name="encoder")
        self.latent_dims = latent_dims
        self.filters = filters
        self.image_resize_target = image_resize_target
       
        self._input_shape = (*self.image_resize_target, 3)

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
            
            self.transformation_layers.append(keras.layers.MaxPooling2D(pool_size=(2,2),
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


        if self.image_resize_target != None:
            inputs = tf.image.resize(inputs, self.image_resize_target, method="lanczos5")
            x = tf.cast(inputs, dtype=tf.float32)
        else:

            x = inputs

        print(tool_box.color_string('cyan', f"\n\nENCODER FILTERS: {self.filters}\n\nENCODER INPUT SHAPE: {x.shape}\n\n"))
      #  time.sleep(3)
        for i in range(1,len(self.transformation_layers)):
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



 # self.pca = PCA(n_components=50)

        # if self.target_image_size != None:
        #     inputs = tf.image.resize(inputs, self.target_image_size, method="lanczos5")
        #     inputs = tf.cast(inputs, dtype=tf.float32)

    #   image_data_flattened = inputs.reshape(-1, self._input_shape[0], self._input_shape[1])  # Flatten the images
    #     image_data_pca = self.pca.fit_transform(image_data_flattened)