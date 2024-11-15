import tensorflow as tf
import keras.layers
import keras
import tool_box
import numpy as np


class Decoder(keras.Model):

    def __init__(self, latent_dims, filters):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims

        self.filters = filters
    

    def build(self, input_shape):
        #use self.filters to find reshape shape dimensions (encoder's shape before bottleneck); note: self.filters are the reverse order of encoder's filters, hence self.filters[-1] = input shape of encoder and self.filters[0] = output shape of filters (shape before bottleneck)
        self.reshape_layer_shape = (self.filters[-1]//(2**len(self.filters))*2, self.filters[-1]//(2**len(self.filters))*2, self.filters[0]//2)
    
        #define reshape and fully connected layers
        self.latent_space_layer = keras.layers.Dense(np.prod(self.reshape_layer_shape), activation="relu", name="decoder_latent_space_layer")
        self.reshape_layer = keras.layers.Reshape(self.reshape_layer_shape, name="decoder_reshape_layer")
        
        #combine upsample and conv2d into single array for correct layer stack order (e.g. [upsample, conv2d, upsample, conv2d])
        self.upsampling_layers = []
        for i in range(1, len(self.filters)):
            self.upsampling_layers.append(keras.layers.UpSampling2D((2, 2), name=f"decoder_upsampling_{i}"))
            self.upsampling_layers.append(keras.layers.Conv2D(self.filters[i], (3, 3), activation='relu', padding='same', name=f"decoder_conv_{i}"))

        #define output layer
        self.output_layer = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        super(Decoder, self).build(input_shape)


    def call(self, inputs):
        x = inputs
        x = self.latent_space_layer(x)
        x = self.reshape_layer(x)
        for i in range(len(self.upsampling_layers)):
            x = self.upsampling_layers[i](x)

        return self.output_layer(x)
    

    
 #8 * 8 * 64

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype("float32")/255.0
# x_test = x_test.astype("float32")/255.0


# filters = [32, 64, 128][::-1]
# latent_dims = filters[0]//2
# model = Decoder(latent_dims=latent_dims, filters=filters)
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# model.build((None, latent_dims))
# model.summary()

# # Define the decoder part
# def build_decoder(latent_dim, output_shape, filters):
#     #configer reshape layer and hidden_units of initial dense layer based on encoder's filters (note: encoder_filter order reveresed prior to being passed here)
#     reshape_layer_shape = (filters[-1]//(2**len(filters)), filters[-1]//(2**len(filters)), filters[0])
#     hidden_units = np.prod(reshape_layer_shape)

#     #add additional final layer
#     filters.append(input_shape[0]//2)

#     latent_inputs = layers.Input(shape=(latent_dim,))
#     # Latent vector is reshaped into a small feature map
#     x = layers.Dense(hidden_units, activation='relu')(latent_inputs) 
#     x = layers.Reshape(reshape_layer_shape)(x)
    
#     for i in range(1, len(filters)):
#         x = layers.UpSampling2D((2, 2), name=f"decoder_upsampling_{i}")(x)  
#         x = layers.Conv2D(filters[i], (3, 3), activation='relu', padding='same', name=f"decoder_conv_{i}")(x)


#     # Output layer to reconstruct the input 
#     output = layers.Conv2D(output_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

#     # Decoder model
#     decoder = models.Model(latent_inputs, output, name="decoder")
  
#     return decoder

