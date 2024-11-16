from keras import layers, models
import tensorflow as tf
import numpy as np
from custom_loss import VAELoss, KLDivergenceMetric, ReconstructionLossMetric
from keras.datasets import cifar10

def get_filters(input_shape):
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



# Define the encoder part
def build_encoder(input_shape, latent_dim, filters):
    inputs = layers.Input(shape=input_shape)

    # transformation_layers = [
    #     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2), padding='same'),
    #     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2), padding='same'),
    #     layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2), padding='same'),
    #     layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2), padding='same'),
    #     layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2), padding='same'),
    #     layers.Conv2D(2048, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2), padding='same')
    # ]

    transformation_layers = []
    for filter in filters:
        transformation_layers.append(layers.Conv2D(filter, (3,3), activation="relu", padding="same"))
        transformation_layers.append(layers.MaxPooling2D((2, 2), padding='same'))
    x = transformation_layers[0](inputs)
    for layer in transformation_layers[1:]:
        x = layer(x)
        print(f"ENCODER SHAPE: {x.shape}")

 

    # Latent space representation (bottleneck)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # Sampling from the latent space (reparameterization trick)
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Encoder model
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder


# Define the decoder part
def build_decoder(latent_dim, output_shape, filters):
    #configer reshape layer and hidden_units of initial dense layer based on encoder's filters (note: encoder_filter order reveresed prior to being passed here)

    filters = filters[1:]
    reshape_layer_shape = (2,2,filters[0])
    hidden_units = np.prod(reshape_layer_shape)
    latent_inputs = layers.Input(shape=(latent_dim,))
    # Latent vector is reshaped into a small feature map
    x = layers.Dense(hidden_units, activation='relu')(latent_inputs) 
    x = layers.Reshape(reshape_layer_shape)(x)
    
    for i in range(1, len(filters)):
        x = layers.UpSampling2D((2, 2), name=f"decoder_upsampling_{i}")(x)  
        x = layers.Conv2D(filters[i], (3, 3), activation='relu', padding='same', name=f"decoder_conv_{i}")(x)


    # Output layer to reconstruct the input 
    output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Decoder model
    decoder = models.Model(latent_inputs, output, name="decoder")
    decoder.summary()
    return decoder



# VAE model that combines encoder and decoder
def build_vae(input_shape, latent_dim):
    # Build encoder and decoder
    encoder_filters = get_filters(input_shape)
   
    decoder_filters = encoder_filters[::-1]
   
    #set latent_dim to output filters of last encoder layer; note: the decoder can have an initial layer set to have equal to or less than the latent dim, as long as "volume" of layer is same as latent space (i.e. the height, width are same as encoder input and latent dim <= output of encoder)
 
    
    
    encoder = build_encoder(input_shape, latent_dim, encoder_filters)
    decoder = build_decoder(latent_dim, input_shape, decoder_filters)

    # Define the VAE architecture
    inputs = layers.Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)

    # Define the VAE model
    vae = models.Model(inputs, reconstructed, name="vae")

    # Define the loss and metrics
    vae_loss = VAELoss(encoder=encoder)

    # Compile the VAE model with the custom loss function and metrics
    vae.compile(optimizer='adam', loss=vae_loss, metrics=[ReconstructionLossMetric(), KLDivergenceMetric()])

    return vae, encoder, decoder


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0



x_train = tf.image.resize(x_train, (64,64), method="lanczos5")
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.image.resize(x_test, (64,64), method="lanczos5")
x_test = tf.cast(x_test, dtype=tf.float32)


input_shape = (64,64,3)
latent_dim = 256

vae, encoder, decoder = build_vae(input_shape=input_shape, latent_dim=latent_dim)
batch_size = 256
vae.fit(x_train, x_train, batch_size=batch_size, epochs=10, validation_data=(x_test, x_test))



