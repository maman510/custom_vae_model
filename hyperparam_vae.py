import keras.callbacks
import keras.optimizers
from custom_encoder import Encoder
from custom_decoder import Decoder
import tensorflow as tf
import keras
import tool_box
from tabulate import tabulate
from custom_loss import VAELoss, KLDivergenceMetric, ReconstructionLossMetric
from keras.datasets import cifar10
import keras_tuner as kt

class VAE(keras.Model):

    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.latent_dims = latent_dims
        # self.input_dims = input_dims

    def build(self, input_shape, hp):
        self.encoder = Encoder(latent_dims=self.latent_dims, input_dims=input_shape[1:])
        self.decoder = Decoder(latent_dims=self.latent_dims, reconstruction_shape=input_shape[1:])
        #build models
        self.encoder.build(input_shape, hp)
        self.decoder.build((None, self.latent_dims), hp)
        self.input_dims = input_shape[1:]
        
        super(VAE, self).build(input_shape)

    def call(self, inputs):
     
        self.z_log_var, self.z_mean, self.z = self.encoder(inputs)
        self.reconstructed = self.decoder(self.z)
        return self.reconstructed
    
    def summary(self):
        table = []
        headers = ["Layer Name", "Output Shape", "Model"]
        encoder_input = (None, *self.input_dims)
        encoder_output = self.encoder.layers[0].compute_output_shape(encoder_input)

        #leave out last layer of encoder as it is lambda layer and dynamically determines output shape when called - use (None, latent_dims) as sub
        for layer in self.encoder.layers[1:-1]:
            
            table.append([tool_box.color_string('yellow', f"{layer.name}"), tool_box.color_string('yellow', str(encoder_output)), tool_box.color_string('yellow', "Encoder")])
            encoder_output = layer.compute_output_shape(encoder_output)

        encoder_lambda_output = (None, self.latent_dims)
        table.append([tool_box.color_string('yellow', f"{self.encoder.layers[-1].name}"), tool_box.color_string('yellow', str(encoder_lambda_output)), tool_box.color_string('yellow', "Encoder")])


        #add decoder layer outputs
        decoder_input = encoder_lambda_output
        decoder_output = self.decoder.layers[0].compute_output_shape(decoder_input)
        
        for layer in self.decoder.layers[1:]:
            table.append([tool_box.color_string('green', f"{layer.name}"), tool_box.color_string('green', str(decoder_output)), tool_box.color_string('green', "Decoder")])
            decoder_output = layer.compute_output_shape(decoder_output)

        table.append([tool_box.color_string('green', f"{self.decoder.layers[-1].name}"), tool_box.color_string('green', str(decoder_output)), tool_box.color_string('green', "Decoder")])
        
        self.summary_table = tabulate(table, headers, tablefmt="fancy_grid")
        print(f"\n\n{self.summary_table}\n\n")


    def run_training(self, x_train, x_test, optimizer,  epochs, kl_beta=1, learning_rate=None, batch_size=1, reshape_dims=None, callbacks=None):
        #normalize train and test data
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        if reshape_dims != None:
            print(tool_box.color_string("yellow", f"\nRESHAPING INPUT DATA FROM {x_train.shape[1:]}\tTO\t{(*reshape_dims,3)}....\n"))
            x_train = tf.image.resize(x_train, reshape_dims, method="lanczos5")
            x_train = tf.cast(x_train, dtype=tf.float32)
            x_test = tf.image.resize(x_test, reshape_dims, method="lanczos5")
            x_test = tf.cast(x_test, dtype=tf.float32)
            
        #set input_shape and build model
        input_shape = x_train.shape[1:]
        self.build((None, *input_shape))

        if optimizer == "adam":
            if learning_rate == None:
                optimizer = keras.optimizers.Adam()
            else:
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        #configure custom loss and compile
        vae_loss = VAELoss(encoder=self.encoder, kl_beta=kl_beta)
        self.compile(optimizer=optimizer, loss=vae_loss, metrics=[ReconstructionLossMetric(), KLDivergenceMetric(beta=kl_beta)])
        
        self.summary()
        return self.fit(x_train, x_train,batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test), callbacks=callbacks)



def build_vae(hp):
    latent_dims = 256
    vae = VAE(latent_dims=latent_dims)
    vae.build((None, 32,32,3), hp)
    kl_beta = 0.1
    vae_loss = VAELoss(encoder=vae.encoder, kl_beta=kl_beta)
    vae.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss, metrics=[ReconstructionLossMetric(), KLDivergenceMetric(beta=kl_beta)])
    return vae


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0



# Set up Keras Tuner
tuner = kt.GridSearch(
    build_vae,  # Function to build the VAE model
    objective='val_loss',  # Minimize validation loss
    max_trials=100,  # Maximum epochs per trial
    # factor=3,  # Reduce the search space by a factor of 3 after each round
    directory='hp_results',  # Directory to store results
    project_name='vae_tuning'
)

# Perform the hyperparameter search
tuner.search(x_train, x_train, epochs=50, validation_data=(x_test, x_test))


# batch_size = 512
# reshape_dims = (64,64)
# kl_beta = 0.1
# epochs = 500

cb = [keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001, patience=5, mode="min")]
# vae.run_training(x_train=x_train, 
#                  x_test=x_test,
#                  optimizer="adam", 
#                  epochs=epochs, 
#                  kl_beta=kl_beta, 
#                  learning_rate=0.001, 
#                  batch_size=56,
#                  callbacks=cb
#                 )
