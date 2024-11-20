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
import random
import numpy as np
import math
# Set random seeds for reproducibility
random.seed(42)  # For Python's built-in random module
np.random.seed(42)  # For NumPy
tf.random.set_seed(42)  # For TensorFlow


class VAE(keras.Model):

    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.latent_dims = latent_dims
        # self.input_dims = input_dims

    def build(self, input_shape):
     
        self.encoder = Encoder(latent_dims=self.latent_dims, input_dims=input_shape[1:])
        self.decoder = Decoder(latent_dims=self.latent_dims, reconstruction_shape=input_shape[1:])
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




    # def _update_vae_loss_weights(self, epoch, logs):
    #     #update weights and current_epoch
    #     if epoch > 0:

           
    #         self.vae_loss.current_epoch = epoch
    #         current_recon_loss = logs["reconstruction_loss"]
    #         current_kl_loss = logs["kl_divergence"]

    #         target_kl = .10 * current_recon_loss
    #         updated_kl_beta = math.log(target_kl, current_kl_loss)



    #         self.kl_beta = updated_kl_beta
       
    #         self.vae_loss.kl_beta = updated_kl_beta

    #         z_log_var = self.vae_loss.z_log_var
    #         z_mean = self.vae_loss.z_mean
    #         self.kl_metric.update_state(z_mean, z_log_var, updated_kl_beta)
    #         display = f"\nEPOCH #{epoch}\tkl_divergence: {logs['kl_divergence'] ** updated_kl_beta}\treconstruction_loss: {logs['reconstruction_loss']} KL_BETA: {updated_kl_beta}\n\n"
    #         print(tool_box.color_string('green', display))
        #    print(f"TARGET KL FOR EARLY TRAINING: {target_kl}")

   
          #  print(f"\n\nUPDATED KL_BETA: {updated_kl_beta}")
          #  print(f"UPDATED KL LOSS: {current_kl_loss**updated_kl_beta}\n\n")

           # self.vae_loss.call()



    def run_training(self, x_train, x_test, optimizer,  epochs, learning_rate=None, batch_size=1, reshape_dims=None, callbacks=None):
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


            #configure loss and metrics

              
            self.kl_beta = 1.0
            self.reconstruction_weight = 1.0
            self.vae_loss = VAELoss(encoder=self.encoder, kl_beta=self.kl_beta, reconstruction_weight=self.reconstruction_weight)
            self.reconstruction_metric = ReconstructionLossMetric()
            self.kl_metric = KLDivergenceMetric(vae_loss=self.vae_loss)
            self.compile(optimizer=optimizer, loss=self.vae_loss, metrics=[self.reconstruction_metric, self.kl_metric])
            
            self.summary()

       

           # cb = [keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self._update_vae_loss_weights(epoch, logs))]
            return self.fit(x_train, x_train,batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test))



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:100]
x_test = x_test[:100]
latent_dims = 2
vae = VAE(latent_dims=latent_dims)
batch_size = 1
reshape_dims = (64,64)

epochs = 500

x_train = x_train
x_test = x_test
learning_rate=0.0001
vae.run_training(x_train=x_train, 
                x_test=x_test,
                optimizer="adam", 
                epochs=epochs,
                learning_rate=learning_rate, 
                batch_size=batch_size,
                # callbacks=cb,
                #  reshape_dims=reshape_dims
)


'''
TO-DO:

1) update weights on val_kl_loss/val_recon_loss in _check_loss training callback; currently terminates once val_loss stabilizes for self.target_epochs epochs

'''

