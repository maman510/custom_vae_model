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


    def _check_loss(self, epoch, logs):

        #read epoch loss from Metric objects:
        kl_loss = logs["kl_divergence"]
        recon_loss = logs["reconstruction_loss"]
        val_kl_loss = self.current_kl_loss.result() 
        val_recon_loss = self.current_reconstruction_loss.result()
        
        kl_beta = self.initial_kl_beta + (self.kl_beta_max - self.initial_kl_beta) * (epoch / self.target_epochs)
        reconstruction_weight = self.initial_reconstruction_weight + (self.reconstruction_weight_max - self.initial_reconstruction_weight) * (epoch / self.target_epochs)
        print(tool_box.color_string("red", f"\n\n\tkl_loss{kl_loss}\treconstruction_loss: {recon_loss}\ninitial_kl_beta{self.initial_kl_beta}\nreconstruction_weight: {self.reconstruction_weight}"))
        
        print(logs)
        # ratio = kl_loss / recon_loss

        # # Decide whether to adjust the weights based on observed losses
        # if kl_loss > 2 * recon_loss:
        #     # If KL loss is much larger than reconstruction loss, reduce kl_beta
        #     print(tool_box.color_string('red',f"\n\nFinal Epoch KL_LOSS (no weight): {kl_loss} Recon Loss (no weight): {recon_loss}; KL_loss weighted: {kl_loss * self.kl_beta}\tRecon Loss weighted: {recon_loss * self.reconstruction_weight}"))
            
        #     print(tool_box.color_string('green', f"\nKL Loss much higher than recons loss\tReducing Current KL Beta:\t{self.kl_beta}"))
            
        #     self.kl_beta *= 0.1  # Decrease the influence of KL loss
        
        #     print(tool_box.color_string('green', f"\nNew KL Beta:\t{self.kl_beta}\n"))

           
        # elif recon_loss > 2 * kl_loss:
        #     # If reconstruction loss is much larger, increase reconstruction weight

        #     self.reconstruction_weight += 1.1  # Increase the influence of reconstruction loss
            
        #     print(tool_box.color_string('red',f"\n\nFinal Epoch KL_LOSS (no weight): {kl_loss} Recon Loss (no weight): {recon_loss}; KL_loss weighted: {kl_loss * self.kl_beta}\tRecon Loss weighted: {recon_loss * self.reconstruction_weight}"))
        #     print(tool_box.color_string('green', f"\nRecons loss much highter than kl loss\tIncreasing Reconstruction Weight:\t{self.reconstruction_weight}"))
        #     print(tool_box.color_string('green', f"\nNew Reconstruction Weight:\t{self.reconstruction_weight}"))
        # else:

            
        #     if self.current_epoch_count == self.target_epochs:
        #         #reset current_epoch_count or save 1) self.warm_up_beta 2) self.warm_up_reconstruction_weight 3) save model configs and look up adjustments to reconcile transformations to epoch logs)
        #         self.current_epoch_count = self.current_epoch_count + 1
        #         self.warmup_kl_beta = self.kl_beta
        #         self.warm_up_reconstruction_weight = self.reconstruction_weight

        #         warm_up_data = {"kl_beta": self.warm_up_kl_beta, "reconstruction_weight": self.warm_up_reconstruction_weight}
        #         save_path = f"warm_up_weights.pkl"
        #         tool_box.Create_Pkl(save_path, warm_up_data)
                
        #         print(tool_box.color_string('green', f"\n\n\nTARGET EPOCHS SATISFIED; SAVING MODEL, BETA, AND RECONSTRUCTION WEIGHTS TO: {save_path} - TERNMINATING TRAINING\n\n"))

        #         self.stop_training = True

        #         #reset if contining
        #        # self.current_epoch_count = 0
        
        #     else:
        #         self.current_epoch_count = self.current_epoch_count + 1
        #         print(tool_box.color_string('red',f"\n\nFinal Epoch (target epoch #{self.current_epoch_count} out of {self.target_epochs} \nKL_LOSS (no weight): {kl_loss} Recon Loss (no weight): {recon_loss}; KL_loss weighted: {kl_loss * self.kl_beta}\tRecon Loss weighted: {recon_loss * self.reconstruction_weight}"))




    def run_training(self, x_train, x_test, optimizer,  epochs, target_epochs, learning_rate=None, batch_size=1, reshape_dims=None, callbacks=None):
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

            #initiate loss params
         
            self.kl_beta = 1.0
            self.reconstruction_weight = 1.0
            self.vae_loss = VAELoss(encoder=self.encoder, kl_beta=self.kl_beta, reconstruction_weight=self.reconstruction_weight)
                       # self.current_kl_val_loss = None


            self.current_reconstruction_loss = ReconstructionLossMetric()
            self.current_kl_loss = KLDivergenceMetric(beta=self.kl_beta)
            self.compile(optimizer=optimizer, loss=self.vae_loss, metrics=[self.current_reconstruction_loss, self.current_kl_loss])
            
            self.summary()

            cb = [
                keras.callbacks.LambdaCallback(on_epoch_end=lambda epochs, logs: self._check_loss(epochs, logs)),
                ]



            #===   set warm_up parameters here ===
      
            self.initial_kl_beta = 0.1
            self.kl_beta_max = 0.1
            self.initial_reconstruction_weight = 1.0
            self.reconstruction_weight_maxself = 1.0
            self.reconstruction_weight_max = 1.0

            self.target_epochs = target_epochs
            self.current_epoch_count = 0
            self.warm_up_kl_beta = None
            self.warm_up_reconstruction_loss = None


            return self.fit(x_train, x_train,batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test), callbacks=cb, verbose=0)



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

latent_dims = 256
vae = VAE(latent_dims=latent_dims)
batch_size = 256
reshape_dims = (64,64)

epochs = 5000
target_epochs = 100

vae.run_training(x_train=x_train, 
                x_test=x_test,
                optimizer="adam", 
                epochs=epochs,
                learning_rate=0.0001, 
                batch_size=batch_size,
                target_epochs = target_epochs
                # callbacks=cb,
                #  reshape_dims=reshape_dims
)


'''
Use self.stop_training = True in self._check_loss callback if validation condition met (check for options on good checkpoint)

'''