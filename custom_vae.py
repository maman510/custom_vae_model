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

    def _check_loss(self, epochs, logs):
        'callback mimics EarlyStop by checking min_delta and patience against current and past epoch data - avoids restarting while allowing update to weights'
        if self.current_val_loss == None:
            self.current_val_loss = logs["loss"]
            display = f"\n\nIN CHECK LOSSES - INITIAL LOSS\n\n"
            print(tool_box.color_string('cyan', display))
            return self._update_losses(logs, update_weights=False)
        else:
            epoch_val_loss = logs["loss"]
            delta = self.current_val_loss - epoch_val_loss
            print(tool_box.color_string('white', f"\n\nCURRENT_VAL_LOSS: {epoch_val_loss}; PREVIOUS VAL_LOSS: {self.current_val_loss}; DELTA: {delta}; min_delta: {self.min_delta}\n\n"))
            if delta < self.min_delta:
                #handle min_delta and patience compromise

                if self.current_patience_count >= self.patience:
                    #handle update weights here
                    display = f"\n\nIN CHECK LOSSES - TRIGGERING WEIGHT UPDATE\n\n"
                    self.current_patience_count = 0
                    self.current_val_loss = epoch_val_loss
                    print(tool_box.color_string('red', display))
                    return self._update_losses(logs, update_weights=True)
                else:
                    #handle incrementing patience only
                    self.current_val_loss = epoch_val_loss
                    self.current_patience_count += 1
                    display = f"\n\nIN CHECK LOSSES - MIN DELTA THRESHOLD MET; UPDATING PATIENCE COUNT: {self.current_patience_count}\n"
                    print(tool_box.color_string('yellow', display))
                    return self._update_losses(logs, update_weights=False)
            else:
                #handle regular epoch output (no min_delta)
                print(tool_box.color_string('green', f"\n\nIN _UPDATE_LOSSES - ADDING NEW LOSSES FROM EPOCH...\n\n"))
                return self._update_losses(logs, update_weights=False)

    def _update_losses(self, logs, update_weights):
        '''method called by self._check_losses - updates aggregated losses and update kl_beta and recon_weight if needed'''
        total_loss = logs["loss"]
        kl_loss = logs["kl_divergence"]
        recon_loss = logs["reconstruction_loss"]

        if update_weights == False:
            #scale losses and add to aggregate here
            self.current_total_losses.append(total_loss)
            self.current_kl_val_losses.append(kl_loss)
            self.current_reconstruction_val_losses.append(recon_loss)
            display = f"\n\nIN _UPDATE_LOSSES - LOSSES UPDATED\n\nKL: {self.current_kl_val_losses}\tRECON: {self.current_reconstruction_val_losses}\tTOTAL: {self.current_total_losses}\n\n"
            print(tool_box.color_string('yellow', display))
            return None
        else:
            #handle update here
            display = f"\n\nIN _UPDATE_LOSSES; MIN_DELTA AND PATIENCE THRESHOLD MET....UPDATING WEIGHTS...."
            self.current_total_losses.append(total_loss)
            self.current_kl_val_losses.append(kl_loss)
            self.current_reconstruction_val_losses.append(recon_loss)
            print(tool_box.color_string('cyan', display))
            return None
            #check sum(self.current_total_loss) against sum(kl_loss) and sum(recon_loss) to see proportions; NOTE: losses are scaled so comparison


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

        #initiate loss params
        kl_beta = 1.0
        reconstruction_weight = 1.0
        # self.current_kl_val_loss = None
        # self.current_reconstruction_val_loss = None

        self.min_delta = 0.1
        self.patience = 5
        self.current_val_loss = None
        self.current_patience_count = 0

        self.current_total_losses = []
        self.current_kl_val_losses = []
        self.current_reconstruction_val_losses = []



        self.vae_loss = VAELoss(encoder=self.encoder, kl_beta=kl_beta, reconstruction_weight=reconstruction_weight)
        self.compile(optimizer=optimizer, loss=self.vae_loss, metrics=[ReconstructionLossMetric(), KLDivergenceMetric(beta=kl_beta)])
        
        self.summary()

        cb = [
            keras.callbacks.LambdaCallback(on_epoch_end=lambda epochs, logs: self._check_loss(epochs, logs)),
            ]
        return self.fit(x_train, x_train,batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test), callbacks=cb)



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

latent_dims = 2
vae = VAE(latent_dims=latent_dims)
batch_size = 2056
reshape_dims = (64,64)
kl_beta = 0.1
epochs = 500
cb = [keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001, patience=5, mode="min")]
vae.run_training(x_train=x_train, 
                 x_test=x_test,
                 optimizer="adam", 
                 epochs=epochs, 
                 kl_beta=kl_beta, 
                 learning_rate=0.001, 
                 batch_size=56,
                 callbacks=cb,
                #  reshape_dims=reshape_dims
                )




