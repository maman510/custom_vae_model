import keras
import keras.applications
import keras.applications.vgg16
import keras.callbacks
import keras.datasets
import keras.layers
import keras.losses
import keras.optimizers
import keras.utils
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import tool_box
import os
from keras.datasets import cifar10
from pprint import pprint

tf.config.run_functions_eagerly(True)

class DynamicVAE(keras.Model):
    '''
        Custom VAE model that adjusts the following weights during training (based on schedule):
            - recon_weight (vgg weight)
            - vgg_weight (ssim weight)
            - Beta (kl divergence weight)

    '''


 
#=============== Public Instance Methods ===================

    def __init__(self, model_name, latent_dims, optimizer_fn, learning_rate, 
                 recon_weight=1.0, 
                 kl_beta=.01, 
                 vgg_weight=0.1, 
                 ssim_weight=1.0,
                 recon_decay_rate=1.0,
                 vgg_decay_rate = 0.95,
                 kl_beta_decay_rate=0.95,
                 ssim_decay_rate=1.0,
                 ):
        super(DynamicVAE, self).__init__()
        self.model_name = model_name
        self.latent_dims = latent_dims

        #NOTE: PASS UNINVOKED OPTIMIZER (e.g. pass: keras.optimizers.Nadam NOT: keras.optimizers.Nadam() - optimizer invoked in build)
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        
        #set weights
        train_loss_weights = False
        self.recon_weight = tf.Variable(recon_weight, trainable=train_loss_weights) #for reconstruction loss
        self.kl_beta = tf.Variable(kl_beta, trainable=train_loss_weights) #for vgg_loss
        self.vgg_weight = tf.Variable(vgg_weight, trainable=train_loss_weights)  #for ssim_loss
        self.ssim_weight = tf.Variable(ssim_weight, trainable=train_loss_weights)

   


        self.recon_decay_rate = recon_decay_rate
        self.vgg_decay_rate = vgg_decay_rate
        self.kl_beta_decay_rate = kl_beta_decay_rate
        self.ssim_decay_rate = ssim_decay_rate
        self.current_epoch = 0

        #self.test_weight = tf.Variable(1.0, trainable=False)
        self.last_loss_leader = None
        self.decrement_amount = 0.1
        #compile
        self.compile(optimizer=self.optimizer_fn(self.learning_rate))


    def build(self, input_shape):
        self.build_input_shape = input_shape
        self.encoder = self.build_encoder(input_shape[1:], self.latent_dims)
        self.decoder = self.build_decoder(self.latent_dims, input_shape[1:])
        self.vgg_model = self._build_vgg_model()

        self.optimizer = self.optimizer_fn(self.learning_rate)

        return super(DynamicVAE, self).build(input_shape)
    
    def build_encoder(self, input_shape, latent_dim):
        inputs = keras.Input(shape=input_shape)
        x = keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", strides=2, padding="same")(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
       # x = keras.layers.BatchNormalization()(x)
        #start bottleneck
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        z_mean = keras.layers.Dense(latent_dim)(x)
        z_log_var = keras.layers.Dense(latent_dim)(x)
        return keras.Model(inputs, [z_mean, z_log_var], name="encoder")
    
    def build_decoder(self, latent_dim, original_shape):
        reshape_layer_shape = (original_shape[0]//4, original_shape[1]//4,64)
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = keras.layers.Dense(128, activation="relu")(latent_inputs)
        
        x = keras.layers.Dense(np.prod(reshape_layer_shape), activation="relu")(x)
        x = keras.layers.Reshape(reshape_layer_shape)(x)
        #begin upsampling
        x = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)

        x = keras.layers.Conv2DTranspose(32, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
       
        outputs = keras.layers.Conv2DTranspose(original_shape[2], kernel_size=(3,3),  activation="linear", padding="same")(x)
        return keras.Model(latent_inputs, outputs, name="decoder")

  
    def combined_loss(self, original, reconstructed, z_mean, z_log_var):
        self.reconstruction_loss = self._compute_reconstruction_error(original, reconstructed)
        vgg_loss = self._vgg_loss(original, reconstructed)
        ssim_loss = self._ssim_loss(original, reconstructed)
        self.kl_divergence = self._compute_kl_divergence(z_mean, z_log_var)

        #set weights for reconstruction, vgg, and ssim losses
        total_loss = (self.recon_weight * self.reconstruction_loss) + (self.vgg_weight * vgg_loss) + (self.ssim_weight * ssim_loss) + (self.kl_divergence * self.kl_beta)
        return self.kl_divergence, self.reconstruction_loss,  vgg_loss, ssim_loss, total_loss
    
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self._sampling(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        
        return reconstructed, z_mean, z_log_var
    
    


    def train_step(self, original_images, **kwargs):
        
       # print(f"\n\nTOP OF CALL\n\n")
      
        self.max_loss = None
        self.current_min_loss = "n/a"
        self.min_delta = 0.001
        self.patience = 3
        self.early_stop_count = 0
        

    
        with tf.GradientTape() as tape:
            #forward pass:
            reconstructed, z_mean, z_log_var = self(original_images)

            #calculate kl_div and recon error
            # ssim_weight, vgg_weight, kl_beta = self.update_weights()
            # self.kl_beta.assign(kl_beta)
            # self.vgg_weight.assign(vgg_weight)
            # self.ssim_weight.assign(ssim_weight)
            self.current_epoch += 1

           # self.vgg_weight, self.ssim_weight, self.kl_beta = self.old_update_weights()
            self.kl_divergence, self.reconstruction_loss, self.vgg_loss, self.ssim_loss, self.total_loss = self.combined_loss(original_images, reconstructed, z_mean, z_log_var)

            #update loss with updated weights
            self.vgg_loss = self.vgg_loss * self.vgg_weight
            self.ssim_loss = self.ssim_loss * self.ssim_weight
        
            #print(f'\n\nSETTING KL BETA: {self.kl_beta}\n\n')
            total_loss = (self.kl_divergence * self.kl_beta) + self.reconstruction_loss + self.vgg_loss + self.ssim_loss

        gradients = tape.gradient(self.total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        

        #handle val
        

    
        
        return {"loss": total_loss,
                "reconstruction_error": self.reconstruction_loss, 
                "kl_divergence": self.kl_divergence,
                "vgg_loss": self.vgg_loss, 
                "ssim_loss": self.ssim_loss,
                "recon_weight": self.recon_weight,
                "kl_beta": self.kl_beta,
                "vgg_weight": self.vgg_weight,
                "ssim_weight": self.ssim_weight,
           
                }
        

            #keep weights the same






    def update_weights(self, epoch, logs):
       # Linear schedule for alpha (VGG loss) and gamma (SSIM loss)
        
        ssim_loss = logs["ssim_loss"] * logs["ssim_weight"]
        vgg_loss = logs["vgg_loss"] * logs["vgg_weight"]
        kl_divergence = logs["kl_divergence"] * logs["kl_beta"]
        recon_pct = logs["reconstruction_error"] * logs["recon_weight"]

        #find total_loss without recon error
        total_loss = (ssim_loss + vgg_loss + kl_divergence)

        losses = {
            "kl_pct": (kl_divergence/total_loss).numpy(),
            "vgg_pct": (vgg_loss/total_loss).numpy(),
            "ssim_pct": (ssim_loss/total_loss).numpy(),            
        }

        max_loss_pct = max([v for v in losses.values()])
        max_loss = [k for k in losses.keys() if losses[k] == max_loss_pct][0]

        #set initial last_loss
        if self.last_loss_leader == None:
            self.last_loss_leader = max_loss
            self.decrement_amount = 0.1
            repeat_loss = False
        #handle setting new loss
        elif max_loss != self.last_loss_leader:
            self.last_loss_leader = max_loss
            self.decrement_amount = 0.1
            repeat_loss = False
        #handle repeat
        else:
            self.decrement_amount = self.decrement_amount + 0.1
            repeat_loss = True




        # print(tool_box.color_string('yellow', f"\nEPOCH {epoch} LOGS (after applied weights for losses):\n"))
        # for k, v in logs.items():
        #     print(tool_box.color_string("green", f"\t{k}:\t{v}"))
        # print("\n\n")
        # print(losses)
        # print(f"\n\nMAX LOSS: {max_loss} \tloss leader: {self.last_loss_leader}\tdecrement_amount: {self.decrement_amount}\trepeat_loss: {repeat_loss}\n\n")
       

        alpha_start, alpha_end = 0.1, 1.0
        kl_beta_start, kl_beta_end = 0.01, 1.0
        gamma_start, gamma_end = 1.0, 0.5
        
   
        # vgg_weight = alpha_start + (alpha_end - alpha_start) * (int(self.current_epoch) / max_epoch) * 100
        # ssim_weight = gamma_start + (gamma_end - gamma_start) * (int(self.current_epoch) / max_epoch) * 100
        
        # kl_beta = kl_beta_start + (kl_beta_end - kl_beta_start) * (int(self.current_epoch) / max_epoch) * 100


        #current weights to key into using keys from losses
        current_weights = {
            "kl_pct": [self.kl_beta,self.kl_beta.numpy()],
            "vgg_pct": [self.vgg_weight, self.vgg_weight.numpy()],
            "ssim_pct": [self.ssim_weight, self.ssim_weight.numpy()]
        }



        #increment largest loss and assign new value
       
        current_weight = current_weights[max_loss][1]
        if current_weight - self.decrement_amount < 0:
            #ensure weight not less than 0
            new_weight = 0.0
            current_weights[max_loss][0].assign(new_weight)
        else:
            new_weight = current_weight - self.decrement_amount
            current_weights[max_loss][0].assign(new_weight)

        #increment non-max_loss keys to increase influence
      #  print(f"\n\nMAX LOSS = {max_loss}: {new_weight}\trepeat_loss: {repeat_loss}\tdecrement_amount: {self.decrement_amount}\n")
        for loss in losses.keys():
            if loss != max_loss:
                new_weight = current_weights[loss][1] + 0.01
                current_weights[loss][0].assign(new_weight)
              #  print(f"NEW {loss}: {new_weight}\n")

       # time.sleep(2)

       
     
        # vgg_weight = self.vgg_weight.numpy() + 0.01
        # kl_beta = self.kl_beta.numpy() + 0.01
        # ssim_weight = self.ssim_weight.numpy() + 0.01
        
        # self.kl_beta.assign(kl_beta)
        # self.vgg_weight.assign(vgg_weight)
        # self.ssim_weight.assign(ssim_weight)

     

#{'kl': 6.7519795e-05, 'vgg': 0.5763013, 'ssim': 0.42363122, 'total_loss': 13.168390426784754, 'kl_beta': 0.01, 'ssim_weight': 1.0, 'vgg_weight': 0.1}
     
        #return vgg_weight, ssim_weight, kl_beta

#{'kl': 0.0013423592, 'vgg': 0.99628246, 'ssim': 0.0023751582}


    
    def reconstruct_image(self, original_image):
        # Reconstruct the image using the VAE
        reconstructed_image = self.predict(original_image)

        # The model returns a tuple (outputs, loss) from the VAE. We need to get the output (reconstructed image).
        reconstructed_image = reconstructed_image[0]  # This extracts the first element of the tuple, which is the reconstructed image.

        # Display the original and reconstructed image side by side
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Original Image
        axes[0].imshow(original_image.squeeze())
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Reconstructed Image
        axes[1].imshow(reconstructed_image.squeeze())
        axes[1].set_title("Reconstructed Image")
        axes[1].axis("off")

        plt.show()

    def save(self, save_path=None, overwrite=False):
        if save_path == None:

            file_path = f"{os.getcwd()}/trained_models/{self.model_name}.pkl"
            checkpoint = False
        else:
            file_path = save_path

            checkpoint = True
        if os.path.exists(file_path) == True and overwrite == False and checkpoint == False:
            response = input(tool_box.color_string('red', f'\n\nMODEL CONFIGS FOUND @ PATH: {file_path}; OVERWRITE EXISTING MODEL? ("y" or "n")\n\n'))
            if response.lower() == 'n':
                return None
            elif response.lower() == 'y':
                print(tool_box.color_string('green', f"\nOVERWRITING EXISTING FILE....\n"))
                return self.save(overwrite=True)
            else:
                print(tool_box.color_string('yellow', f"\nINVALID RESPONSE; USE 'n' or 'y'\n"))
                return self.save(overwrite=False)
        else:
            configs = self.get_config()
            tool_box.Create_Pkl(file_path, configs)
            print(tool_box.color_string('green', f'\n\nSAVED MODEL CONFIGS @ PATH: {file_path}\n\n'))
            return configs
        

    def get_config(self):
        

        #NOTE: IF NOT RESET WEIGHTS (i.e. - save latest decayed value after training), remove "initial" weights and uncomment custom configs keys

        initial_recon_weight = 1.0 #recon loss
        initial_kl_beta = .01   #kl_loss
        initial_ssim_weight = 1.0 #ssim loss
        initial_vgg_weight = 0.1

        recon_decay_rate=1.0,
        vgg_decay_rate = 0.95,
        kl_beta_decay_rate=0.95,
        ssim_decay_rate=0.95,

        custom_configs = {
            "model_name": self.model_name,
            "latent_dims": self.latent_dims,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "vgg_model": self.vgg_model,
            "build_input_shape": self.build_input_shape,
            "optimizer_fn": self.optimizer_fn,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "recon_weight": initial_recon_weight,
            "kl_beta": initial_kl_beta,
            "vgg_weight": initial_vgg_weight,
            "ssim_weight": initial_ssim_weight,
            "recon_decay_rate": recon_decay_rate,
            "vgg_decay_rate": vgg_decay_rate,
            "kl_beta_decay_rate": kl_beta_decay_rate,
            "ssim_decay_rate": ssim_decay_rate,
            # "recon_weight": self.recon_weight,
            # "kl_beta": self.kl_beta,
            # "vgg_weight": self.vgg_weight
        }
        
        return custom_configs



#===========    PRIVATE INSTANCE METHODS =============================



    def _build_vgg_model(self):
        vgg = keras.applications.VGG16(weights="imagenet", include_top=False)
      #  'block1_conv2', 'block2_conv2' ,'block5_conv2', 'block5_conv3'
     # Conv1_1, Conv1_2, and Conv2_1
        #layer_names =    ['block1_conv2', 'block2_conv2' ,'block3_conv2','block4_conv2', 'block5_conv2', 'block5_conv3']
        layer_names =    ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2' ,'block3_conv2','block4_conv2', 'block5_conv2', 'block5_conv3']
        vgg_outputs = [vgg.get_layer(name).output for name in layer_names]
        vgg_model = keras.Model(inputs=vgg.input, outputs=vgg_outputs)
        vgg_model.trainable = False
        return vgg_model
 
    def _vgg_loss(self, original, reconstructed):

        # original_preprocessed = keras.applications.VGG19.preprocess_input(original * 255.0)
        # reconstructed_preprocessed = keras.applications.VGG19.preprocess_input(reconstructed * 255.0)

        original_vgg_loss = self.vgg_model(original)
        reconstructed_vgg_loss = self.vgg_model(reconstructed)
        loss = 0
        #tf.reduce_mean(tf.square(original_vgg_loss -  reconstructed_vgg_loss))
        for original_feat, recon_feat in zip(original_vgg_loss, reconstructed_vgg_loss):
            loss += tf.reduce_mean(tf.square(original_feat -  recon_feat))
        
        return loss

    def _ssim_loss(self, original, reconstructed):
        return 1 - tf.reduce_mean(tf.image.ssim(original, reconstructed, max_val=1.0))
    

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _compute_kl_divergence(self, z_mean, z_log_var):
        kl_div = -0.5 * tf.reduce_sum(1 + z_log_var - 
                        tf.square(z_mean) - 
                        tf.exp(z_log_var), axis=-1)
        
        return tf.reduce_mean(kl_div)
    

    def _compute_reconstruction_error(self, input_images, reconstructed):
        # Use Mean Squared Error (MSE) instead of Huber loss
        mse_loss = tf.keras.losses.MeanSquaredError(reduction="sum")
      
        return mse_loss(input_images, reconstructed)


#===================================    class methods ==========================
    @classmethod
    def from_config(cls, config):
        #extract modeling init params
        model_name = config["model_name"]
        latent_dims = config["latent_dims"]
        input_shape = config["build_input_shape"]
        model_optimizer = config["optimizer_fn"]
        learning_rate = config["learning_rate"]
        recon_weight = config["recon_weight"]
        kl_beta = config["kl_beta"]
        vgg_weight = config["vgg_weight"]
        ssim_weight = config["ssim_weight"]

        recon_decay_rate = 1.0,
        vgg_decay_rate = 0.95,
        kl_beta_decay_rate = 0.95,
        ssim_decay_rate= 0.95,
        #rebuild model
        model = DynamicVAE(model_name=model_name, 
                           latent_dims=latent_dims, 
                           optimizer_fn=model_optimizer, 
                           learning_rate=learning_rate,
                           recon_weight=recon_weight,
                           vgg_weight = vgg_weight,
                           kl_beta=kl_beta,
                           ssim_weight=ssim_weight,
                           recon_decay_rate = recon_decay_rate,
                           vgg_decay_rate = vgg_decay_rate,
                           kl_beta_decay_rate = kl_beta_decay_rate,
                           ssim_decay_rate= ssim_decay_rate,
                           )
        model.build(input_shape)
        
        #load model weights and set
        weights = config["weights"]
        model.set_weights(weights)

        return model


    @classmethod
    def load_model(cls, model_name, path=None):
        if path == None:

            file_path = f"{os.getcwd()}/trained_models/{model_name}.pkl"
        else:
            file_path = path

        if os.path.exists(file_path) == False:
            print(tool_box.color_string('red', f'\n\nNO MODEL CONFIGS FOUND @ PATH: {file_path}\n\n'))
            return None
        else:
            configs = tool_box.Load_Pkl(file_path)
            model = DynamicVAE.from_config(configs)
            print(tool_box.color_string('green', f'\n\nRETURNING MODEL BUILT WITH CONFIGS FOUND @ PATH: {file_path}\n\n'))
            return model


    @classmethod
    # Function to compare weights
    def compare_model_weights(cls,weights_1, weights_2):
        if len(weights_1) != len(weights_2):
            print("The models have different numbers of layers or weights.")
            return False
        
        for i in range(len(weights_1)):
            if not np.array_equal(weights_1[i], weights_2[i]):
                print(f"Layer {i} weights are different.")
                return False

        print("The weights of both models are identical.")
        return True



    def _train_checkpoint(self, epoch, logs):
        checkpoint_path = f"{os.getcwd()}/model_checkpoints/{model_name}_checkpoint_{epochs}"
        self.save(save_path=checkpoint_path)
        print(tool_box.color_string('cyan', f'\n\nSAVING BEST LOSS: {checkpoint_path}\n'))


    def _early_stop(self, epoch, logs):
        #initiate
        if self.current_min_loss == "n/a":
            self.current_min_loss = logs["loss"]
            self.current_loss = logs["loss"]
        
        #check if min delta passed
        if logs["loss"] - self.current_loss < 0:
    

            #handle success; reset early stop count if needed
            if self.early_stop_count > 0:
                self.early_stop_count = 0
            
            #check if beats current min
            if logs["loss"] < self.current_min_loss:
                self.current_min_loss = logs["loss"]
                #call checkpoint
                self._train_checkpoint(epoch, logs)
        
        #handle min delta faile
        else:

            if self.early_stop_count + 1 >= self.patience:
            
                print(tool_box.color_string('red', f"\n\nMAX PATIENCE EXCEEDED TERMINATING\n\n"))
                self.stop_training = True
                #self.update_weights()
                return None
            else:

                self.early_stop_count += 1
                print(tool_box.color_string('red', f"\n\nMIN DELTA FAIL INCREMENTING COUNT: {self.early_stop_count}\n\n"))
            
        self.current_loss = logs["loss"]
        epoch_loss = logs["loss"] - self.current_loss
      
       # logs.update({"stop_count": self.early_stop_count, "min_loss": self.current_min_loss, "patience": self.patience, "epoch_diff": {*epoch_loss}, "min_delta": {*self.min_delta}})

        ssim_loss_pct = logs["ssim_loss"] * logs["ssim_weight"]
        vgg_loss_pct = logs["vgg_loss"] * logs["vgg_weight"]
        kl_loss_pct = logs["kl_divergence"] * logs["kl_beta"]
        recon_pct = logs["reconstruction_error"] * logs["recon_weight"]

        #find total_loss without recon error
        total_loss = (ssim_loss_pct + vgg_loss_pct + kl_loss_pct)

        losses = {
            "kl": kl_loss_pct/total_loss,
            "vgg": vgg_loss_pct/total_loss,
            "ssim": ssim_loss_pct/total_loss,
          
        }

        #of losses find one that accounts for largest proportion of loss and adjust its weight to discount value (decrease its weight)
        max_loss_pct = max([v for v in losses.values()])
        max_loss = [k for k in losses.keys() if losses[k] == max_loss_pct][0]


        print(tool_box.color_string('yellow', f"\nEPOCH {epoch} LOGS (after applied weights for losses):\n"))
        for k, v in logs.items():
            print(tool_box.color_string("green", f"\t{k}:\t{v}"))
        print("\n\n")
        print(losses)
        print(f"MAX LOSS: {max_loss}")
        print(sum([v for v in losses.values()]))
        print("\n\n")

        self.max_loss = max_loss



 
        return None





# tool_box.get_wiki_images("elephant")




(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0


# (X_train, y_train), (X_test, y_test) = tool_box.load_tf_dataset("tf_flowers")






learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, 
            decay_steps=5000, 
            decay_rate=0.8,
            staircase=True
        
        )



optimizer = keras.optimizers.Adam

latent_dims = 256
batch_size = 32
epochs = 500
model_name = f"{latent_dims}_dim_{batch_size}_bs_{batch_size}_epochs_{epochs}"


vae = DynamicVAE(model_name=model_name,
                 optimizer_fn=optimizer,
                 learning_rate=learning_rate,
                 latent_dims=latent_dims,
                 kl_beta=0.01
)







vae.fit(X_train[:10000],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: vae.update_weights(epoch, logs))]
)

save_path = f"./trained_models/{model_name}.pkl"
vae.save(save_path)

# vae = DynamicVAE.load_model(model_name)


# for i in range(10):

#     test_image = X_test[i:i+1] 
#     vae.reconstruct_image(test_image)




'''
    _early_stop_working with "loss" - switch to "val_loss" after updating train_step with .evaluate

 - decoder as for vector classification
'''

# Epoch 425/500 #trained good 
# 13/32 [===========>..................] - ETA: 0s - loss: 122.7144