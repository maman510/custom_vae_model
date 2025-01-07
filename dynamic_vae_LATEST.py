import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
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
import librosa
tf.config.run_functions_eagerly(True)

class DynamicVAE(keras.Model):
    '''
        Custom VAE model that adjusts the following weights during training (based on schedule):
            - recon_weight (vgg weight)
            - vgg_weight (ssim weight)
            - Beta (kl divergence weight)

    '''


 
#=============== Public Instance Methods ===================

    def __init__(self, model_name, latent_dims, optimizer_fn,
                 recon_weight=1.0, 
                 kl_beta=.01, 
                 vgg_weight=0.1, 
                 ssim_weight=1.0,
                 recon_decay_rate=1.0,
                 vgg_decay_rate = 0.95,
                 kl_beta_decay_rate=0.95,
                 ssim_decay_rate=1.0,
                 max_epoch=500,
                 loaded_model = False
                 ):

        
        super(DynamicVAE, self).__init__()
        self.model_name = model_name
        self.latent_dims = latent_dims
    
        #NOTE: PASS UNINVOKED OPTIMIZER (e.g. pass: keras.optimizers.Nadam NOT: keras.optimizers.Nadam() - optimizer invoked in build)
        self.optimizer_fn = optimizer_fn

        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, 
            decay_steps=5000, 
            decay_rate=0.8,
            staircase=True
        
        )
        self.last_loss_leader = None
        self.decrement_amount = 0.1

 
#======================================== ROTATE WEIGHTS HERE


        self.recon_decay_rate = recon_decay_rate
        self.vgg_decay_rate = vgg_decay_rate
        self.kl_beta_decay_rate = kl_beta_decay_rate
        self.ssim_decay_rate = ssim_decay_rate
        self.current_epoch = 0
        
        #set weights
        train_loss_weights = False
        self.recon_weight = tf.Variable(recon_weight, trainable=train_loss_weights) #for reconstruction loss
        self.kl_beta = tf.Variable(kl_beta, trainable=train_loss_weights) #for vgg_loss
        self.vgg_weight = tf.Variable(vgg_weight, trainable=train_loss_weights)  #for ssim_loss
        self.ssim_weight = tf.Variable(ssim_weight, trainable=train_loss_weights)

  


#=====================================
        self.max_epoch = max_epoch
        self.loaded_model = loaded_model
        #compile model
        self.compile(optimizer=self.optimizer_fn(self.learning_rate))


    def build(self, input_shape):
        self.build_input_shape = input_shape
        self.encoder = self.build_new_encoder(input_shape[1:], self.latent_dims)
        #self.encoder = self.build_separable_encoder(input_shape[1:], self.latent_dims)
        self.decoder = self.build_decoder(self.latent_dims, input_shape[1:])
        self.vgg_model = self._build_vgg_model()

        self.optimizer = self.optimizer_fn(self.learning_rate)

        return super(DynamicVAE, self).build(input_shape)

    def build_separable_encoder(self,input_shape, latent_dim,update=False):
        encoder_layers = []
        inputs = keras.Input(shape=input_shape)
        encoder_layers.append(inputs)
        if update == False:
            #handle dropout
            
            x = keras.layers.SeparableConv2D(32, kernel_size=(3,3), activation="relu", strides=2, padding="same")(inputs)
            x = keras.layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            x = keras.layers.Dropout(0.2)(x) #=== dropout =====
    
            x = keras.layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            x = keras.layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            x = keras.layers.Dropout(0.2)(x) #=== dropout =====
        
            x = keras.layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            x = keras.layers.SeparableConv2D(256, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
    
        else:
            #handle no dropout in case update = True (post-plateu); use MaxPooling instead of dropout
            x = keras.layers.SeparableConv2D(32, kernel_size=(3,3), activation="relu", strides=2, padding="same")(inputs)
            x = keras.layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)

            x = keras.layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            x = keras.layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
    

            x = keras.layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            x = keras.layers.SeparableConv2D(256, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)


        #start bottleneck
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        z_mean = keras.layers.Dense(latent_dim)(x)
        z_log_var = keras.layers.Dense(latent_dim)(x)
        encoder = keras.Model(inputs, [z_mean, z_log_var], name="separable_encoder")
        return encoder

   
    def build_new_encoder(self, input_shape, latent_dim,update=False):
        encoder_layers = []
        inputs = keras.Input(shape=input_shape)
        encoder_layers.append(inputs)
        if update == False:
            #handle dropout
            x = keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", strides=2, padding="same")(inputs)
            encoder_layers.append(x)
            x = keras.layers.Dropout(0.2)(x)
            encoder_layers.append(x)
            x = keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            encoder_layers.append(x)
            x = keras.layers.Dropout(0.2)(x)
            encoder_layers.append(x)
            x = keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
            encoder_layers.append(x)
        else:
            #handle no dropout
            x = keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", strides=1, padding="same")(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D()(x)
            encoder_layers.append(x)
            x = keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", strides=1, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D()(x)
            encoder_layers.append(x)
            x = keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", strides=1, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D()(x)
            encoder_layers.append(x)
        #start bottleneck
        x = keras.layers.Flatten()(x)
        encoder_layers.append(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        encoder_layers.append(x)
        z_mean = keras.layers.Dense(latent_dim)(x)
        encoder_layers.append(x)
        z_log_var = keras.layers.Dense(latent_dim)(x)
        encoder_layers.append(x)
        encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")
       

        #print(tool_box.color_string(f"yellow", f"\nRETURNING ENCODER LAYERS: "))
        for layer in encoder_layers:
            print(layer)
        return encoder
    
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
        #check if model is a new or loaded one; set weights if loaded on first call
        if self.loaded_model == True:
            print(f"\nSETTING LOADED MODEL WEIGHTS...\n")
            self.set_loaded_model_weights()
            self.loaded_model = False


        with tf.GradientTape() as tape:
            #forward pass:
            reconstructed, z_mean, z_log_var = self(original_images)
            self.current_epoch += 1

            #calculate kl_div and recon error
            self.kl_divergence, self.reconstruction_loss, self.vgg_loss, self.ssim_loss, self.total_loss = self.combined_loss(original_images, reconstructed, z_mean, z_log_var)

            #update loss with updated weights
            self.vgg_loss = self.vgg_loss * self.vgg_weight
            self.ssim_loss = self.ssim_loss * self.ssim_weight
        
           
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

       
        ssim_loss = logs["ssim_loss"] * logs["ssim_weight"]
        vgg_loss = logs["vgg_loss"] * logs["vgg_weight"]
        kl_divergence = logs["kl_divergence"] * logs["kl_beta"]
        recon_pct = logs["reconstruction_error"] * logs["recon_weight"]
        combined_loss = logs["loss"]

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

        for loss in losses.keys():
            if loss != max_loss:
                new_weight = current_weights[loss][1] + 0.01
                current_weights[loss][0].assign(new_weight)
          

    
    def training_checkpoint(self, epoch, logs):
 
        
        combined_loss = logs["loss"]
        if epoch == 0:
            self.current_best_loss = combined_loss
            print(tool_box.color_string('yellow', f"\nSETTING INITIAL LOSS: {self.current_best_loss}\n"))
           
            return None
        else:
            if self.current_best_loss > combined_loss:
                self.current_best_loss = combined_loss
                save_path = f"{os.getcwd()}/model_checkpoints/{self.model_name}_checkpoint.pkl"
                self.save(save_path=save_path,overwrite=True)
                print(tool_box.color_string('yellow', f"\nSETTING NEW LOSS: {self.current_best_loss} and saved model: {save_path}\n"))
             
                return None
            else:
                return None



    
    def plot_reconstructed_image(self, original_image):
        # Reconstruct the image using the VAE
        reconstructed_image = self.predict(original_image)[0]
        print(reconstructed_image)
        print(reconstructed_image.shape)
        print(reconstructed_image.dtype)
        # The model returns a tuple (outputs, loss) from the VAE. We need to get the output (reconstructed image).
        #reconstructed_image = reconstructed_image[0]  # This extracts the first element of the tuple, which is the reconstructed image.

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
            "recon_weight": self.recon_weight,
            "kl_beta": self.kl_beta,
            "vgg_weight": self.vgg_weight,
            "ssim_weight": self.ssim_weight,
            "recon_decay_rate": self.recon_decay_rate,
            "vgg_decay_rate":self.vgg_decay_rate,
            "kl_beta_decay_rate": self.kl_beta_decay_rate,
            "ssim_decay_rate": self.ssim_decay_rate,
            "loaded_model": True    #set loaded model as true so it is reflected on load_model and passed to __init__
        }
        
        return custom_configs



#===========    PRIVATE INSTANCE METHODS =============================



    def _build_vgg_model(self):
        '''
            available layers for vgg16:
            ['input_3', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']

        '''
        vgg = keras.applications.VGG16(weights="imagenet", include_top=False)
      #  'block1_conv2', 'block2_conv2' ,'block5_conv2', 'block5_conv3'
     # Conv1_1, Conv1_2, and Conv2_1
        #layer_names =    ['block1_conv2', 'block2_conv2' ,'block3_conv2','block4_conv2', 'block5_conv2', 'block5_conv3']
        #layer_names =    ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2' ,'block3_conv2','block4_conv2', 'block5_conv2', 'block5_conv3']

        layer_names = ['block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']
        print(layer_names)
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



    def set_loaded_model_weights(self):
        updated_weights = {
            "recon_weight": 0.01,
            "ssim_weight": 0.1,
            "kl_beta": 1.0,
            "vgg_weight": 1.0
        }

        updated_decay_rates = {
            "recon_decay_rate": 0.95,
            "vgg_decay_rate": 0.95,
            "kl_beta_decay_rate": 1.0,
            "ssim_decay_rate":1.0,
        }
        train_loss_weights = False
        self.recon_weight = tf.Variable(updated_weights["recon_weight"], trainable=train_loss_weights) #for reconstruction loss
        self.kl_beta = tf.Variable(updated_weights["kl_beta"], trainable=train_loss_weights) #for vgg_loss
        self.vgg_weight = tf.Variable(updated_weights["vgg_weight"], trainable=train_loss_weights)  #for ssim_loss
        self.ssim_weight = tf.Variable(updated_weights["ssim_weight"], trainable=train_loss_weights)
        
        for i in range(len(updated_weights.keys())):
            new_weight_key = list(updated_weights.keys())[i]
            new_decay_rate = list(updated_decay_rates.keys())[i]
            self.__dict__[new_weight_key].assign(updated_weights[new_weight_key])
            self.__dict__[new_decay_rate] = updated_decay_rates[new_decay_rate]
            print(tool_box.color_string('yellow',f"MODEL {new_weight_key.upper()}: {self.__dict__[new_weight_key]}\nNEW {new_decay_rate.upper()}: {self.__dict__[new_decay_rate]}\n"))
        
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

# #====   UPDATE MODEL WEIGHTS
#             updated_weights = {
#                 "recon_weight": 0.01,
#                 "ssim_weight": 0.1,
#                 "kl_beta": 1.0,
#                 "vgg_weight": 1.0
#             }

#             updated_decay_rates = {
#                 "recon_decay_rate": 0.95,
#                 "vgg_decay_rate": 0.95,
#                 "kl_beta_decay_rate": 1.0,
#                 "ssim_decay_rate":1.0,
#             }

#             train_loss_weights = False
#             model.recon_weight = tf.Variable(updated_weights["recon_weight"], trainable=train_loss_weights) #for reconstruction loss
#             model.kl_beta = tf.Variable(updated_weights["kl_beta"], trainable=train_loss_weights) #for vgg_loss
#             model.vgg_weight = tf.Variable(updated_weights["vgg_weight"], trainable=train_loss_weights)  #for ssim_loss
#             model.ssim_weight = tf.Variable(updated_weights["ssim_weight"], trainable=train_loss_weights)
            
#             for i in range(len(updated_weights.keys())):
#                 new_weight_key = list(updated_weights.keys())[i]
#                 new_decay_rate = list(updated_decay_rates.keys())[i]
#                 model.__dict__[new_weight_key].assign(updated_weights[new_weight_key])
#                 model.__dict__[new_decay_rate] = updated_decay_rates[new_decay_rate]
#                 print(tool_box.color_string('yellow',f"MODEL {new_weight_key.upper()}: {model.__dict__[new_weight_key]}\nNEW {new_decay_rate.upper()}: {model.__dict__[new_decay_rate]}\n"))
            
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



    def _train_checkpoint(self, current_epoch, logs):
        checkpoint_path = f"{os.getcwd()}/model_checkpoints/{self.model_name}_checkpoint_{current_epochs}"
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





    def generate_image(self,img_data, save_path):
        '''
            - accepts img_data as np.ndarray (use X_test[0].numpy() before passing if you are using test or validation from tensors)
            - uses predict to generate and save image
        '''

        test_image = img_data.reshape((-1, *img_data.shape))
        generated_img = self.predict(test_image)[0].squeeze()
        plt.axis("off")
       # plt.tight_layout()
        plt.imshow(generated_img)
        save_path = f"{os.getcwd()}/generated_images/{save_path}"
        plt.savefig(save_path, transparent = True, pad_inches = 0, bbox_inches="tight")
        plt.close()

        print(tool_box.color_string('green', f"\nADDED GENERATED IMAGE: {save_path}\n\n"))





def load_and_preprocess_image(path):
    # Load the image
    image = tf.io.read_file(path)    
    image = tf.image.decode_jpeg(image, channels=3)
    # # Preprocess the image (resize, normalize, etc.)
    image = tf.image.resize(image, [32, 32])

    image = tf.cast(image, tf.float32) / 255.0 
    return image




paths = [f"{os.getcwd()}/train_session_data/male/{f}" for f in os.listdir(f"{os.getcwd()}/train_session_data/male")] + [f"{os.getcwd()}/train_session_data/female/{f}" for f in os.listdir(f"{os.getcwd()}/train_session_data/female")]

test_split = int(len(paths) * .1)

# Create a dataset from the image paths
dataset = tf.data.Dataset.from_tensor_slices(paths)

# Map the dataset to load and preprocess images
dataset = dataset.map(load_and_preprocess_image)

# Create a tensor from the dataset


X_train_paths = paths[test_split:]
X_test_paths = paths[:test_split]

train_dataset = tf.data.Dataset.from_tensor_slices(X_train_paths)
train_dataset = train_dataset.map(load_and_preprocess_image)

test_dataset = tf.data.Dataset.from_tensor_slices(X_test_paths)
test_dataset = test_dataset.map(load_and_preprocess_image)


X_train = tf.convert_to_tensor(list(train_dataset))
X_test = tf.convert_to_tensor(list(test_dataset))





#==== NEW MODEL ===== 

# model_name = "separable_encoder"
# latent_dims = 256
# vae = DynamicVAE(model_name=model_name,
#                  latent_dims=latent_dims,
#                  optimizer_fn=keras.optimizers.Adam,
#                  kl_beta=0.01)




#=== LOAD MODEL ====

model_name = "new_mel"
latent_dims = 256
vae = DynamicVAE.load_model(model_name)
# vae.loaded_model = True
# updated_encoder = vae.build_new_encoder((32,32,3), latent_dims, update=True)
# vae.encoder = updated_encoder


# ===== TRAINING PARAMS ===== 
epochs = 200

cb = [#keras.callbacks.LambdaCallback(on_batch_end=lambda epoch, logs: vae.update_weights(epoch, logs)),
      keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: vae.training_checkpoint(epoch, logs))
      #keras.callbacks.ModelCheckpoint(filepath=f"{os.getcwd()}/model_checkpoints/{model_name}_checkpoint.pkl", save_best_only=True, monitor="loss", mode="min", min_delta=0.001, patience=25)
      ]
vae.fit(X_train,
        epochs=epochs,
        callbacks=cb
)

save_path = f"{os.getcwd()}/trained_models/{model_name}.pkl"
vae.save(save_path)

# for i in range(10):

#     test_image = X_test[i]
#     test_image = test_image.numpy()#.reshape((-1,*test_image.shape))
#     save_path = f"{X_test_paths[i].split('/')[-1].split('.png')[0]}_generated.png"
#     vae.generate_image(test_image, save_path = save_path)





'''
    1) change self.build_vgg_model to self.build_new_vgg_model, like for encoder; 
        - update model blocks to add fine perceptual focus (google)
    2) update learning rate after load 
    3) create 3d 10ms tensors

'''













