import keras
import keras.applications
import keras.applications.vgg16
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


tf.random.set_seed(42)
np.random.seed(42)
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
                 vgg_weight=1.0, 
                 ssim_weight=0.1,
                 recon_decay_rate=1.0,
                 vgg_decay_rate = 0.95,
                 kl_beta_decay_rate=0.01,
                 ssim_decay_rate=0.95,
                 ):
        super(DynamicVAE, self).__init__()
        self.model_name = model_name
        self.latent_dims = latent_dims

        #NOTE: PASS UNINVOKED OPTIMIZER (e.g. pass: keras.optimizers.Nadam NOT: keras.optimizers.Nadam() - optimizer invoked in build)
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        
        #set weights
        self.recon_weight = recon_weight #for reconstruction loss
        self.kl_beta = kl_beta #for vgg_loss
        self.vgg_weight = vgg_weight  #for ssim_loss
        self.ssim_weight = ssim_weight


        self.recon_decay_rate = recon_decay_rate
        self.vgg_decay_rate = vgg_decay_rate
        self.kl_betacay_rate= kl_beta_decay_rate
        self.ssim_decaate= ssim_decay_rate

        self.current_epoch = 0
        
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
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
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
        outputs = keras.layers.Conv2DTranspose(original_shape[2], kernel_size=(3,3), activation="sigmoid", padding="same")(x)
        return keras.Model(latent_inputs, outputs, name="decoder")

  
    def combined_loss(self, original, reconstructed, z_mean, z_log_var):
        reconstruction_loss = self._compute_reconstruction_error(original, reconstructed)
        vgg_loss = self._vgg_loss(original, reconstructed)
        ssim_loss = self._ssim_loss(original, reconstructed)
        kl_divergence = self._compute_kl_divergence(z_mean, z_log_var)

        #set weights for reconstruction, vgg, and ssim losses
        total_loss = (self.recon_weight * reconstruction_loss) + (self.vgg_weight * vgg_loss) + (self.ssim_weight * ssim_loss) + (kl_divergence * self.kl_beta)
        return kl_divergence, reconstruction_loss,  vgg_loss, ssim_loss, total_loss
    
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self._sampling(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        
        return reconstructed, z_mean, z_log_var
    
    

    


    def train_step(self, original_images):
        
        with tf.GradientTape() as tape:
            #forward pass:
            reconstructed, z_mean, z_log_var = self(original_images)

            #calculate kl_div and recon error
            kl_divergence, reconstruction_loss, perceptual_loss, ssim_loss, total_loss = self.combined_loss(original_images, reconstructed, z_mean, z_log_var)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
 
        self.current_epoch += 1

        return {"loss": total_loss,
                "reconstruction_error": reconstruction_loss, 
                "kl_divergence": kl_divergence,
                "vgg_loss": perceptual_loss, 
                "ssim_loss": ssim_loss,
                "recon_weight": self.recon_weight,
                "kl_beta": self.kl_beta,
                "vgg_weight": self.vgg_weight,
                "ssim_weight": self.ssim_weight
                }
    
    
    def scheduler_callback(self, epoch, logs, recon_weight_decay_rate, ssim_weight_decay_rate, kl_beta_decay_rate, vgg_weight_decay_rate):
        # Update the values of recon_weight, kl_beta, vgg_weight
        if self.current_epoch > 5: 
           self.recon_weight *= recon_weight_decay_rate
           self.kl_beta *= kl_beta_decay_rate
           self.vgg_weight *= vgg_weight_decay_rate
           self.ssim_weight *= ssim_weight_decay_rate
      
    
            #keep weights the same

    
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

    def save(self, overwrite=False):
        file_path = f"{os.getcwd()}/trained_models/{self.model_name}.pkl"
        if os.path.exists(file_path) == True and overwrite == False:
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
        kl_beta_decay_rate=0.01,
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
        layer_names =    ['block3_conv2', 'block3_conv3', 'block4_conv2','block4_conv3']
        vgg_outputs = [vgg.get_layer(name).output for name in layer_names]
        vgg_model = keras.Model(inputs=vgg.input, outputs=vgg_outputs)
        vgg_model.trainable = True
        return vgg_model
 
    def _vgg_loss(self, original, reconstructed):
        original_preprocessed = keras.applications.vgg16.preprocess_input(original * 255.0)
        reconstructed_preprocessed = keras.applications.vgg16.preprocess_input(reconstructed * 255.0)
        original_features = self.vgg_model(original_preprocessed)
        reconstructed_features = self.vgg_model(reconstructed_preprocessed)
        loss = 0
        for original_feat, recon_feat in zip(original_features, reconstructed_features):
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
    

    def _compute_reconstruction_error(self, input_images, reconstructed, delta=1.0):
        huber_loss = tf.keras.losses.Huber(delta=delta)
        return huber_loss(input_images, reconstructed) 

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



        #initial decay rates
        recon_decay_rate = config["recon_decay_rate"]
        vgg_decay_rate = config["vgg_decay_rate"]
        kl_beta_decay_rate = config["kl_beta_decay_rate"]
        ssim_decay_rate= config["ssim_decay_rate"]

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
                           ssim_decay_rate = ssim_decay_rate,
                           )
        model.build(input_shape)
        
        #load model weights and set
        weights = config["weights"]
        model.set_weights(weights)

        return model


    @classmethod
    def load_model(cls, model_name):
        file_path = f"{os.getcwd()}/trained_models/{model_name}.pkl"
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



