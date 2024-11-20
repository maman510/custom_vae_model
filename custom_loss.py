
import tensorflow as tf
from keras import backend as K
import numpy as np
import tool_box
import math

class VAELoss(tf.keras.losses.Loss):
    def __init__(self, encoder, kl_beta=1, reconstruction_weight=1, **kwargs):
        super(VAELoss, self).__init__(**kwargs)
        self.encoder = encoder  # The encoder model to perform the forward pass
        self.kl_beta = kl_beta
        self.reconstruction_weight = reconstruction_weight
        self.current_epoch = 0
        
    def call(self, y_true=None, y_pred=None, **kwargs):
        # check to see if call was done without kwargs (not possible during train) - call with recursive addition of kwargs for weights
        if len(list(kwargs.keys())) == 0:
            #update weights if needed

            kwargs = {"kl_beta": self.kl_beta, "reconstruction_weight": self.reconstruction_weight, "current_epoch": self.current_epoch}

            return self.call(y_true=y_true, y_pred=y_pred, kwargs=kwargs)
        
        else:
                kwargs = kwargs["kwargs"]
                self.kl_beta = kwargs["kl_beta"]
                self.reconstruction_weight = kwargs["reconstruction_weight"]
                self.current_epoch = kwargs["current_epoch"]
                if y_true == None:
                


                
                    display = f"\n\nIN VAELOSS CALL (epoch #{self.current_epoch})\tkl_beta: {self.kl_beta}\trecon_weight: {self.reconstruction_weight}\tKWARGS: {kwargs}\n\n"
                    print(tool_box.color_string('green', display))
                else:
        
                    display = f"\n\nIN VAELOSS CALL (epoch #{self.current_epoch})\tkl_beta: {self.kl_beta}\trecon_weight: {self.reconstruction_weight}\tKWARGS: {kwargs}\n\n"
                    print(tool_box.color_string('green', display))
                    self.z_log_var, self.z_mean, self.z = self.encoder(y_true)

                    # Reconstruction loss (binary crossentropy)
                    reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1, 2, 3))

                    # KL divergence loss
                    kl_loss = - 0.5 * K.mean(
                        1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1
                    )

                
                    #log_tensor_base = tf.math.log(tensor) / tf.math.log(tf.constant(15.0))

                    total_kl_loss = kl_loss ** self.kl_beta
                    total_reconstruction_loss = self.reconstruction_weight * reconstruction_loss


                    return total_reconstruction_loss +  total_kl_loss 

        
 

class KLDivergenceMetric(tf.keras.metrics.Metric):
    def __init__(self,  vae_loss, name='kl_divergence', **kwargs):
        super(KLDivergenceMetric, self).__init__(name=name, **kwargs)
        self.vae_loss = vae_loss
       # self.beta = self.vae_loss.beta
        self.total_kl_loss = self.add_weight(name='total_kl_loss', initializer='zeros')


    def update_state(self, z_mean, z_log_var, sample_weight=1.0):
        # Calculate KL divergence loss for each sample in the batch
        if sample_weight == None:
            return self.update_state(z_mean, z_log_var, self.vae_loss.kl_beta)
        else:

            kl_loss = -0.5 * K.mean(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
            )
            kl_loss = kl_loss ** sample_weight

            self.total_kl_loss.assign_add(K.mean(kl_loss))
           # print(sample_weight, kl_loss, self.total_kl_loss)
        #self.total_kl_loss.assign_add(K.mean(kl_loss)

    def result(self):
        return self.total_kl_loss

    def reset_state(self):
        self.total_kl_loss.assign(0.0)

class ReconstructionLossMetric(tf.keras.metrics.Metric):
    def __init__(self, name='reconstruction_loss', **kwargs):
        super(ReconstructionLossMetric, self).__init__(name=name, **kwargs)
        self.total_reconstruction_loss = self.add_weight(name='total_reconstruction_loss', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate reconstruction loss for each sample in the batch
        reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1, 2, 3))
        self.total_reconstruction_loss.assign_add(K.mean(reconstruction_loss))


    def result(self):
        return self.total_reconstruction_loss

    def reset_state(self):
        self.total_reconstruction_loss.assign(0.0)