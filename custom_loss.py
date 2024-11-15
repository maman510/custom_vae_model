import tensorflow as tf
from keras import backend as K

class VAELoss(tf.keras.losses.Loss):
    def __init__(self, encoder, **kwargs):
        super(VAELoss, self).__init__(**kwargs)
        self.encoder = encoder  # The encoder model to perform the forward pass

    def call(self, y_true, y_pred):
        # Perform forward pass through encoder
        z_log_var, z_mean, z = self.encoder(y_true)

        # Reconstruction loss (binary crossentropy)
        reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1, 2, 3))

        # KL divergence loss
        kl_loss = - 0.5 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )

        # Return both the total loss (for optimization) and the individual components (for metrics)
        return reconstruction_loss #+ kl_loss  # Total loss for optimization, single tensor



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


class KLDivergenceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='kl_divergence', **kwargs):
        super(KLDivergenceMetric, self).__init__(name=name, **kwargs)
        self.total_kl_loss = self.add_weight(name='total_kl_loss', initializer='zeros')

    def update_state(self, z_mean, z_log_var, sample_weight=None):
        # Calculate KL divergence loss for each sample in the batch
        kl_loss = -0.5 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )
        self.total_kl_loss.assign_add(K.mean(kl_loss))

    def result(self):
        return self.total_kl_loss

    def reset_state(self):
        self.total_kl_loss.assign(0.0)