import tensorflow as tf
from layers import ConvWN, InvertedResidual, ConvWNElu, Cell
from utils import utils


class AutoregresiveCell(tf.keras.layers.Layer):
    def __init__(self, num_channels_of_latent, cell_type, cell_archs, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.cell_type = cell_type
        
        ex = 3
        self.concat_op = tf.keras.layers.Concatenate(axis=-1)
        self.autoregressive = InvertedResidual(num_channels_of_latent, kernel_size=5, strides=1, ex=ex)
        self.conv_mu = ConvWN(num_channels_of_latent, kernel_size=1)
        self.conv_s = ConvWN(num_channels_of_latent, kernel_size=1)

    def call(self, z, h):
        z = self.concat_op([z, h])
        z = self.autoregressive(z)
        mu = self.conv_mu(z)
        log_s = self.conv_s(z)
        s = tf.math.exp(log_s)
        sigma = 1/(1+tf.exp(-s))
        new_z = tf.multiply(z, sigma) + tf.multiply(mu, 1-sigma)

        # return new_z, log_sigma
        return new_z, tf.zeros(shape=tf.shape(sigma))