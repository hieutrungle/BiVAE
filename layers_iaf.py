import tensorflow as tf
from layers import ConvWN, InvertedResidual, ConvWNElu, Cell
from utils import utils
import tensorflow_addons as tfa
import numpy as np


class MaskedConv2D(tf.keras.layers.Layer):
    """Convolutional layers with masks.

    Convolutional layers with simple implementation of masks type A and B for
    autoregressive models.

    Arguments:
    mask_type: one of `"A"` or `"B".`
    filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    """

    def __init__(self,
                 mask_type,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 kernel_initializer='he_uniform',
                 bias_initializer='zeros',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=(self.kernel_size,
                                             self.kernel_size,
                                             int(input_shape[-1]),
                                             self.filters),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.bias = self.add_weight('bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        center = self.kernel_size // 2

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        mask[center, center + (self.mask_type == 'B'):, :, :] = 0.
        mask[center + 1:, :, :, :] = 0.

        self.mask = tf.constant(mask, dtype=tf.float32, name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = tf.nn.conv2d(input,
                      masked_kernel,
                      strides=[1, self.strides, self.strides, 1],
                      padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_type": self.mask_type,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
        })
        return config

class AutoregressiveCell(tf.keras.layers.Layer):
    def __init__(self, num_channels_of_latent, kernel_size=1, strides=1, ex=1) -> None:
        super().__init__()

        self.hidden_dim = int(round(num_channels_of_latent * ex))
        self.cell_1 = tfa.layers.WeightNormalization(
                MaskedConv2D(mask_type='A', filters=self.hidden_dim, kernel_size=1, strides=strides)),
        
        self.cell_2 = tfa.layers.WeightNormalization(
                MaskedConv2D(mask_type='B', filters=self.hidden_dim, kernel_size=kernel_size, strides=strides)),
        
        self.activation = tf.keras.layers.Activation('elu')

    def cell(self, x):
        x = self.cell_1(x)
        x = self.activation(x)
        x = self.cell_2(x)
        x = self.activation(x)
        return x

class InvertedAutoregressiveFlow(tf.keras.layers.Layer):
    def __init__(self, num_channels_of_latent, cell_type, cell_archs, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.cell_type = cell_type
        self.cell_archs = cell_archs

        ex = 3
        self.concat_op = tf.keras.layers.Concatenate(axis=-1)
        self.autoregressive = AutoregressiveCell(num_channels_of_latent, kernel_size=5, strides=1, ex=ex)

        self.conv_mu = ConvWN(num_channels_of_latent, kernel_size=1)
        self.conv_s = ConvWN(num_channels_of_latent, kernel_size=1)

    def call(self, z, h):
        z_iaf = self.concat_op([z, h])
        z_iaf = self.autoregressive(z_iaf)
        mu = self.conv_mu(z_iaf)
        s = self.conv_s(z_iaf)
        sigma = 1/(1+tf.exp(-s))
        new_z = tf.multiply(z, sigma) + tf.multiply(mu, 1-sigma)

        return new_z, tf.zeros(shape=tf.shape(sigma))


class PairedIAF(tf.keras.layers.Layer):
    def __init__(self, num_channels_of_latent, cell_type, cell_archs, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell1 = InvertedAutoregressiveFlow(num_channels_of_latent, cell_type, cell_archs)
        self.cell2 = InvertedAutoregressiveFlow(num_channels_of_latent, cell_type, cell_archs)

    def call(self, z, h):
        new_z, log_det1 = self.cell1(z, h)
        new_z, log_det2 = self.cell2(new_z, h)

        log_det = tf.add(log_det1, log_det2)
        return new_z, tf.reduce_sum(log_det)


if __name__ == '__main__':
    pass