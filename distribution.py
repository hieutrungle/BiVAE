import tensorflow as tf
import numpy as np
import math


def soft_clamp(x: tf.Tensor):
    # This ensure x to be in [-n;n]
    # This help stablize KL divergence
    n = tf.constant(5.0)
    return tf.math.tanh(tf.divide(x,n)) * n

class KLCalculator(tf.keras.layers.Layer):
    def __init__(self, batch_size, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.batch_size = batch_size

    def call(self, mu_q, log_sigma_q, mu_p, log_sigma_p):
        mu1 = soft_clamp(mu_q)
        sig1 = tf.exp(soft_clamp(log_sigma_q))
        mu2 = soft_clamp(mu_p)
        sig2 = tf.exp(soft_clamp(log_sigma_p))
        
        sig1_sq = tf.square(sig1)
        sig2_sq = tf.square(sig2)
        mu_dif_sq = tf.square(tf.math.subtract(mu2, mu1))

        kl_divergence = tf.math.log((sig2 / tf.add(sig1, 1e-6)) + 1e-6) \
                        + 0.5*((sig1_sq + mu_dif_sq) / tf.add(sig2_sq, 1e-6)) \
                        - 0.5

        return tf.reduce_sum(kl_divergence)

class NormalSampler(tf.keras.layers.Layer):
    def __init__(self, mu=None, log_sigma=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if mu is not None and log_sigma is not None:
            self.sigma = tf.add(tf.exp(log_sigma), 1e-3) # numerical stability
            self.mu = mu

    def call(self, mu, log_sigma, training=False):
        sigma = tf.add(soft_clamp(tf.exp(log_sigma)), 1e-3)
        mu = soft_clamp(mu)
        eps = tf.random.normal(shape=tf.shape(mu))
        z = tf.add(tf.multiply(eps, sigma), mu)

        # eta = (z - mu) / sigma
        # log_pdf = -0.5*(tf.square(eta) + tf.math.log(2*math.pi) + 2*tf.math.log(sigma))
        log_pdf = 0
        
        return z, log_pdf
        # return z

    def sample(self):
        eps = tf.random.normal(shape=tf.shape(self.sigma))
        return eps * self.sigma + self.mu

    def store_parameters(self, mu, log_sigma):
        self.mu = mu
        self.sigma = tf.add(tf.exp(log_sigma), 1e-3)

    def kl(self, normal_dist):
        mu1 = self.mu
        sig1 = self.sigma
        mu2 = normal_dist.mu
        sig2 = normal_dist.sigma
        
        term1 = tf.math.divide(
            tf.math.subtract(mu2, mu1), 
            tf.add(sig2,1e-4)
            )
        term2 = tf.math.divide(
            tf.add(sig1,1e-4), 
            tf.add(sig2,1e-4)
            )

        kl_divergence = tf.subtract(
                tf.subtract(
                    tf.multiply(
                        0.5, 
                        (tf.add(
                            tf.square(term1), 
                            tf.square(term2))
                            )
                        ),
                    0.5
                    ), 
                tf.math.log(
                    tf.add(
                        term2,
                        1e-4)
                    )
                )

        return kl_divergence
