import numpy as np
import os
import sys
from utils import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import *
from layers import *
from layers_iaf import *
import distribution
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'swish': tf.keras.layers.Activation(tf.keras.activations.swish)})
import tensorflow as tf

activation_and_inits = {
    'tanh':(tf.keras.layers.Activation('tanh'), tf.keras.initializers.GlorotNormal()),
    'sigmoid':(tf.keras.layers.Activation('sigmoid'), tf.keras.initializers.GlorotNormal()),
    'relu':(tf.keras.layers.Activation('relu'), tf.keras.initializers.HeNormal()),
    'softplus':(tf.keras.layers.Activation('softplus'), tf.keras.initializers.HeNormal()),
    'elu':(tf.keras.layers.Activation('elu'), tf.keras.initializers.HeNormal()),
    'swish':(tf.keras.layers.Activation('swish'), tf.keras.initializers.HeNormal()),
    'selu':(tf.keras.layers.Activation('selu'), tf.keras.initializers.LecunNormal())
}

CHANNEL_MULT = 2

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, args, model_arch, global_batch_size, in_shape, name="Variational_AutoEncoder", **kwargs):
        super().__init__(name=name, **kwargs)
        # self.writer = writer
        self.model_name = name
        self.model_arch = model_arch
        self.dataset = args.dataset
        self.crop_output = self.dataset in {'mnist', 'omniglot', 'stacked_mnist'}
        self.use_se = args.use_se
        self.batch_size = global_batch_size

        self.num_scales = args.num_scales
        self.num_groups_per_scale = args.num_groups_per_scale
        self.num_channels_of_latent = args.num_channels_of_latent

        # Adjust number of groups per scale in the top-down fashion
        self.groups_per_scale = utils.groups_per_scale(self.num_scales, 
                                    self.num_groups_per_scale, 
                                    args.is_adaptive,
                                    minimum_groups=args.min_groups_per_scale)

        self.vanilla_vae = self.num_scales == 1 and self.num_groups_per_scale == 1

        # Pre-process and post-process parameter
        self.num_initial_channel = args.num_initial_channel

        # encoder parameteres
        self.num_process_blocks = args.num_process_blocks  # block is defined as series of Normal followed by Down
        self.num_preprocess_cells = args.num_preprocess_cells   # number of cells per block
        self.num_cell_per_group_enc = args.num_cell_per_group_enc  # number of cell for each group encoder

        # decoder parameters
        self.num_postprocess_cells = args.num_postprocess_cells
        self.num_cell_per_group_dec = args.num_cell_per_group_dec  # number of cell for each group decoder

        # general cell parameters
        self.in_shape = in_shape
        self.input_side_len = self.in_shape[0]

        # used for generative block
        channel_scaling = CHANNEL_MULT ** (self.num_process_blocks + self.num_scales - 1)
        final_side_len = self.input_side_len // channel_scaling
        
        self.z0_size = [final_side_len, final_side_len, self.num_channels_of_latent]

        # Map channel to self.num_initial_channel
        self.stem = self.init_stem()

        # preproccess does not change channel size until the last layer
        # After the last layer of each proprocess block, channel size increases by 2, mult = 2
        self.pre_process, mult = self.init_pre_process(mult=1)

        if self.vanilla_vae:
            self.enc_tower = []
        else:
            self.prior_shape = (final_side_len, final_side_len, int(channel_scaling * self.num_initial_channel))
            self.pre_prior = tf.Variable(tf.random.normal(shape=self.prior_shape), trainable=True)
            self.pre_prior_layer = PrePriorLayer(self.prior_shape)
            self.prior = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=self.prior_shape),
                ConvWNElu(self.prior_shape[-1], kernel_size=1, padding="same", name="prior_0"),
                ConvWN(self.prior_shape[-1], kernel_size=1, padding="same", name="prior_1")
            ], name="prior")

            self.enc_tower, mult, self.enc_combiners, self.input_enc_combiners = self.init_encoder_tower(mult)
            
        self.is_nf = args.num_nf > 0
        self.num_nf = args.num_nf

        self.enc0 = self.init_encoder0(mult)
        self.enc_mu_log_sig, self.dec_mu_log_sig, self.nf_cells = self.init_mu_log_sigma(mult)
        self.kl_calculator = distribution.KLCalculator(self.batch_size)

        self.sampler_qs, self.sampler_ps = self.init_sampler()

        if self.vanilla_vae:
            self.dec_tower = []
            self.stem_decoder = ConvWN(mult * self.num_initial_channel, kernel_size=3, strides=1, padding="same",
                                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                                bias_initializer='zeros', name="stem_decoder")
        else:
            self.dec_tower, mult = self.init_decoder_tower(mult)

        self.post_process, mult = self.init_post_process(mult)

        self.decoder_output = self.init_decoder_output(mult)

        # param to calculate kl divergence
        self.log_qs, self.log_ps = list(), list()

        # print(f"len: {len(self.enc_combiners)}")

    def init_stem(self):
        stem = ConvWN(self.num_initial_channel, kernel_size=1, strides=1, padding="same",
                                    kernel_initializer=tf.keras.initializers.HeNormal(),
                                    bias_initializer='zeros', name="init_stem")
        return stem

    def init_pre_process(self, mult):
        pre_process = list()
        for b in range(self.num_process_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:
                    cell_type='down_sampling_pre'
                    cell_archs = self.model_arch[cell_type]
                    channel = int(CHANNEL_MULT * mult * self.num_initial_channel)
                    mult = CHANNEL_MULT * mult
                else:
                    cell_type='normal_pre'
                    cell_archs = self.model_arch[cell_type]
                    channel = self.num_initial_channel * mult
                name = cell_type + "_" + str(b) + "_" + str(c)
                cell = Cell(channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name=name)
                pre_process.append(cell)

        return pre_process, mult

    def init_encoder_tower(self, mult):
        enc_tower = list()
        input_enc_combiners, enc_combiners = list(), list()
        for s in range(self.num_scales):
            for g in range(self.groups_per_scale[s]):
                channel = int(self.num_initial_channel * mult)

                for c in range(self.num_cell_per_group_enc):
                    cell_type = 'normal_enc'
                    cell_archs = self.model_arch[cell_type]
                    name = cell_type + "_" + str(s) + "_" + str(g) + "_" + str(c)
                    cell = Cell(channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name=name)
                    enc_tower.append(cell)

                # add encoder combiner for each group
                if not (s == (self.num_scales - 1) and g == (self.groups_per_scale[s] - 1)):
                    cell_type = 'combiner_enc'
                    name = cell_type + "_" + str(s) + "_" + str(g)
                    cell = EncCombinerCell(channel, cell_type=cell_type, name=name)
                    enc_tower.append(cell)
                    enc_combiners.append(cell)
                    out_shape = [self.input_side_len//mult, self.input_side_len//mult, int(mult*self.num_initial_channel)]
                    input_enc_combiners.append(tf.random.normal(shape=[self.batch_size]+out_shape))

            # down sampling after finishing a scale
            if s < self.num_scales - 1:
                cell_type = 'down_sampling_enc'
                cell_archs = self.model_arch[cell_type]
                name = cell_type + "_" + str(s+1)
                channel = int(CHANNEL_MULT * mult * self.num_initial_channel)
                cell = Cell(channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name=name)
                enc_tower.append(cell)
                mult = CHANNEL_MULT * mult

        return enc_tower, mult, enc_combiners, input_enc_combiners

    def init_encoder0(self, mult):
        channel = int(self.num_initial_channel * mult)
        return ConvWNElu(channel, name="encoder0")

    def init_mu_log_sigma(self, mult):
        # This goes from the top to bottom
        enc_mu_log_sig, dec_mu_log_sig, nf_cells = list(), list(), list()
        
        for s in range(self.num_scales):
            for g in range(self.groups_per_scale[self.num_scales - s - 1]):
                # build mu, log sigma generator for encoder
                cell = ConvWN(2 * self.num_channels_of_latent, kernel_size=3, padding="same", 
                                name="enc_mu_log_sig_"+str(len(enc_mu_log_sig)))
                enc_mu_log_sig.append(cell)
                # build NF
                for _ in range(self.num_nf):
                    cell_type='ar_nn'
                    cell_archs = self.model_arch[cell_type]
                    name = "NF_"+str(len(nf_cells))
                    # cell = InvertedAutoregressiveFlow(self.num_channels_of_latent, cell_type=cell_type, 
                    #                     cell_archs=cell_archs, name=name)
                    cell = PairedIAF(self.num_channels_of_latent, cell_type=cell_type, 
                                        cell_archs=cell_archs, name=name)                 
                    nf_cells.append(cell)
                
                if not (s == 0 and g == 0):
                    # for the first group at the top, use a fixed standard Normal.
                    cell = ConvWN(2*self.num_channels_of_latent, kernel_size=3, padding="same",
                                    name="dec_mu_log_sig_"+str(len(dec_mu_log_sig)))
                    dec_mu_log_sig.append(cell)
            mult = mult // CHANNEL_MULT

        return tuple(enc_mu_log_sig), tuple(dec_mu_log_sig), tuple(nf_cells)

    def init_sampler(self):
        sampler_qs, sampler_ps = list(), list()
        for s in range(self.num_scales):
            for g in range(self.groups_per_scale[self.num_scales - s - 1]):
                sampler_qs.append(distribution.NormalSampler(name="sampler_q_"+str(s)+"_"+str(g)))
                sampler_ps.append(distribution.NormalSampler(name="sampler_p_"+str(s)+"_"+str(g)))
        return sampler_qs, sampler_ps

    def init_decoder_tower(self, mult):
        # create decoder tower
        dec_tower = list()
        for s in range(self.num_scales):
            for g in range(self.groups_per_scale[self.num_scales - s - 1]):
                channel = int(self.num_initial_channel * mult)
                if not (s == 0 and g == 0):
                    for c in range(self.num_cell_per_group_dec):
                        cell_type = 'normal_dec'
                        cell_archs = self.model_arch[cell_type]
                        name = cell_type + "_" + str(s) + "_" + str(g) + "_" + str(c)
                        cell = Cell(channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name=name)
                        dec_tower.append(cell)
                cell_type='combiner_dec'
                name = cell_type + "_" + str(s) + "_" + str(g)
                cell = DecCombinerCell(channel, cell_type=cell_type, name=name)
                dec_tower.append(cell)

            # down cells after finishing a scale
            if s < self.num_scales - 1:
                cell_type='up_sampling_dec'
                cell_archs = self.model_arch[cell_type]
                name = cell_type + "_" + str(s)
                channel = int(self.num_initial_channel * mult // CHANNEL_MULT)
                cell = Cell(channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name=name)
                dec_tower.append(cell)
                mult = mult // CHANNEL_MULT

        return dec_tower, mult

    def init_post_process(self, mult):
        post_process = list()
        for b in range(self.num_process_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    cell_type = 'up_sampling_post'
                    cell_archs = self.model_arch[cell_type]
                    channel = int(self.num_initial_channel * mult // CHANNEL_MULT)
                    mult = mult // CHANNEL_MULT
                else:
                    cell_type='normal_post'
                    cell_archs = self.model_arch[cell_type]
                    channel = int(self.num_initial_channel * mult)
                name = cell_type + "_" + str(b) + "_" + str(c)
                cell = Cell(channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name=name)
                post_process.append(cell)

        return post_process, mult

    def init_decoder_output(self, mult):
        return tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding="same",
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                name="decoder_output")

    def call(self, x, training=False):

        # Init to map channel to self.num_initial_channel
        x = self.stem(x)

        # perform pre-processing
        for cell in self.pre_process:
            x = cell(x)

        # encoder tower
        enc_combiners = self.enc_combiners

        input_enc_combiners = self.input_enc_combiners
        idx_enc = 0
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                input_enc_combiners[idx_enc] = x
                idx_enc += 1
            else:
                x = cell(x)

        # param to calculate kl divergence
        # self.log_qs, self.log_ps = list(), list()

        idx_dec = 0
        ftr = self.enc0(x)
        mu_and_log_sigma_q = self.enc_mu_log_sig[idx_dec](ftr)
        mu_q, log_sigma_q = tf.split(mu_and_log_sigma_q, num_or_size_splits=2, axis=-1) # B x H x W x C
        z, log_q = self.sampler_qs[idx_dec](mu_q, log_sigma_q)
        
        # apply normalizing flows
        nf_offset = 0
        log_det = 0.0
        for i in range(self.num_nf):
            z, cur_log_det = self.nf_cells[i](z, ftr)
            # log_q = tf.math.subtract(log_q, log_det)
            log_det = tf.add(log_det, cur_log_det)
        nf_offset += self.num_nf
        
        # prior for z0
        mu_p, log_sigma_p = tf.zeros(shape=tf.shape(z)), tf.zeros(shape=tf.shape(z))
        # _, log_p = self.sampler_ps[idx_dec](mu_p, log_sigma_p)
        kl = self.kl_calculator(mu_q, log_sigma_q, mu_p, log_sigma_p)
        kl_loss = kl

        # To make sure we do not pass any deterministic features from x to decoder.
        x = 0
        x = self.pre_prior_layer(self.pre_prior, z)
        x = self.prior(x)

        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0: # This calculate log_p and log_q for KL divergence
                    # form prior
                    mu_and_log_sigma_p = self.dec_mu_log_sig[idx_dec - 1](x)
                    mu_p, log_sigma_p = tf.split(mu_and_log_sigma_p, num_or_size_splits=2, axis=-1)

                    # combiner_enc then get mu & log_sig
                    ftr = enc_combiners[-idx_dec](input_enc_combiners[-idx_dec], x)
                    mu_and_log_sigma_q = self.enc_mu_log_sig[idx_dec](ftr)
                    mu_q, log_sigma_q = tf.split(mu_and_log_sigma_q, num_or_size_splits=2, axis=-1)

                    # evaluate log_q(z)
                    z, log_q = self.sampler_qs[idx_dec](tf.math.multiply(tf.add(mu_p, mu_q), 0.5), 
                                            tf.math.multiply(tf.add(log_sigma_p, log_sigma_q), 0.5))

                    # apply NF
                    for i in range(self.num_nf):
                        z, cur_log_det = self.nf_cells[nf_offset + i](z, ftr)
                        # log_q = tf.math.subtract(log_q, log_det)
                        log_det = tf.add(log_det, cur_log_det)
                    nf_offset += self.num_nf

                    # evaluate log_p(z)
                    # _, log_p = self.sampler_ps[idx_dec](mu_p, log_sigma_p)
                    kl = self.kl_calculator(mu_q, log_sigma_q, mu_p, log_sigma_p)
                    kl = tf.add(kl, log_det)
                    kl_loss = tf.add(kl_loss, kl)
                    
                # combiner_dec
                x = cell(x, z)
                idx_dec += 1
            else:
                x = cell(x)

        if self.vanilla_vae:
            x = self.stem_decoder(z)

        for cell in self.post_process:
            x = cell(x)

        x = self.decoder_output(x)

        return x, kl_loss

    def model(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.model_name)

    def cal_kl_components(self):
        kl_all, kl_diag = [], []
        total_log_p, total_log_q = 0.0, 0.0
        for sampler_q, sampler_p in zip(self.sampler_qs, self.sampler_ps):
            kl_per_var = sampler_q.kl(sampler_p)
            kl_diag.append(tf.reduce_mean(tf.reduce_sum(kl_per_var, axis=[1, 2]), axis=0))
            kl_all.append(tf.reduce_sum(kl_per_var, axis=[1, 2, 3]))
        return total_log_q, total_log_p, kl_all, kl_diag
    
    # def cal_kl_components(self):
    #     kl_all, kl_diag = [], []
    #     total_log_p, total_log_q = 0.0, 0.0
    #     for sampler_q, sampler_p, log_q, log_p in zip(self.sampler_qs, self.sampler_ps, self.log_qs, self.log_ps):
    #         kl_per_var = sampler_q.kl(sampler_p)
    #         # if self.is_nf:
    #         #     kl_per_var = log_q - log_p
    #         # else:
    #         #     kl_per_var = sampler_q.kl(sampler_p)

    #         kl_diag.append(tf.reduce_mean(tf.reduce_sum(kl_per_var, axis=[1, 2]), axis=0))
    #         kl_all.append(tf.reduce_sum(kl_per_var, axis=[1, 2, 3]))
    #         # total_log_q += tf.reduce_sum(log_q, axis=[1, 2, 3])
    #         # total_log_p += tf.reduce_sum(log_p, axis=[1, 2, 3])
    #     return total_log_q, total_log_p, kl_all, kl_diag

    def generate_images(self, ftrs):
        # z0_size = [num_samples] + self.z0_size
        assert ftrs[0].shape[0]==self.sampler_qs[0].shape[0], \
                            "batch size of ftrs and sampler should be equal"

        idx_dec = 0
        z = self.sampler_qs[idx_dec].sample()
        nf_offset = 0
        for i in range(self.num_nf):
            z, _ = self.nf_cells[i](z, ftrs[idx_dec])
        nf_offset += self.num_nf

        x = self.pre_prior_layer(self.pre_prior, z)
        x = self.prior(x)
        
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    z = self.sampler_qs[idx_dec].sample()
                    # apply NF
                    for n in range(self.num_nf):
                        z, _ = self.nf_cells[nf_offset + n](z, ftrs[idx_dec])
                    nf_offset += self.num_nf
                # 'combiner_dec'
                x = cell(x, z)
                idx_dec += 1
            else:
                x = cell(x)

        if self.vanilla_vae:
            x = self.stem_decoder(z)

        for cell in self.post_process:
            x = cell(x)

        output = self.decoder_output(x)
        return output

    def reconstruct_images_from_latents(self, z_samples):
        idx_dec = 0
        x = self.pre_prior_layer(self.pre_prior, z_samples[idx_dec])
        x = self.prior(x)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                x = cell(x, z_samples[idx_dec])
                idx_dec += 1
            else:
                x = cell(x)

        if self.vanilla_vae:
            x = self.stem_decoder(z_samples[0])

        for cell in self.post_process:
            x = cell(x)

        output = self.decoder_output(x)
        return output

    def get_latent(self, x):

        # Init to map channel to self.num_initial_channel
        x = self.stem(x)

        # perform pre-processing
        for cell in self.pre_process:
            x = cell(x)

        # encoder tower
        enc_combiners, input_enc_combiners = list(), list()
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                enc_combiners.append(cell)
                input_enc_combiners.append(x)
            else:
                x = cell(x)

        # reverse combiner cells and their input for decoder
        enc_combiners.reverse()
        input_enc_combiners.reverse()

        # param to calculate kl divergence
        self.log_qs, self.log_ps = list(), list()

        idx_dec = 0
        ftr = self.enc0(x)
        mu_and_log_sigma_q = self.enc_mu_log_sig[idx_dec](ftr)
        mu_q, log_sigma_q = tf.split(mu_and_log_sigma_q, num_or_size_splits=2, axis=-1) # B x H x W x C
        z, _ = self.sampler_qs[idx_dec](mu_q, log_sigma_q)

        # apply normalizing flows
        nf_offset = 0
        for i in range(self.num_nf):
            z, _ = self.nf_cells[i](z, ftr)
        nf_offset += self.num_nf

        z_samples = [z]
        ftrs = [ftr]

        # To make sure we do not pass any deterministic features from x to decoder.
        x = 0
        x = self.pre_prior_layer(self.pre_prior, z)
        x = self.prior(x)

        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0: # This calculate log_p and log_q for KL divergence
                    # form prior
                    mu_and_log_sigma_p = self.dec_mu_log_sig[idx_dec - 1](x)
                    mu_p, log_sigma_p = tf.split(mu_and_log_sigma_p, num_or_size_splits=2, axis=-1)

                    # combiner_enc then get mu & log_sig
                    ftr = enc_combiners[idx_dec - 1](input_enc_combiners[idx_dec - 1], x)
                    mu_and_log_sigma_q = self.enc_mu_log_sig[idx_dec](ftr)
                    mu_q, log_sigma_q = tf.split(mu_and_log_sigma_q, num_or_size_splits=2, axis=-1)

                    # evaluate log_q(z)
                    z, _ = self.sampler_qs[idx_dec](tf.math.multiply(tf.add(mu_p, mu_q), 0.5), 
                                                tf.math.multiply(tf.add(log_sigma_p, log_sigma_q), 0.5))
                    # apply NF
                    for n in range(self.num_nf):
                        z, _ = self.nf_cells[nf_offset + n](z, ftr)
                    nf_offset += self.num_nf
                    z_samples.append(z)
                    ftrs.append(ftr)
                    
                # combiner_dec
                x = cell(x, z)
                idx_dec += 1
            else:
                x = cell(x)

        return z_samples, ftrs

if __name__ == "__main__":
    pass