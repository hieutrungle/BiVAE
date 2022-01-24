import tensorflow_addons as tfa
from collections import OrderedDict
from utils import utils
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
get_custom_objects().update({'swish': tf.keras.layers.Activation(tf.keras.activations.swish)})


BN_EPS = 1e-5

OPS = OrderedDict([
    ('res_elu', lambda channel, strides: ConvWN(channel, kernel_size=3, strides=strides, padding='same')),
    ('res_wnelu', lambda channel, strides: ConvWNElu(channel, kernel_size=3, strides=strides, padding='same')),
    ('res_bnelu', lambda channel, strides: BNELUConv(channel, kernel_size=3, strides=strides, padding='same')),
    ('res_bnswish', lambda channel, strides: BNSwishConv(channel, kernel_size=3, strides=strides, padding='same')),
    ('res_convbnswish', lambda channel, strides: ConvBNSwish(channel, kernel_size=3, strides=strides, padding='same')),
    ('mconv_e6k5g0', lambda channel, strides: InvertedResidual(channel, kernel_size=5, strides=strides, ex=6, padding='same')),
    ('mconv_e3k5g0', lambda channel, strides: InvertedResidual(channel, kernel_size=5, strides=strides, ex=3, padding='same'))
])

def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner') or cell_type.startswith('ar_nn'):
        strides = 1
    elif cell_type.startswith('down'):
        strides = 2
    elif cell_type.startswith('up'):
        strides = -1
    else:
        raise NotImplementedError(cell_type)

    return strides

def get_skip_connection(channel, strides, kernel_initializer=tf.keras.initializers.HeNormal(), padding='same'):
    
    if strides == 1:
        x_shortcut = Identity()
    elif strides == 2:
        x_shortcut = FactorizedReduce(channel)
    elif strides == -1:
        x_shortcut = tf.keras.layers.Conv2DTranspose(channel, 1,  strides=2,
                                    kernel_initializer=kernel_initializer,
                                    padding=padding)
    return x_shortcut

class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class FactorizedReduce(tf.keras.layers.Layer):
    def __init__(self, channel, **kwargs):
        super(FactorizedReduce, self).__init__(**kwargs)
        assert channel % 2 == 0
        self.conv_1 = tf.keras.layers.Conv2D(channel // 4, 1, strides=2, padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(channel // 4, 1, strides=2, padding="same")
        self.conv_3 = tf.keras.layers.Conv2D(channel // 4, 1, strides=2, padding="same")
        self.conv_4 = tf.keras.layers.Conv2D(channel - 3 * (channel // 4), 1, strides=2, padding="same")

    def call(self, x):
        conv1 = self.conv_1(x)
        conv2 = self.conv_2(x[:, 1:, 1:, :])
        conv3 = self.conv_3(x[:, :, 1:, :])
        conv4 = self.conv_4(x[:, 1:, :, :])
        out = tf.concat([conv1, conv2, conv3, conv4], axis=-1)
        return out

@tf.keras.utils.register_keras_serializable(package='Custom', name='sr')
class SpectralNormRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, sr=0.1, num_iter=1) -> None:
        super().__init__()
        self.sr = sr
        self.num_iter = num_iter
        # self.v = []
        # self.u = []

    def __call__(self, x):
        h, w, c_in, c_out = x.shape
        num_iter = self.num_iter
        # if self.is_init:
        #     self.v = tf.random.normal(shape=(h,w,c_out,1))
        #     num_iter = 10 * num_iter
        #     self.is_init = False

        # for _ in range(num_iter):
        #     self.u = tf.linalg.normalize(tf.matmul(x, self.v), ord=2)[0]
        #     self.v = tf.linalg.normalize(tf.matmul(tf.transpose(x, perm=[0,1,3,2]), self.u), ord=2)[0]
        # sigma = tf.matmul(tf.transpose(self.u, perm=[0,1,3,2]), tf.matmul(x, self.v))

        v = tf.random.normal(shape=(h,w,c_out,1))

        for _ in range(num_iter):
            u = tf.linalg.normalize(tf.matmul(x, v), ord=2)[0]
            v = tf.linalg.normalize(tf.matmul(tf.transpose(x, perm=[0,1,3,2]), u), ord=2)[0]
        sigma = tf.matmul(tf.transpose(u, perm=[0,1,3,2]), tf.matmul(x, v))

        return self.sr*tf.reduce_sum(sigma)

    def get_config(self):
        return {'sr': float(self.sr)}


class ConvWN(tf.keras.layers.Layer):

    def __init__(self, channel, kernel_size=1, strides=1, padding='same', groups=1,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                # kernel_regularizer=SpectralNormRegularizer(sr=0.1,num_iter=2),
                kernel_regularizer=tf.keras.regularizers.L2(l2=0.0025),
                name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv2d_weight_normalization = tfa.layers.WeightNormalization(
                                tf.keras.layers.Conv2D(channel, kernel_size, strides=strides,
                                padding=padding, groups=groups,
                                kernel_initializer=kernel_initializer, 
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer))
    
    def call(self, x):
        x = self.conv2d_weight_normalization(x)
        return x

class ConvWNElu(tf.keras.layers.Layer):
    """WeightNormalization-Conv2D + Elu"""

    def __init__(self, channel, kernel_size=1, strides=1, padding='same', groups=1,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.upsample = strides == -1
        strides = abs(strides)
        self.conv_wn = ConvWN(channel, kernel_size=kernel_size, strides=strides,
                        padding=padding, groups=groups,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        # kernel_regularizer=SpectralNormRegularizer(sr=0.1,num_iter=2),
                        name=name)

    def call(self, x):
        if self.upsample:
            x = tf.keras.layers.UpSampling2D()(x)
        x = self.conv_wn(x)
        out = tf.keras.layers.Activation('elu')(x)
        return out

class BNELUConv(tf.keras.layers.Layer):
    """BN + Elu + Conv2D"""

    def __init__(self, channel, kernel_size=1, strides=1, padding='same', groups=1,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.upsample = strides == -1
        strides = abs(strides)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=BN_EPS, momentum=0.05)
        self.activation = tf.keras.layers.Activation('elu')
        self.conv_0 = tf.keras.layers.Conv2D(channel, kernel_size=kernel_size, strides=strides,
                        padding=padding, groups=groups,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer)
    def forward(self, x):
        x = self.bn(x)
        x = self.activation(x)
        if self.upsample:
            x = tf.keras.layers.UpSampling2D()(x)
        out = self.conv_0(x)
        return out

class BNSwishConv(tf.keras.layers.Layer):
    """BN + Swish + Conv2D"""

    def __init__(self, channel, kernel_size=1, strides=1, padding='same', groups=1,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.upsample = strides == -1
        strides = abs(strides)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=BN_EPS, momentum=0.05)
        self.activation = tf.keras.layers.Activation('swish')
        self.conv_0 = tf.keras.layers.Conv2D(channel, kernel_size=kernel_size, strides=strides,
                        padding=padding, groups=groups,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer)

    def call(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, H, W, C_in)
        """
        x = self.bn(x)
        x = self.activation(x)
        if self.upsample:
            x = tf.keras.layers.UpSampling2D()(x)
        out = self.conv_0(x)
        return out

class ConvBNSwish(tf.keras.layers.Layer):
    def __init__(self, channel, kernel_size=1, strides=1, padding='same', groups=1,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                name=None, **kwargs):
        super(ConvBNSwish, self).__init__(name=name, **kwargs)

        self.conv = tf.keras.Sequential(
            tf.keras.layers.Conv2D(channel, kernel_size=kernel_size, strides=strides,
                        padding=padding, groups=groups,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer),
            tf.keras.layers.BatchNormalization(epsilon=BN_EPS, momentum=0.05),
            tf.keras.layers.Activation('swish')
        )

    def call(self, x):
        return self.conv(x)

class SqueezeAndExcitation(tf.keras.layers.Layer):
    def __init__(self, channel, ratio=8,
                name=None, **kwargs):
        super().__init__(name=name, **kwargs)
 
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(channel//ratio, use_bias=False)
        self.relu = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(channel, use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        x = tf.keras.layers.multiply([x, inputs])
        return x

class EncCombinerCell(tf.keras.layers.Layer):
    def __init__(self, channel, cell_type, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell_type = cell_type
        self.output_channel = channel
        self.conv = ConvWN(channel, kernel_size=3, strides=1, padding="same",
                        kernel_initializer=tf.keras.initializers.HeNormal(),
                        bias_initializer='zeros')

    def call(self, x1, x2):
        # x1: input from encoder
        # x2: input from decoder; thus, need to maps num_channel of x2 to x1
        x2 = self.conv(x2)
        out = x1 + x2
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channel": self.output_channel
        })
        return config

class DecCombinerCell(tf.keras.layers.Layer):
    def __init__(self, channel, cell_type, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell_type = cell_type
        self.conv = ConvWN(channel, kernel_size=1, strides=1, padding="same",
                        kernel_initializer=tf.keras.initializers.HeNormal(),
                        bias_initializer='zeros')

    def call(self, x1, x2):
        # x1: input from previous decoder
        # x2: input from latent vector space
        out = tf.concat([x1, x2], axis=3)
        out = self.conv(out)
        return out

class InvertedResidual(tf.keras.layers.Layer):

    def __init__(self, channel, kernel_size, strides, ex, padding='same', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        assert strides in [1, 2, -1]

        self.hidden_dim = int(round(channel * ex))
        # groups = self.hidden_dim
        groups = 1

        self.conv0 = ConvWNElu(self.hidden_dim, kernel_size=1, strides=strides, padding=padding)
        self.conv1 = ConvWN(self.hidden_dim, kernel_size=kernel_size, strides=1, padding=padding, groups=groups)
        self.conv2 = ConvWN(channel, kernel_size=1, strides=1, padding=padding)

    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        out = self.conv2(x)
        return out

class Cell(tf.keras.layers.Layer):
    def __init__(self, channel, cell_type, cell_archs, use_se, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell_type = cell_type

        strides = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(channel, strides)
        self.use_se = use_se
        self._num_cell_archs = len(cell_archs)
        self._ops = []
        for i in range(self._num_cell_archs):
            stride = strides if i == 0 else 1
            cell_arch = cell_archs[i]
            op = OPS[cell_arch](channel, stride)
            self._ops.append(op)

        if self.use_se:
            self.se = SqueezeAndExcitation(channel)

        self.add_op = tf.keras.layers.Add()

    def call(self, x):
        # skip branch
        x_shortcut = self.skip(x)
        for i in range(self._num_cell_archs):
            x = self._ops[i](x)

        x = self.se(x) if self.use_se else x
        x = 0.5 * x
        x_output = self.add_op([x, x_shortcut])
        return x_output

class PrePriorLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = shape

        # self.pre_prior = tf.Variable(tf.random.normal(shape=self.shape), trainable=True)

    def call(self, pre_prior, z):
        return tf.broadcast_to(pre_prior, [tf.shape(z)[0]] + list(self.shape))


def test_ConvWNElu_resnet(num_convWNElu, channel, strides, input_shape):
    kernel_initializer=tf.keras.initializers.HeNormal()
    padding = "same"
    m = get_skip_connection(channel, strides, kernel_initializer, padding)

    inputs = tf.keras.Input(shape=input_shape)
    x = ConvWNElu(channel, strides=strides)(inputs)
    for _ in range(num_convWNElu-1):
        x = ConvWNElu(channel)(x)
        # print(f"x.shape: {x.shape}")
    x = SqueezeAndExcitation(channel)(x)
    x_shortcut = m(inputs)
    # print(f"x_shortcut.shape: {x_shortcut.shape}")
    outputs = tf.keras.layers.Add()([x, x_shortcut])
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def test_enc_combiner(channel1, channel2, strides, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x1 = ConvWNElu(channel1, strides=strides)(inputs)
    x2 = ConvWNElu(channel2, strides=strides)(inputs)
    outputs = EncCombinerCell(channel1, cell_type="enc_combiner")(x1, x2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def test_dec_combiner(channel1, channel2, strides, input_shape):
    # channel 1
    inputs = tf.keras.Input(shape=input_shape)
    x1 = ConvWNElu(channel1, strides=strides)(inputs)
    x2 = ConvWNElu(channel2, strides=strides)(inputs)
    outputs = DecCombinerCell(channel1, cell_type="dec_combiner")(x1, x2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def test_inverted_residual(channel, strides, input_shape):
    m = get_skip_connection(channel, strides)

    inputs = tf.keras.Input(shape=input_shape)
    x = InvertedResidual(channel, kernel_size=5, strides=strides, ex=6)(inputs)
    x = SqueezeAndExcitation(channel)(x)
    print(f"x.shape: {x.shape}")
    x_shortcut = m(inputs)
    print(f"x_shortcut.shape: {x_shortcut.shape}")
    outputs = tf.keras.layers.Add()([x, x_shortcut])
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def test(test):

    input_shape = (32, 32, 64)
    for strides, name in zip([-1, 1, 2],["up-sampling", "identity", "down-sampling"]):
        print(f"strides: {strides}, name: {name}")
        if test=="convWNElu":
            num_convWNElu = 1
            channel = input_shape[-1]*strides if strides > 0 else input_shape[-1]//2
            model = test_ConvWNElu_resnet(num_convWNElu, channel=channel, strides=strides, input_shape=input_shape)
            model.summary()
        elif test == "enc_combiner":
            channel1 = input_shape[-1]*strides if strides > 0 else input_shape[-1]//2
            channel2 = input_shape[-1]*strides*2 if strides > 0 else input_shape[-1]//4
            model = test_enc_combiner(channel1, channel2, strides, input_shape)
            model.summary()
        elif test == "dec_combiner":
            channel1 = input_shape[-1]*strides if strides > 0 else input_shape[-1]//2
            channel2 = input_shape[-1]*strides*2 if strides > 0 else input_shape[-1]//4
            model = test_dec_combiner(channel1, 20, strides, input_shape)
            model.summary()
        elif test == "invert_res":
            channel = input_shape[-1]*strides if strides > 0 else input_shape[-1]//2
            model = test_inverted_residual(channel=channel, strides=strides, input_shape=input_shape)
            model.summary()
        print("\n")

def test_cell(channel, cell_type, arch_type, input_shape, use_se):
    model_arch = utils.get_model_arch(arch_type)
    inputs = tf.keras.Input(shape=input_shape)
    outputs = Cell(channel, cell_type=cell_type, cell_archs=model_arch[cell_type], use_se=use_se)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':

    input_shape = (32, 32, 64)
    arch_type = "res_wnelu"
    model_arch = utils.get_model_arch(arch_type)
    print(model_arch)
    for cell_type in model_arch.keys():
        print(f"cell_type: {cell_type}")
        strides = get_stride_for_cell_type(cell_type)
        channel = input_shape[-1]*strides if strides > 0 else input_shape[-1]//2
        use_se = False if cell_type=="ar_nn" else True
        model = test_cell(channel=channel, cell_type=cell_type, arch_type=arch_type, input_shape=input_shape, use_se=use_se)
        model.summary()
        print("\n")
    