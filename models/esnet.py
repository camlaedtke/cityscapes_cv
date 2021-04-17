import tensorflow as tf
from tensorflow.keras.layers import *

def DownsamplingBlock(input_tensor, input_channels, output_channels):
    '''Downsampling Block
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor    -> Input Tensor
        input_channels  -> Number of channels in the input tensor
        output_channels -> Number of output channels
    '''
    x1 = Conv2D(
        output_channels - input_channels, (3, 3),
        strides=(2, 2), use_bias=True, padding='same'
    )(input_tensor)
    x2 = MaxPool2D((2, 2), (2, 2))(input_tensor)
    x = Concatenate(axis = 3)([x1, x2])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def FCU(input_tensor, output_channels, K=3, dropout_prob=0.03):
    '''Factorized Convolutional Unit
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor -> Input Tensor
        K -> Size of Kernel
    '''
    x = Conv2D(
        output_channels, (K, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(input_tensor)
    x = ReLU()(x)
    x = Conv2D(
        output_channels, (1, K),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        output_channels, (K, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = ReLU()(x)
    x = Conv2D(
        output_channels, (1, K),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Add()([input_tensor, x])
    x = Dropout(dropout_prob)(x)
    x = ReLU()(x)
    return x


def PFCU(input_tensor, output_channels):
    '''Parallel Factorized Convolutional Unit
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor -> Input Tensor
        output_channels -> Number of output channels
    '''
    x = Conv2D(output_channels, (3, 1), strides=(1, 1), use_bias=True, padding='same')(input_tensor)
    x = ReLU()(x)
    x = Conv2D(output_channels, (1, 3), strides=(1, 1), use_bias=True, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Branch 1
    branch_1 = Conv2D(
        output_channels, (3, 1), dilation_rate = (2, 2),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_1 = ReLU()(branch_1)
    branch_1 = Conv2D(
        output_channels, (1, 3), dilation_rate = (2, 2),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_1)
    branch_1 = BatchNormalization()(branch_1)
    # Branch 2
    branch_2 = Conv2D(
        output_channels, (3, 1), dilation_rate = (5, 5),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_2 = ReLU()(branch_2)
    branch_2 = Conv2D(
        output_channels, (1, 3), dilation_rate = (5, 5),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_2)
    branch_2 = BatchNormalization()(branch_2)
    # Branch 3
    branch_3 = Conv2D(
        output_channels, (3, 1), dilation_rate = (9, 9),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_3 = ReLU()(branch_3)
    branch_3 = Conv2D(
        output_channels, (1, 3), dilation_rate = (9, 9),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_3)
    branch_3 = BatchNormalization()(branch_3)
    x = Add()([input_tensor, branch_1, branch_2, branch_3])
    x = Dropout(0.3)(x)
    x = ReLU()(x)
    return x


def UpsamplingBlock(input_tensor, output_channels):
    '''Upsampling Block
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor    -> Input Tensor
        output_channels -> Number of output channels
    '''
    x = Conv2DTranspose(output_channels, 3, padding='same', strides=(2, 2), use_bias=True)(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def ESNet(input_height, input_width, n_classes=20):
    inp = tf.keras.layers.Input((input_height, input_width, 3))
    ##### Encoder #####
    # Block 1
    x = DownsamplingBlock(inp, 3, 32)
    x = FCU(x, 32, K=3)
    x = FCU(x, 32, K=3)
    x = FCU(x, 32, K=3)
    # Block 2
    x = DownsamplingBlock(x, 32, 128)
    x = FCU(x, 128, K=5)
    x = FCU(x, 128, K=5)
    # Block 3
    x = DownsamplingBlock(x, 128, 256)
    x = PFCU(x, 256)
    x = PFCU(x, 256)
    x = PFCU(x, 256)
    ##### Decoder #####
    # Block 4
    x = UpsamplingBlock(x, 128)
    x = FCU(x, 128, K=5, dropout_prob=0.0)
    x = FCU(x, 128, K=5, dropout_prob=0.0)
    # Block 5
    x = UpsamplingBlock(x, 32)
    x = FCU(x, 32, K=3, dropout_prob=0.0)
    x = FCU(x, 32, K=3, dropout_prob=0.0)
    output = Conv2DTranspose(n_classes, 3, padding='same', strides=(2, 2), use_bias=True)(x)
    return tf.keras.models.Model(inp, output, name="ESNet")