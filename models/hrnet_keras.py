import tensorflow as tf
from tensorflow.keras.layers import *

BN_MOMENTUM = 0.01



def conv3x3(x, out_filters, strides=(1, 1), dilation=(1, 1)):
    """3x3 convolution with padding"""
    x = Conv2D(out_filters, 3, padding='same', strides=strides, 
               dilation_rate=dilation, use_bias=False)(x)
    return x



def basic_Block(x_input, out_filters, strides=(1, 1), with_conv_shortcut=False, final_activation=True):
    x = conv3x3(x_input, out_filters, strides)
    x = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False)(x_input)
        residual = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(residual)
        x = add([x, residual])
    else:
        x = add([x, x_input])

    if final_activation:
        x = Activation('relu')(x)
        
    return x



def bottleneck_Block(x_input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False)(x_input)
    x = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False)(x_input)
        residual = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(residual)
        x = add([x, residual])
    else:
        x = add([x, x_input])

    x = Activation('relu')(x)
    return x


def stem_net(x_input):
    x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False)(x_input)
    x = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x)
    x = Activation('relu')(x)

    x = bottleneck_Block(x, 256, with_conv_shortcut=True)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)

    return x


def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False)(x)
    x0 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='same', use_bias=False)(x)
    x1 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


def make_branch1_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch1_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer1(x, out_filters_list=[32, 64]):
    x0_0 = x[0]
    x0_1 = Conv2D(out_filters_list[0], 1, use_bias=False)(x[1])
    x0_1 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x0_1)
    x0 = add([x0_0, x0_1])

    x1_0 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='same', use_bias=False)(x[0])
    x1_0 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x1_0)
    x1_1 = x[1]
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def transition_layer2(x, out_filters_list=[32, 64, 128]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False)(x[0])
    x0 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False)(x[1])
    x1 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='same', use_bias=False)(x[1])
    x2 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


def make_branch2_0(x, out_filters=32):
    for i in range(4):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


def make_branch2_1(x, out_filters=64):
    for i in range(4):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


def make_branch2_2(x, out_filters=128):
    for i in range(4):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x



def fuse_layer2(x, out_filters_list=[32, 64, 128]):
    
    # add( identity (x0) | upsample x 2 (x1) | upsample x 4 (x2) ) --> x0
    x0_0 = x[0]
    x0_1 = Conv2D(out_filters_list[0], 1, use_bias=False)(x[1])
    x0_1 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x0_1)
    x0_2 = Conv2D(out_filters_list[0], 1, use_bias=False)(x[2])
    x0_2 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4), interpolation="bilinear")(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    # add( downsample x 2 (x0) | identity (x1) | upsample x 2 (x2) ) --> x1
    x1_0 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='same', use_bias=False)(x[0])
    x1_0 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(out_filters_list[1], 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x1_2)
    x1 = add([x1_0, x1_1, x1_2])

    # add( downsample x 4 (x0) | downsample x 2 (x1) | identity (x2) ) --> x2
    x2_0 = Conv2D(out_filters_list[0], 3, strides=(2, 2), padding='same', use_bias=False)(x[0])
    x2_0 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='same', use_bias=False)(x2_0)
    x2_0 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x2_0)
    x2_1 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='same', use_bias=False)(x[1])
    x2_1 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x2_1)
    x2_2 = x[2]
    x2 = add([x2_0, x2_1, x2_2])
    return [x0, x1, x2]



def transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False)(x[0])
    x0 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False)(x[1])
    x1 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False)(x[2])
    x2 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2), padding='same', use_bias=False)(x[2])
    x3 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


def make_branch3_0(x, out_filters=32):
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


def make_branch3_1(x, out_filters=64):
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


def make_branch3_2(x, out_filters=128):
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


def make_branch3_3(x, out_filters=256):
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x



def fuse_layer3(x, out_filters_list=[32, 64, 128, 256]):
    x0_0 = x[0]
    
    x0_1 = Conv2D(out_filters_list[1], 1, use_bias=False)(x[1])
    x0_1 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x0_1)
    
    x0_2 = Conv2D(out_filters_list[2], 1, use_bias=False)(x[2])
    x0_2 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4), interpolation="bilinear")(x0_2)
    
    x0_3 = Conv2D(out_filters_list[3], 1, use_bias=False)(x[3])
    x0_3 = BatchNormalization(axis=3, momentum=BN_MOMENTUM)(x0_3)
    x0_3 = UpSampling2D(size=(8, 8), interpolation="bilinear")(x0_3)
    
    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    return x0



def final_layer(x, n_classes=20, layernameprefix='model'):
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Conv2D(n_classes, 1, use_bias=False, name=layernameprefix+'_conv2d', dtype="float32")(x)
    return x


def HRNet(input_height, input_width, n_classes=20, W=32, channel=3, layername='model'):
    
    C, C2, C4, C8 = W, int(W*2), int(W*4), int(W*8)
    # 32, 64, 128, 256
    
    inputs = tf.keras.Input(shape=(input_height, input_width, channel))

    x = stem_net(inputs) # (64, 64, 256) x 4

    x = transition_layer1(x, out_filters_list = [C, C2])
    x0 = make_branch1_0(x[0], out_filters = C)
    x1 = make_branch1_1(x[1], out_filters = C2)
    x = fuse_layer1([x0, x1], out_filters_list = [C, C2])

    x = transition_layer2(x, out_filters_list = [C, C2, C4])
    x0 = make_branch2_0(x[0], out_filters = C)
    x1 = make_branch2_1(x[1], out_filters = C2)
    x2 = make_branch2_2(x[2], out_filters = C4)
    x = fuse_layer2([x0, x1, x2], out_filters_list = [C, C2, C4])

    x = transition_layer3(x, out_filters_list = [C, C2, C4, C8])
    x0 = make_branch3_0(x[0], out_filters = C)
    x1 = make_branch3_1(x[1], out_filters = C2)
    x2 = make_branch3_2(x[2], out_filters = C4)
    x3 = make_branch3_3(x[3], out_filters = C8)
    x = fuse_layer3([x0, x1, x2, x3], out_filters_list=[C, C2, C4, C8])
    
    x = Conv2D(C, 1, use_bias=False)(x)

    out = final_layer(x, n_classes=n_classes, layernameprefix=layername)

    model = tf.keras.models.Model(inputs=inputs, outputs=out, name="HRNet_W{}".format(W))

    return model