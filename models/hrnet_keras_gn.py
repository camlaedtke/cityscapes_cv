import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow_addons.layers import GroupNormalization


def conv3x3(x, out_filters, strides=(1, 1), dilation=(1, 1)):
    """3x3 convolution with padding"""
    x = Conv2D(out_filters, 3, padding='same', strides=strides, 
               dilation_rate=dilation, use_bias=False, kernel_initializer='he_normal')(x)
    return x



def basic_Block(x_input, out_filters, strides=(1, 1), with_conv_shortcut=False, final_activation=True, GROUPS = 18):
    x = conv3x3(x_input, out_filters, strides)
    x = GroupNormalization(axis=3, groups=GROUPS)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = GroupNormalization(axis=3, groups=GROUPS)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(x_input)
        residual = GroupNormalization(axis=3, groups=GROUPS)(residual)
        x = add([x, residual])
    else:
        x = add([x, x_input])

    if final_activation:
        x = Activation('relu')(x)
        
    return x



def bottleneck_Block(x_input, out_filters, strides=(1, 1), with_conv_shortcut=False, GROUPS = 18):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(x_input)
    x = GroupNormalization(axis=3, groups=GROUPS)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = GroupNormalization(axis=3, groups=GROUPS)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = GroupNormalization(axis=3, groups=GROUPS)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(x_input)
        residual = GroupNormalization(axis=3, groups=GROUPS)(residual)
        x = add([x, residual])
    else:
        x = add([x, x_input])

    x = Activation('relu')(x)
    return x


def stem_net(x_input, GROUPS = 32):
    x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x_input)
    x = GroupNormalization(axis=3, groups=GROUPS)(x)
    x = Activation('relu')(x)

    x = bottleneck_Block(x, 256, with_conv_shortcut=True, GROUPS=GROUPS)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False, GROUPS=GROUPS)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False, GROUPS=GROUPS)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False, GROUPS=GROUPS)

    return x


def transition_layer1(x, out_filters_list=[32, 64], GROUPS = 18):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = GroupNormalization(axis=3, groups=GROUPS)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = GroupNormalization(axis=3, groups=GROUPS)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


<<<<<<< HEAD
def make_branch1_0(x, out_filters=32, GROUPS=18):
=======
def make_branch1_0(x, out_filters=32, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    return x


<<<<<<< HEAD
def make_branch1_1(x, out_filters=64, GROUPS=18):
=======
def make_branch1_1(x, out_filters=64, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
    return x


def fuse_layer1(x, out_filters_list=[32, 64], GROUPS = 18):
    x0_0 = x[0]
    x0_1 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = GroupNormalization(axis=3, groups=GROUPS)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x0_1)
    x0 = add([x0_0, x0_1])

    x1_0 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = GroupNormalization(axis=3, groups=GROUPS)(x1_0)
    x1_1 = x[1]
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def transition_layer2(x, out_filters_list=[32, 64, 128], GROUPS = 18):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = GroupNormalization(axis=3, groups=GROUPS)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = GroupNormalization(axis=3, groups=GROUPS)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = GroupNormalization(axis=3, groups=GROUPS)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


<<<<<<< HEAD
def make_branch2_0(x, out_filters=32, GROUPS=18):
=======
def make_branch2_0(x, out_filters=32, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    for i in range(4):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False, GROUPS=GROUPS)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


<<<<<<< HEAD
def make_branch2_1(x, out_filters=64, GROUPS=18):
=======
def make_branch2_1(x, out_filters=64, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    for i in range(4):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False, GROUPS=GROUPS)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


<<<<<<< HEAD
def make_branch2_2(x, out_filters=128, GROUPS=18):
=======
def make_branch2_2(x, out_filters=128, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    for i in range(4):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False, GROUPS=GROUPS)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x



def fuse_layer2(x, out_filters_list=[32, 64, 128], GROUPS = 18):
    
    # add( identity (x0) | upsample x 2 (x1) | upsample x 4 (x2) ) --> x0
    x0_0 = x[0]
    x0_1 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = GroupNormalization(axis=3, groups=GROUPS)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x0_1)
    x0_2 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = GroupNormalization(axis=3, groups=GROUPS)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4), interpolation="bilinear")(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    # add( downsample x 2 (x0) | identity (x1) | upsample x 2 (x2) ) --> x1
    x1_0 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = GroupNormalization(axis=3, groups=GROUPS)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(out_filters_list[1], 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = GroupNormalization(axis=3, groups=GROUPS)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x1_2)
    x1 = add([x1_0, x1_1, x1_2])

    # add( downsample x 4 (x0) | downsample x 2 (x1) | identity (x2) ) --> x2
    x2_0 = Conv2D(out_filters_list[0], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = GroupNormalization(axis=3, groups=GROUPS)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = GroupNormalization(axis=3, groups=GROUPS)(x2_0)
    x2_1 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = GroupNormalization(axis=3, groups=GROUPS)(x2_1)
    x2_2 = x[2]
    x2 = add([x2_0, x2_1, x2_2])
    return [x0, x1, x2]



def transition_layer3(x, out_filters_list=[32, 64, 128, 256], GROUPS = 18):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = GroupNormalization(axis=3, groups=GROUPS)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = GroupNormalization(axis=3, groups=GROUPS)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = GroupNormalization(axis=3, groups=GROUPS)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = GroupNormalization(axis=3, groups=GROUPS)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


<<<<<<< HEAD
def make_branch3_0(x, out_filters=32, GROUPS=18):
=======
def make_branch3_0(x, out_filters=32, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False, GROUPS=GROUPS)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


<<<<<<< HEAD
def make_branch3_1(x, out_filters=64, GROUPS=18):
=======
def make_branch3_1(x, out_filters=64, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False, GROUPS=GROUPS)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


<<<<<<< HEAD
def make_branch3_2(x, out_filters=128, GROUPS=18):
=======
def make_branch3_2(x, out_filters=128, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False, GROUPS=GROUPS)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x


<<<<<<< HEAD
def make_branch3_3(x, out_filters=256, GROUPS=18):
=======
def make_branch3_3(x, out_filters=256, GROUPS=GROUPS):
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    for i in range(3):
        residual = x
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, GROUPS=GROUPS)
        x = basic_Block(x, out_filters, with_conv_shortcut=False, final_activation=False, GROUPS=GROUPS)
        x = Add()([x, residual])
        x = Activation('relu')(x)
    return x



def fuse_layer3(x, out_filters_list=[32, 64, 128, 256], GROUPS = 18):
    x0_0 = x[0]
    
    x0_1 = Conv2D(out_filters_list[1], 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = GroupNormalization(axis=3, groups=GROUPS)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(x0_1)
    
    x0_2 = Conv2D(out_filters_list[2], 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = GroupNormalization(axis=3, groups=GROUPS)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4), interpolation="bilinear")(x0_2)
    
    x0_3 = Conv2D(out_filters_list[3], 1, use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = GroupNormalization(axis=3, groups=GROUPS)(x0_3)
    x0_3 = UpSampling2D(size=(8, 8), interpolation="bilinear")(x0_3)
    
    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    return x0



def final_layer(x, n_classes=20, layernameprefix='model'):
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Conv2D(n_classes, 1, use_bias=False, kernel_initializer='he_normal', name=layernameprefix+'_conv2d')(x)
<<<<<<< HEAD
=======
    x = GroupNormalization(axis=3, groups=n_classes)(x)
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44
    x = Activation('softmax', dtype="float32", name=layernameprefix+'_classification')(x)
    return x


def HRNet(input_height, input_width, n_classes=20, W=32, channel=3, layername='model'):
    
    C, C2, C4, C8 = W, int(W*2), int(W*4), int(W*8)
    G = int(W // 2)
    # 32, 64, 128, 256
    
    inputs = tf.keras.Input(shape=(input_height, input_width, channel))

    x = stem_net(inputs, GROUPS=32) # (64, 64, 256) x 4

    x = transition_layer1(x, out_filters_list = [C, C2], GROUPS=G)
    x0 = make_branch1_0(x[0], out_filters = C, GROUPS=G)
    x1 = make_branch1_1(x[1], out_filters = C2, GROUPS=G)
    x = fuse_layer1([x0, x1], out_filters_list = [C, C2], GROUPS=G)

    x = transition_layer2(x, out_filters_list = [C, C2, C4], GROUPS=G)
    x0 = make_branch2_0(x[0], out_filters = C, GROUPS=G)
    x1 = make_branch2_1(x[1], out_filters = C2, GROUPS=G)
    x2 = make_branch2_2(x[2], out_filters = C4, GROUPS=G)
    x = fuse_layer2([x0, x1, x2], out_filters_list = [C, C2, C4], GROUPS=G)

    x = transition_layer3(x, out_filters_list = [C, C2, C4, C8], GROUPS=G)
    x0 = make_branch3_0(x[0], out_filters = C, GROUPS=G)
    x1 = make_branch3_1(x[1], out_filters = C2, GROUPS=G)
    x2 = make_branch3_2(x[2], out_filters = C4, GROUPS=G)
    x3 = make_branch3_3(x[3], out_filters = C8, GROUPS=G)
    x = fuse_layer3([x0, x1, x2, x3], out_filters_list=[C, C2, C4, C8], GROUPS=G)
<<<<<<< HEAD
    
    x = Conv2D(C, 1, 1, padding="SAME")(x)
    x = GroupNormalization(axis=3, groups=G)(x)
=======
>>>>>>> 880885dfe9f84200730308e0e7f0d12c21bbea44

    out = final_layer(x, n_classes=n_classes, layernameprefix=layername)

    model = tf.keras.models.Model(inputs=inputs, outputs=out, name="HRNet_W{}".format(W))

    return model