# https://github.com/pikabite/segmentations_tf2/blob/master/models/bisenet.py

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import Xception


def ConvAndBatch(x, n_filters=64, kernel=(2, 2), strides=(1, 1), padding='valid', activation='relu'):
    filters = n_filters

    conv_ = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)

    batch_norm = BatchNormalization()

    activation = Activation(activation)

    x = conv_(x)
    x = batch_norm(x)
    x = activation(x)

    return x


def ConvAndAct(x, n_filters, kernel=(1, 1), activation='relu', pooling=False):
    poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')
    convLayer = Conv2D(filters=n_filters, kernel_size=kernel, strides=1)

    if activation != None :
        activation = Activation(activation)

    if pooling:
        x = poolingLayer(x)

    x = convLayer(x)
    if activation != None :
        x = activation(x)

    return x


def AttentionRefinmentModule(inputs, n_filters):
    filters = n_filters

    poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')

    x = poolingLayer(inputs)
    x = ConvAndBatch(x, kernel=(1, 1), n_filters=filters, activation='sigmoid')

    return multiply([inputs, x])


def FeatureFusionModule(input_f, input_s, n_filters):
    concatenate = Concatenate(axis=-1)([input_f, input_s])

    branch0 = ConvAndBatch(concatenate, n_filters=n_filters, kernel=(3, 3), padding='same')
    branch_1 = ConvAndAct(branch0, n_filters=n_filters, pooling=True, activation='relu')
    # branch_1 = self.ConvAndAct(branch_1, n_filters=n_filters, pooling=False, activation='sigmoid')
    branch_1 = ConvAndAct(branch_1, n_filters=n_filters, pooling=False, activation=None)

    x = multiply([branch0, branch_1])
    return Add()([branch0, x])


def ContextPath(layer_13, layer_14):
    globalmax = GlobalAveragePooling2D()

    block1 = AttentionRefinmentModule(layer_13, n_filters=1024)
    block2 = AttentionRefinmentModule(layer_14, n_filters=2048)

    global_channels = globalmax(block2)
    block2_scaled = multiply([global_channels, block2])

    block1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(block1)
    block2_scaled = UpSampling2D(size=(4, 4), interpolation='bilinear')(block2_scaled)

    cnc = Concatenate(axis=-1)([block1, block2_scaled])

    return cnc


def Bisnet(input_height, input_width, n_classes=20):
    
    input_image = tf.keras.Input(shape=(input_height, input_width, 3), name="input_image", dtype=tf.float32)

    # x = Lambda(lambda image: preprocess_input(image))(inputs)
    x = input_image

    xception = Xception(weights='imagenet', input_tensor=x, include_top=False)

    tail_prev = xception.get_layer('block13_pool').output
    tail = xception.output

    layer_13, layer_14 = tail_prev, tail

    x = ConvAndBatch(x, 32, strides=2)
    x = ConvAndBatch(x, 64, strides=2)
    x = ConvAndBatch(x, 156, strides=2)

    # context path
    cp = ContextPath(layer_13, layer_14)
    # fusion = self.FeatureFusionModule(cp, x, 32)
    fusion = FeatureFusionModule(cp, x, n_classes)
    ans = UpSampling2D(size=(8, 8), interpolation='bilinear')(fusion)
    
    output = tf.keras.layers.Softmax(axis=3, dtype=tf.float32)(ans)

    return tf.keras.Model(inputs=input_image, outputs=output, name="bisnet")

    # output = self.FinalModel(x, tail_prev, tail)
    # return inputs, xception.input, output