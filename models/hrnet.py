# https://github.com/pikabite/segmentations_tf2/blob/master/models/hrnet.py

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Lambda, add, concatenate


def cbr(net, channels, name="", i=0, k=1, bn_mom = 0.01):
    net = Conv2D(filters=channels, kernel_size=k, strides=1, padding="SAME", use_bias=False)(net)
    net = BatchNormalization(momentum=bn_mom)(net)
    net = ReLU(name=name + str(i))(net)
    return net


def cb(net, channels, k=3, bn_mom = 0.01) :
    net = Conv2D(filters=channels, kernel_size=k, strides=1, padding="SAME", use_bias=False)(net)
    net = BatchNormalization(momentum=bn_mom)(net)
    return net


def stage1_layers(input_layer, channels, name=""):
    net = input_layer
    for i in range(4) :
        residual = net
        net = cbr(net, channels, name+"_1", i)
        net = cbr(net, channels, name+"_2", i, k=3)
        net = cb(net, channels)
        net = Add()([net, residual])
        net = ReLU(name=name + "_2_" + str(i))(net)
    return net


def stage2_layers(input_layer, channels, multiple=1, name=""):
    net = input_layer
    for ii in range(multiple) :
        residual = net
        for i in range(4) :
            net = cbr(net, channels, name+"_1", str(ii)+str(i), k=3)
            net = cb(net, channels, k=3)
            net = Add()([net, residual])
            net = ReLU(name=name + "_r1" + str(ii)+str(i))(net)
    return net


def downsample(input_layer, downsize, channels, bn_mom=0.01):

    residual = input_layer
    net = Conv2D(filters=channels, kernel_size=1, strides=1, padding="SAME", use_bias=False)(input_layer)
    net = BatchNormalization(momentum=bn_mom)(net)
    net = ReLU()(net)
    net = Conv2D(filters=channels, kernel_size=3, strides=downsize, padding="SAME", use_bias=False)(net)
    net = BatchNormalization(momentum=bn_mom)(net)
    net = ReLU()(net)
    net = Conv2D(filters=channels, kernel_size=1, strides=1, padding="SAME", use_bias=False)(net)
    net = BatchNormalization(momentum=bn_mom)(net)
    net = ReLU()(net)

    residual = Conv2D(filters=channels, kernel_size=1, strides=downsize, padding="SAME", use_bias=False)(residual)
    residual = BatchNormalization(momentum=bn_mom)(residual)

    net = Add()([net, residual])
    net = ReLU()(net)

    return net


def upsample(input_layer, upsize, channels, bn_mom=0.01):

    net = Conv2D(channels, 1, 1, padding="SAME", use_bias=False)(input_layer)
    net = BatchNormalization(momentum=bn_mom)(net)
    net = ReLU()(net)
    net = Conv2D(channels, 3, 1, padding="SAME", use_bias=False)(net)
    net = BatchNormalization(momentum=bn_mom)(net)
    net = ReLU()(net)
    # net = tk.layers.UpSampling2D(size=upsize, interpolation="bilinear")(net)
    net = Lambda(
        lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[1]*upsize, x.shape[2]*upsize], align_corners=True),
        output_shape=(net.shape[1]*upsize, net.shape[2]*upsize)
        )(net)
    net = BatchNormalization(momentum=bn_mom)(net)
    net = ReLU()(net)

    return net


def HRNet(input_height, input_width, n_classes=20, c=48):
    
    input_image = tf.keras.Input(shape=(input_height, input_width, 3), name="input_image", dtype=tf.float32)

    # Introducing stem, 64 channel fix
    stem1 = downsample(input_image, 2, 64)
    stem2 = downsample(stem1, 2, 64)

    # low level stage is also 64
    after_stem2 = cbr(stem2, 64, "after_stem2")
    stage1 = stage1_layers(after_stem2, 64, "stage1")

    fused1_1 = cbr(stage1, c, "fused1_1")
    fused1_2 = downsample(stage1, 2, c*2)

    stage2_r1 = stage2_layers(fused1_1, c, 1, "stage2_r1")
    stage2_r2 = stage2_layers(fused1_2, c*2, 1, "stage2_r2")

    fused2_1 = tf.keras.layers.add([
        cbr(stage2_r1, c, "fused2_1"),
        upsample(stage2_r2, 2, c)
    ])
    fused2_2 = tf.keras.layers.add([
        downsample(stage2_r1, 2, c*2),
        cbr(stage2_r2, c*2, "fused2_2")
    ])
    fused2_3 = tf.keras.layers.add([
        downsample(stage2_r1, 4, c*4),
        downsample(stage2_r2, 2, c*4)
    ])

    stage3_r1 = stage2_layers(fused2_1, c, 4, "stage3_r1")
    stage3_r2 = stage2_layers(fused2_2, c*2, 4, "stage3_r2")
    stage3_r3 = stage2_layers(fused2_3, c*4, 4, "stage3_r3")

    fused3_1 = tf.keras.layers.add([
        cbr(stage3_r1, c, "fused3_1"),
        upsample(stage3_r2, 2, c),
        upsample(stage3_r3, 4, c)
    ])
    fused3_2 = tf.keras.layers.add([
        downsample(stage3_r1, 2, c*2),
        cbr(stage3_r2, c*2, "fused3_2"),
        upsample(stage3_r3, 2, c*2)
    ])
    fused3_3 = tf.keras.layers.add([
        downsample(stage3_r1, 4, c*4),
        downsample(stage3_r2, 2, c*4),
        cbr(stage3_r3, c*4, "fused3_3")
    ])
    fused3_4 = tf.keras.layers.add([
        downsample(stage3_r1, 8, c*8),
        downsample(stage3_r2, 4, c*8),
        downsample(stage3_r3, 2, c*8),
    ])

    stage4_r1 = stage2_layers(fused3_1, c, 3, "stage4_r1")
    stage4_r2 = stage2_layers(fused3_2, c*2, 3, "stage4_r2")
    stage4_r3 = stage2_layers(fused3_3, c*4, 3, "stage4_r3")
    stage4_r4 = stage2_layers(fused3_4, c*8, 3, "stage4_r4")

    upsampled_output = tf.keras.layers.concatenate([
        stage4_r1,
        upsample(stage4_r2, 2, c*2),
        upsample(stage4_r3, 4, c*4),
        upsample(stage4_r4, 8, c*8)
    ])

    logits = tf.keras.layers.Conv2D(n_classes, 1, 1, padding="SAME")(upsampled_output)

    # print(logits.shape)
    # restore the size
    # logits = tk.layers.UpSampling2D(size=2, interpolation="bilinear")(logits)
    # logits = tk.layers.UpSampling2D(size=2, interpolation="bilinear")(logits)
    logits = tf.keras.layers.Lambda(
        lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[1]*2, x.shape[2]*2], align_corners=True),
        output_shape=(logits.shape[1]*2, logits.shape[2]*2)
        )(logits)
    logits = tf.keras.layers.Lambda(
        lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[1]*2, x.shape[2]*2], align_corners=True),
        output_shape=(logits.shape[1]*2, logits.shape[2]*2)
        )(logits)

    output = tf.keras.layers.Softmax(axis=3, dtype=tf.float32)(logits)
    
    model = tf.keras.Model(inputs=input_image, outputs=output, name="hrnet")
    
    return model