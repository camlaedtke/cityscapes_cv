from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.activations import relu
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras import backend as K



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1] 
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        #x = Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
        x = Lambda(lambda x: relu(x, max_value=6.), name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate), name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)

    #x = Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)
    x = Lambda(lambda x: relu(x, max_value=6.), name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    return x


def deeplabv3(input_height, input_width, n_classes=21, OS=16, alpha=1):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
    # Returns
        A Keras model instance.
    """

    img_input = Input(shape=(input_height, input_width, 3))

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv')(img_input)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation("relu", name='Conv_Relu6')(x)
    #x = relu(x, max_value=6., name='Conv_Relu6')

    x = inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, skip_connection=False)

    x = inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, skip_connection=False)
    x = inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, skip_connection=True)

    x = inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3, skip_connection=False)
    x = inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, skip_connection=True)
    x = inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=6, skip_connection=False) 
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=7, skip_connection=True)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=8, skip_connection=True)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=9, skip_connection=True)

    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=10, skip_connection=False)
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=11, skip_connection=True)
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=12, skip_connection=True)

    x = inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2, expansion=6, block_id=13, skip_connection=False)
    x = inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=14, skip_connection=True)
    x = inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=15, skip_connection=True)

    x = inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4, expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    #shape_before = tf.shape(x)
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = GlobalAveragePooling2D()(x)
    
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation("relu")(b4)
    
    # upsample. have to use compat because of the option align_corners
    size_before = K.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3], method='bilinear', align_corners=True))(b4)
    
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation("relu", name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # you can use it with arbitary number of classes
    x = Conv2D(n_classes, (1, 1), padding='same', name='custom_logits_semantic')(x)
    size_before3 = K.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_before3[1:3], method='bilinear', align_corners=True))(x)

    x = Activation("softmax", dtype="float32")(x)

    model = Model(img_input, x, name='deeplabv3plus')

    return model