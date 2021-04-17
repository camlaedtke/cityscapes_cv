import tensorflow as tf
from tensorflow.keras.layers import *


BACKBONES = {
    'resnet50': {
        'model': tf.keras.applications.ResNet50,
        'feature_1': 'conv4_block6_2_relu', # 32 x 64 x 256
        'feature_2': 'conv2_block3_2_relu' # 128 x 256 x 64
    },
    'mobilenetv2': {
        'model': tf.keras.applications.MobileNetV2,
        'feature_1': 'out_relu', # 16x32x1280
        'feature_2': 'block_3_depthwise_relu' # 64x128x144
    },
    'xception': {
        'model': tf.keras.applications.Xception,
        'feature_1': 'block13_sepconv2_act', # 32x64x728
        'feature_2': 'block1_conv1_act' # 256x512x32
    }
}


class ConvBlock(tf.keras.layers.Layer):
    """Convolutional Block for DeepLabV3+
    Convolutional block consisting of Conv2D -> BatchNorm -> ReLU
    Args:
        n_filters:
            number of output filters
        kernel_size:
            kernel_size for convolution
        padding:
            padding for convolution
        kernel_initializer:
            kernel initializer for convolution
        use_bias:
            boolean, whether of not to use bias in convolution
        dilation_rate:
            dilation rate for convolution
        activation:
            activation to be used for convolution
    """
    def __init__(self, n_filters, kernel_size, padding, dilation_rate,
                 kernel_initializer, use_bias, conv_activation=None):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(
            n_filters, kernel_size=kernel_size, padding=padding,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias, dilation_rate=dilation_rate,
            activation=conv_activation)

        self.batch_norm = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        tensor = self.conv(inputs)
        tensor = self.batch_norm(tensor)
        tensor = self.relu(tensor)
        return tensor


class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    """Atrous Spatial Pyramid Pooling layer for DeepLabV3+ architecture."""
    # !pylint:disable=too-many-instance-attributes
    def __init__(self):
        super(AtrousSpatialPyramidPooling, self).__init__()

        # layer architecture components
        self.avg_pool = None
        self.conv1, self.conv2 = None, None
        self.pool = None
        self.out1, self.out6, self.out12, self.out18 = None, None, None, None

    @staticmethod
    def _get_conv_block(kernel_size, dilation_rate, use_bias=False):
        return ConvBlock(256, kernel_size=kernel_size, dilation_rate=dilation_rate,
                         padding='same', use_bias=use_bias, kernel_initializer=tf.keras.initializers.he_normal())

    def build(self, input_shape):
        dummy_tensor = tf.random.normal(input_shape)  # used for calculating
        # output shape of convolutional layers

        self.avg_pool = AveragePooling2D(pool_size=(input_shape[-3], input_shape[-2]))

        self.conv1 = AtrousSpatialPyramidPooling._get_conv_block(kernel_size=1, dilation_rate=1, use_bias=True)

        self.conv2 = AtrousSpatialPyramidPooling._get_conv_block(kernel_size=1, dilation_rate=1)

        dummy_tensor = self.conv1(self.avg_pool(dummy_tensor))

        self.pool = UpSampling2D(
            size=(
                input_shape[-3] // dummy_tensor.shape[1],
                input_shape[-2] // dummy_tensor.shape[2]
            ),
            interpolation='bilinear'
        )

        self.out1, self.out6, self.out12, self.out18 = map(
            lambda tup: AtrousSpatialPyramidPooling._get_conv_block(
                kernel_size=tup[0], dilation_rate=tup[1]
            ),
            [(1, 1), (3, 6), (3, 12), (3, 18)]
        )

    def call(self, inputs, **kwargs):
        tensor = self.avg_pool(inputs)
        tensor = self.conv1(tensor)
        tensor = Concatenate(axis=-1)([
            self.pool(tensor),
            self.out1(inputs),
            self.out6(inputs),
            self.out12(inputs),
            self.out18(inputs)
        ])
        tensor = self.conv2(tensor)
        return tensor


class DeeplabV3Plus(tf.keras.Model):
    """DeeplabV3+ network architecture provider tf.keras.Model implementation.
    Args:
        num_classes:
            number of segmentation classes, effectively - number of output
            filters
        height, width:
            expected height, width of image
        backbone:
            backbone to be used
    """
    def __init__(self, num_classes, backbone='resnet50', **kwargs):
        super(DeeplabV3Plus, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.aspp = None
        self.backbone_feature_1, self.backbone_feature_2 = None, None
        self.input_a_upsampler_getter = None
        self.otensor_upsampler_getter = None
        self.input_b_conv, self.conv1, self.conv2, self.out_conv = (None, None, None, None)

    @staticmethod
    def _get_conv_block(filters, kernel_size, conv_activation=None):
        return ConvBlock(filters, kernel_size=kernel_size, padding='same',
                         conv_activation=conv_activation,
                         kernel_initializer=tf.keras.initializers.he_normal(),
                         use_bias=False, dilation_rate=1)

    @staticmethod
    def _get_upsample_layer_fn(input_shape, factor: int):
        return lambda fan_in_shape: \
            tf.keras.layers.UpSampling2D(
                size=(
                    input_shape[1] // factor // fan_in_shape[1],
                    input_shape[2] // factor // fan_in_shape[2]
                ),
                interpolation='bilinear'
            )

    def _get_backbone_feature(self, feature: str, input_shape) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=input_shape[1:])

        backbone_model = BACKBONES[self.backbone]['model'](
            input_tensor=input_layer, weights='imagenet', include_top=False)

        output_layer = backbone_model.get_layer(BACKBONES[self.backbone][feature]).output
        return tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def build(self, input_shape):
        self.backbone_feature_1 = self._get_backbone_feature('feature_1', input_shape)
        self.backbone_feature_2 = self._get_backbone_feature('feature_2', input_shape)

        self.input_a_upsampler_getter = self._get_upsample_layer_fn(input_shape, factor=4)

        self.aspp = AtrousSpatialPyramidPooling()

        self.input_b_conv = DeeplabV3Plus._get_conv_block(48, kernel_size=(1, 1))

        self.conv1 = DeeplabV3Plus._get_conv_block(256, kernel_size=3, conv_activation='relu')

        self.conv2 = DeeplabV3Plus._get_conv_block(256, kernel_size=3, conv_activation='relu')

        self.otensor_upsampler_getter = self._get_upsample_layer_fn(input_shape, factor=1)

        self.out_conv = Conv2D(self.num_classes, kernel_size=(1, 1), padding='same')
        
        self.out_activation = Activation("softmax", dtype="float32")

    def call(self, inputs, training=None, mask=None):
        input_a = self.backbone_feature_1(inputs)

        input_a = self.aspp(input_a)
        input_a = self.input_a_upsampler_getter(input_a.shape)(input_a)

        input_b = self.backbone_feature_2(inputs)
        input_b = self.input_b_conv(input_b)

        tensor = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        tensor = self.conv2(self.conv1(tensor))

        tensor = self.otensor_upsampler_getter(tensor.shape)(tensor)

        tensor = self.out_conv(tensor)

        return self.out_activation(tensor)