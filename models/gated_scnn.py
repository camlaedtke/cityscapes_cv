import imageio
import tensorflow as tf



def modify_layers(model):
    """
    take the tf.keras.Model and modify the parameters of the layers.
    We will then rebuild the model from json, which will have the updated layers
    but we should still be able to use the pretrained weights
    Originally got this by name, but the name depends on what you call before hand.
    To get the right layer we get what index each are at using the snippet below
    and then hardcode these indices to the code
    # for k, layer in enumerate(model.layers):
    #     if layer.name == 'conv2d_3':
    #         print('conv', k)
    #     if layer.name == 'block13_pool':
    #         print('pool', k)
    #     if layer.name == 'add_6':
    #         print('add', k)
    """
    # modify the last downsampling convolutions
    # we cant get by name as this changes dependening
    # on what you do before!
    conv_layer_index = 122
    # is set to (2, 2) in original model
    model.layers[conv_layer_index].strides = (1, 1)
    # make atrous
    model.layers[conv_layer_index].dilation_rate = 2
    model.layers[conv_layer_index].padding = 'SAME'

    # We also need to turn this maxpool into the identity
    # so that the shapes match up, there is no point
    # in running a max pool filter but keeping the same size
    pool_layer_index = 123
    # maxpools = ['block13_pool']
    model.layers[pool_layer_index].pool_size = (1, 1)
    model.layers[pool_layer_index].strides = (1, 1)
    model.layers[pool_layer_index].padding = 'SAME'

    # add some weight decay
    for layer in model.layers:
        model.get_layer(layer.name).kernel_regularizer = tf.keras.regularizers.l2(l=1e-5)


def build_xception():
    """
    Create an atrous version of tf.keras.applications.Xception
    which uses the pretrained image net weights
    """

    # build original model, save weights, we will modify the layers
    # so that the dilation rate of various convolutions is larger
    # creating atrous convolutions. We will also need to remove the downsampling layers
    model = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=[None, None, 3],)
    modify_layers(model)

    atrous_xception = tf.keras.models.model_from_json(model.to_json())
    atrous_xception.set_weights(model.get_weights())

    return atrous_xception


class AtrousXception(tf.keras.models.Model):
    def __init__(self, **kwargs):
        inception = build_xception()
        super(AtrousXception, self).__init__(inputs=inception.inputs, outputs=inception.outputs, **kwargs)




def resize_to(x, target_t=None, target_shape=None):
    """resize x to shape or target_tensor or target_shape"""
    if target_shape is None:
        s = tf.shape(target_t)
        target_shape = tf.stack([s[1], s[2]])
    return tf.image.resize(x, target_shape, )


def _all_close(x, y, rtol=1e-5, atol=1e-8):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def gradient_mag(tensor, from_rgb=False, eps=1e-12):
    if from_rgb:
        tensor = tf.image.rgb_to_grayscale(tensor[..., :3])
    tensor_edge = tf.image.sobel_edges(tensor)

    def _normalised_mag():
        mag = tf.reduce_sum(tensor_edge ** 2, axis=-1) + eps
        mag = tf.math.sqrt(mag)
        mag /= tf.reduce_max(mag, axis=[1, 2], keepdims=True)
        return mag

    z = tf.zeros_like(tensor)
    normalised_mag = tf.cond(
        _all_close(tensor_edge, tf.zeros_like(tensor_edge)),
        lambda: z,
        _normalised_mag, name='potato')

    return normalised_mag


class GateConv(tf.keras.layers.Layer):
    """
    x                    [b, h, w, c]
    x = batch_norm(x)    [b, h, w, c]
    x = conv(x)          [b, h, w, c]   (1x1) convolution
    x = relu(x)          [b, h, w, c]
    x = conv(x)          [b, h, w, 1]   (1x1) convolution no bias
    x = batch_norm(x)    [b, h, w, 1]
    x = sigmoid(x)       [b, h, w, 1]
    """
    def __init__(self, **kwargs):
        super(GateConv, self).__init__(**kwargs)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            scale=False,
            momentum=0.9)
        self.conv_1 = None
        self.relu = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(
            momentum=0.9)
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=in_channels,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def call(self, x, training=None):
        x = self.batch_norm_1(x, training=training)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x, training=training)
        x = self.sigmoid(x)
        return x


class GatedShapeConv(tf.keras.layers.Layer):
    """
    features, shape                  [b, h, w, c],       [b, h, w, d]
    x = concat([features, shape])    [b, h, w, c + d]
    x = gate_conv(x)                 [b, h, w, 1]
    x = features*(1 + x)             [b, h, w, c]
    x = conv(x)                      [b, h, w, c]
    """
    def __init__(self, **kwargs):
        super(GatedShapeConv, self).__init__(**kwargs)
        self.conv_1 = None
        self.gated_conv = GateConv()

    def build(self, input_shape):
        feature_channels = input_shape[0][-1]
        self.conv_1 = tf.keras.layers.Conv2D(
            feature_channels,
            1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, x, training=None):
        feature_map, shape_map = x
        features = tf.concat([feature_map, shape_map], axis=-1)
        alpha = self.gated_conv(features, training=training)
        gated = feature_map*(alpha + 1.)
        return self.conv_1(gated)


class ResnetPreactUnit(tf.keras.layers.Layer):
    """
    input                   [b, h, w, c]
    x = batch_norm(x)       [b, h, w, c]
    x = relu(x)             [b, h, w, c]
    x = conv(x)             [b, h, w, c]    (3x3) filters
    x = batch_norm(x)       [b, h, w, c]
    x = relu(x)             [b, h, w, c]
    x = conv(x)             [b, h, w, c]    (3x3) filters
    x = x + input           [b, h, w, c]
    """
    def __init__(self, **kwargs):
        super(ResnetPreactUnit, self).__init__(**kwargs)
        self.bn_1 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.relu = tf.keras.layers.ReLU()
        self.conv_1 = None
        self.bn_2 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_2 = None
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        cs = input_shape[-1]

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=cs,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=cs,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def call(self, x, training=None):
        shortcut = x
        x = self.bn_1(x, training)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x, training)
        x = self.relu(x)
        x = self.conv_2(x)
        return self.add([x, shortcut])


class ShapeAttention(tf.keras.layers.Layer):
    """
    s1, s2, s3, s4                         [b, hi, wi, ci] i \\in {1, 2, 3, 4} hi > h{i+1}
    s2 = conv(s2)                          [b, h2, w2, 1] (1x1) conv
    s3 = conv(s3)                          [b, h3, w3, 1] (1x1) conv
    s4 = conv(s4)                          [b, h4, w4, 1] (1x1) conv
    x = res_block(s1)                      [b, h1, w1, c1]
    x, s3 = conv(x), resize(s2)            [b, h1, w1, 32], [b, h1, w1, 1]  (1x1) conv
    x = gated_shape_conv([x, s2])          [b, h1, w1, 32 + 1]
    x = res_block(x)                       [b, h1, w1, 32 + 1]
    x, s3 = conv(x), resize(s3)            [b, h1, w1, 16], [b, h1, w1, 1]  (1x1) conv
    x = gated_shape_conv([x, s3])          [b, h1, w1, 16 + 1]
    x = res_block(x)                       [b, h1, w1, 16 + c3]
    x, s3 = conv(x), resize(s3)            [b, h1, w1, 8], [b, h1, w1, 1]  (1x1) conv
    x = gated_shape_conv([x, s3])          [b, h1, w1, 8 + 1]
    x = conv(x)                            [b, h1, w1, 1]
    x = sigmoid(x)                         [b, h1, w1, 1]
    """
    def __init__(self, **kwargs):
        super(ShapeAttention, self).__init__(**kwargs)

        self.gated_conv_1 = GatedShapeConv()
        self.gated_conv_2 = GatedShapeConv()
        self.gated_conv_3 = GatedShapeConv()

        self.shape_reduction_2 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.shape_reduction_3 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.shape_reduction_4 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        self.res_1 = ResnetPreactUnit()
        self.res_2 = ResnetPreactUnit()
        self.res_3 = ResnetPreactUnit()

        self.reduction_conv_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.reduction_conv_2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.reduction_conv_3 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.reduction_conv_4 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (1,)

    def call(self, x, training=None):
        s1, s2, s3, s4 = x
        s2 = self.shape_reduction_2(s2)
        s3 = self.shape_reduction_3(s3)
        s4 = self.shape_reduction_4(s4)

        # todo these blocks should be a layer
        x = self.res_1(s1, training=training)
        x = self.reduction_conv_1(x)
        s2 = resize_to(s2, target_t=x)
        x = self.gated_conv_1([x, s2], training=training)

        x = self.res_2(x, training=training)
        x = self.reduction_conv_2(x)
        s3 = resize_to(s3, target_t=x)
        x = self.gated_conv_2([x, s3], training=training)

        x = self.res_3(x, training=training)
        x = self.reduction_conv_3(x)
        s4 = resize_to(s4, target_t=x)
        x = self.gated_conv_3([x, s4], training=training)

        x = self.reduction_conv_4(x)
        x = self.sigmoid(x)

        return x


class ShapeStream(tf.keras.layers.Layer):
    """
    shape_intermediate, image_edges                     [b, hi, wi, ci] i \\in {1, 2, 3, 4}, [b, h, w, 1]
    edge_map = shape_attention(shape_intermediate)      [b, h1, w1, 1]
    image_edges = resize(image_edges)                   [b, h1, w1, 1]
    x = concat([edge_map, image_edges])                 [b, h1, w1, 2]
    x = conv(x)                                         [b, h1, w1, 1] (1x1) conv no bias
    x = sigmoid(x)                                      [b, h1, w1, 1]
    return
        x, edge_map
    """
    def __init__(self, **kwargs):
        super(ShapeStream, self).__init__(**kwargs)
        self.shape_attention = ShapeAttention()
        self.reduction_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def compute_output_shape(self, input_shape):
        shape_intermediate_feats, _ = input_shape
        return shape_intermediate_feats[0][:-1] + (1,)

    def call(self, x, training=None):
        shape_backbone_activations, image_edges = x
        edge_out = self.shape_attention(shape_backbone_activations, training=training)
        image_edges = resize_to(image_edges, target_t=edge_out)
        backbone_representation = tf.concat([edge_out, image_edges], axis=-1)
        shape_logits = self.reduction_conv(backbone_representation)
        shape_attention = self.sigmoid(shape_logits)
        return shape_attention, edge_out


class AtrousConvolution(tf.keras.layers.Layer):
    """
    x                    [b, h, w, c]
    x = conv(x)          [b, h, w, c_out]
    """
    def __init__(self, rate, filters, kernel_size, **kwargs):
        super(AtrousConvolution, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.out_channels = filters
        self.rate = rate
        self.depthwise_kernel = None
        self.pointwise_kernel = None
        self.channel_multiplier = 1

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.depthwise_kernel = self.add_weight(
            name='kernel',
            shape=[self.kernel_size, self.kernel_size, in_channels, self.channel_multiplier],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.pointwise_kernel = self.add_weight(
            name='kernel',
            shape=[1, 1, in_channels*self.channel_multiplier, self.out_channels],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4))

    def call(self, x, training=None):
        return tf.nn.separable_conv2d(
            x,
            self.depthwise_kernel,
            self.pointwise_kernel,
            strides=[1, 1, 1, 1],
            dilations=[self.rate, self.rate],
            padding='SAME', )


class AtrousPyramidPooling(tf.keras.layers.Layer):
    """
    takes the two heads from shape stream and backbone, process them, and
    then incorporates an intermediate feature
    backbone_head                         [b, h, w, c]
    x_b = avg_pool(backbone_head)         [b, 1, 1, c]
    x_b = conv(x_b)                       [b, 1, 1, c_out]
    x_b = relu(x_b)                       [b, 1, 1, c_out]
    x_b = batch_norm(x_b)                 [b, 1, 1, c_out]
    x_b = resize(x_b)                     [b, h, w, c_out]
    shape_head                            [b, hp, wp, 1]
    x_s = conv(shape_head)                [b, hp, wp, c_out]
    x_s = relu(x_s)                       [b, hp, wp, c_out]
    x_s = batch_norm(x_s)                 [b, hp, wp, c_out]
    x_s = resize(x_s)                     [b, h,  w,  c_out]
    intermediate_backbone                 [b, hi, wi, ci]
    pyramid_1 = conv(
        intermediate_backbone)            [b, hi, wi, 48]
    # atrous processing of backbone
    x_1 = conv(backbone_head)             [b, h, w, c_out]      (1x1) conv
    x_1 = batch_nrom(x_1)                 [b, h, w, c_out]
    x_1 = relu(x_1)                       [b, h, w, c_out]
    x_2 = conv(backbone_head)             [b, h, w, c_out]      (3x3) atrous conv rate 6
    x_2 = batch_nrom(x_2)                 [b, h, w, c_out]
    x_2 = relu(x_2)                       [b, h, w, c_out]
    x_3 = conv(backbone_head)             [b, h, w, c_out]      (3x3) atrous conv rate 12
    x_3 = batch_nrom(x_3)                 [b, h, w, c_out]
    x_3 = relu(x_3)                       [b, h, w, c_out]
    x_4 = conv(backbone_head)             [b, h, w, c_out]      (3x3) atrous conv rate 16
    x_4 = batch_nrom(x_4)                 [b, h, w, c_out]
    x_4 = relu(x_4)                       [b, h, w, c_out]
    # concat all these ways of processing the
    # head features to form first pyramid layer
    pyramid_0 = concat(                   [b, h, w, c_out*6]
        [x_b,
        x_s,
        x_1,
        x_2,
        x_3,
        x_4])
    pyramid_0 = conv(pyramid_0)           [b, h,  w, 256]
    pyramid_0 = resize(pyramid_0)         [b, hi, wi, 256]
    x = tf.concat(
        [pyramid_0, pyramid_1])           [b, hi, wi, 256 + 48]
    """
    def __init__(self, out_channels, **kwargs):
        super(AtrousPyramidPooling, self).__init__(**kwargs)
        self.relu = tf.keras.layers.ReLU()

        # for final output of backbone
        self.bn_1 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.out_channels = out_channels
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        self.bn_2 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.atrous_conv_1 = AtrousConvolution(rate=6, filters=out_channels, kernel_size=3)

        self.bn_3 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.atrous_conv_2 = AtrousConvolution(rate=12, filters=out_channels, kernel_size=3)

        self.bn_4 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.atrous_conv_3 = AtrousConvolution(rate=18, filters=out_channels, kernel_size=3)

        # for backbone features
        self.bn_img = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_img = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        # for shape features
        self.bn_shape = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_shape = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        # 1x1 reduction convolutions
        self.conv_reduction_1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.conv_reduction_2 = tf.keras.layers.Conv2D(
            filters=48,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def compute_output_shape(self, input_shape):
        return input_shape[2][:-1] + (304,)

    def call(self, x, training=None):
        backbone_head, shape_head, intermediate_backbone = x

        # process backbone features
        img_net = tf.reduce_mean(backbone_head, axis=[1, 2], keepdims=True)
        img_net = self.conv_img(img_net)
        img_net = self.bn_img(img_net, training=training)
        img_net = self.relu(img_net)
        img_net = resize_to(img_net, target_t=backbone_head)

        # process shape head features
        shape_net = self.conv_shape(shape_head)
        shape_net = self.bn_shape(shape_net, training=training)
        shape_net = self.relu(shape_net)
        shape_net = resize_to(shape_net, target_t=backbone_head)

        net = tf.concat([img_net, shape_net], axis=-1)

        # process with atrous
        w = self.conv_1(backbone_head)
        w = self.bn_1(w, training=training)
        w = self.relu(w)

        x = self.atrous_conv_1(backbone_head)
        x = self.bn_2(x, training=training)
        x = self.relu(x)

        y = self.atrous_conv_2(backbone_head)
        y = self.bn_3(y, training=training)
        y = self.relu(y)

        z = self.atrous_conv_3(backbone_head)
        z = self.bn_4(z, training=training)
        z = self.relu(z)

        # atrous output from final layer of backbone
        # and shape stream
        net = tf.concat([net, w, x, y, z], axis=-1)
        net = self.conv_reduction_1(net)

        # combine intermediate representation
        intermediate_backbone = self.conv_reduction_2(intermediate_backbone)
        net = resize_to(net, target_t=intermediate_backbone)
        net = tf.concat([net, intermediate_backbone], axis=-1)

        return net


class FinalLogitLayer(tf.keras.layers.Layer):
    """
    x                            [b, h, w, c]
    x = batch_norm(x)            [b, h, w, c]
    x = conv(x)                  [b, h, w, 256]        (3x3) conv no bias
    x = batch_norm(x)            [b, h, w, 256]
    x = conv(x)                  [b, h, w, 256]        (3x3) conv no bias
    x = batch_norm(x)            [b, h, w, 256]
    x = conv(x)                  [b, h, w, n_classes]  (1x1) conv no bias
    """
    def __init__(self, num_classes, **kwargs):
        super(FinalLogitLayer, self).__init__(**kwargs)
        self.bn_1 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.bn_2 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.bn_3 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)

        self.conv_3 = tf.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding='SAME',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def call(self, x, training=None):
        x = self.bn_1(x, training=training)
        x = self.conv_1(x)
        x = self.bn_2(x, training=training)
        x = self.conv_2(x)
        x = self.bn_3(x, training=training)
        x = self.conv_3(x)
        return x


class XceptionBackbone(tf.keras.layers.Layer):
    ADD_6_LAYER_INDEX = 75
    def __init__(self, **kwargs):
        super(XceptionBackbone, self).__init__(**kwargs)
        self.backbone = None
        backbone = AtrousXception()
        self.backbone = tf.keras.Model(
            backbone.input,
            outputs={
                's1': backbone.get_layer('block2_sepconv2_bn').output,
                's2': backbone.get_layer('block3_sepconv2_bn').output,
                # named add_6 if built on its own, but this is generated automatically
                # and can change, so need to access the layer directly
                's3': backbone.layers[XceptionBackbone.ADD_6_LAYER_INDEX].output,
                's4': backbone.get_layer('block14_sepconv2_act').output,
            })

    def call(self, inputs, training=None):
        inputs = tf.keras.applications.xception.preprocess_input(inputs)
        return self.backbone(inputs, training=training)





class GSCNN(tf.keras.Model):
    def __init__(self, n_classes, **kwargs):
        super(GSCNN, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = XceptionBackbone()
        self.shape_stream = ShapeStream()
        self.atrous_pooling = AtrousPyramidPooling(out_channels=256)
        self.logit_layer = FinalLogitLayer(self.n_classes)

    def call(self, inputs, training=None, mask=None):

        # we need to repeat the input if batch size is 1
        # because in training mode a batch size of 1 will create
        # nans, see:
        # https://github.com/tensorflow/tensorflow/issues/34062
        one_item_batch = tf.shape(inputs)[0] == 1
        if training is None:
            training = True
        inputs = tf.cond(
            tf.logical_and(one_item_batch, training),
            lambda: tf.tile(inputs, (2, 1, 1, 1)),
            lambda: inputs)

        # Backbone
        input_shape = tf.shape(inputs)
        target_shape = tf.stack([input_shape[1], input_shape[2]])
        backbone_feature_dict = self.backbone(inputs, training=training)
        s1, s2, s3, s4 = (backbone_feature_dict['s1'],
                          backbone_feature_dict['s2'],
                          backbone_feature_dict['s3'],
                          backbone_feature_dict['s4'])
        backbone_features = [s1, s2, s3, s4]

        # edge stream
        edge = gradient_mag(inputs, from_rgb=True)
        shape_activations, edge_out = self.shape_stream(
            [backbone_features, edge],
            training=training)

        # aspp
        backbone_activations = backbone_features[-1]
        intermediate_rep = backbone_features[1]
        net = self.atrous_pooling(
            [backbone_activations, shape_activations, intermediate_rep],
            training=training)

        # classify pixels
        net = self.logit_layer(net, training=training)
        net = tf.image.resize(net, target_shape)
        shape_activations = tf.image.resize(shape_activations, target_shape)
        out = tf.concat([net, shape_activations], axis=-1)

        out = tf.cond(one_item_batch, lambda: out[:1], lambda: out)
        return out