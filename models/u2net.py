import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.layers import *


class REBNCONV(tf.keras.layers.Layer):
    def __init__(self, out_ch=3, dirate=1):
        super(REBNCONV,self).__init__()
        self.conv_s1_pad = ZeroPadding2D(padding=1*dirate)
        self.conv_s1 = Conv2D(out_ch, kernel_size=3, dilation_rate=1*dirate, padding="valid")
        self.bn_s1 = BatchNormalization()
        self.relu_s1 = Activation("relu")

    def call(self,x):
        hx = x
        hx = self.conv_s1_pad(hx)
        hx = self.conv_s1(hx)
        hx = self.bn_s1(hx)
        xout = self.relu_s1(hx)
        return xout
    
        
# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar, name=None):
    if name is not None:
        src = layers.Lambda(lambda x: tf.image.resize(x, size=(tar.shape[1], tar.shape[2]), method='bilinear'), name=name)(src)
    else:
        src = layers.Lambda(lambda x: tf.image.resize(x, size=(tar.shape[1], tar.shape[2]), method='bilinear'))(src)
    return src

    
### RSU-7 ###
class RSU7(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = MaxPooling2D(2, strides=2, padding="same") 
        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)
        self.pool3 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv4 = REBNCONV(mid_ch, dirate=1)
        self.pool4 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv5 = REBNCONV(mid_ch, dirate=1)
        self.pool5 = MaxPooling2D(2, strides=2, padding="same")

        self.rebnconv6 = REBNCONV(mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch, dirate=1)

        self.rebnconv5d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)

        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(concatenate([hx7, hx6], axis=-1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d =  self.rebnconv5d(concatenate([hx6dup, hx5], axis=-1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(concatenate([hx5dup, hx4], axis=-1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(concatenate([hx4dup, hx3], axis=-1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis=-1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis=-1))

        return hx1d + hxin
    
    
### RSU-6 ###
class RSU6(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)
        self.pool3 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv4 = REBNCONV(mid_ch, dirate=1)
        self.pool4 = MaxPooling2D(2, strides=2, padding="same")

        self.rebnconv5 = REBNCONV(mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)

        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        hx5d =  self.rebnconv5d(concatenate([hx6, hx5], axis=-1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(concatenate([hx5dup, hx4], axis=-1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(concatenate([hx4dup, hx3], axis=-1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis=-1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis=-1))

        return hx1d + hxin
    
    
### RSU-5 ###
class RSU5(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)
        self.pool3 = MaxPooling2D(2, strides=2, padding="same")

        self.rebnconv4 = REBNCONV(mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)

        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(concatenate([hx5, hx4], axis=-1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(concatenate([hx4dup, hx3], axis=-1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis=-1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis=-1))

        return hx1d + hxin
    
    
### RSU-4 ###
class RSU4(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = MaxPooling2D(2, strides=2, padding="same")
        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = MaxPooling2D(2, strides=2, padding="same")

        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)

        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(concatenate([hx4, hx3], axis=-1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis=-1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis=-1))

        return hx1d + hxin

    
### RSU-4F ###
class RSU4F(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=2)

        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(concatenate([hx4, hx3], axis=-1))
        hx2d = self.rebnconv2d(concatenate([hx3d, hx2], axis=-1))

        hx1d = self.rebnconv1d(concatenate([hx2d, hx1], axis=-1))

        return hx1d + hxin
    
    
def U2NET(input_height, input_width, n_classes):
    
    img_input = tf.keras.layers.Input(shape=(input_height, input_width, 3))
    
    hx = img_input
    
    # --------------------- encoder ---------------------
    # stage 1 down
    hx1 = RSU7(32, 64)(hx) 
    hx = MaxPooling2D(2, strides=2, padding="same")(hx1)
    
    # stage 2 down
    hx2 = RSU6(32, 128)(hx) 
    hx = MaxPooling2D(2, strides=2, padding="same")(hx2)
    
    # stage 3 down
    hx3 = RSU5(64, 256)(hx) 
    hx = MaxPooling2D(2, strides=2, padding="same")(hx3)

    # stage 4 down
    hx4 = RSU4(128, 512)(hx) 
    hx = MaxPooling2D(2, strides=2, padding="same")(hx4)
    
    # stage 5 down
    hx5 = RSU4F(256, 512)(hx) 
    hx = MaxPooling2D(2, strides=2, padding="same")(hx5)
    
    # stage 6 up
    hx6 = RSU4F(256, 512)(hx)
    hx6up = _upsample_like(hx6, hx5)
    
    # ---------------------- decoder ---------------------
    # stage 5 up
    hx5d = RSU4F(256, 512)(concatenate([hx6up, hx5], axis=-1))
    hx5dup = _upsample_like(hx5d, hx4)

    # stage 4 up
    hx4d = RSU4(128, 256)(concatenate([hx5dup, hx4], axis=-1))
    hx4dup = _upsample_like(hx4d, hx3)

    # stage 3 up
    hx3d = RSU5(64, 128)(concatenate([hx4dup, hx3], axis=-1))
    hx3dup = _upsample_like(hx3d, hx2)

    # stage 2 up
    hx2d = RSU6(32, 64)(concatenate([hx3dup, hx2], axis=-1))
    hx2dup = _upsample_like(hx2d, hx1)

    # stage 1
    hx1d = RSU7(32, 64)(concatenate([hx2dup, hx1], axis=-1))
    
    # -------------------- side output --------------------
    d1 = Conv2D(n_classes, kernel_size=3, padding="same")(hx1d)

    d2 = Conv2D(n_classes, kernel_size=3, padding="same")(hx2d)
    d2 = _upsample_like(d2, d1)

    d3 = Conv2D(n_classes, kernel_size=3, padding="same")(hx3d)
    d3 = _upsample_like(d3, d1)

    d4 = Conv2D(n_classes, kernel_size=3, padding="same")(hx4d)
    d4 = _upsample_like(d4, d1)

    d5 = Conv2D(n_classes, kernel_size=3, padding="same")(hx5d)
    d5 = _upsample_like(d5, d1)

    d6 = Conv2D(n_classes, kernel_size=3, padding="same")(hx6)
    d6 = _upsample_like(d6, d1)

    d0 = Conv2D(n_classes, kernel_size=1)(concatenate([d1, d2, d3, d4, d5, d6], axis=-1))
   
    out_0 = Activation("softmax", name="d0", dtype="float32")(d0)
    out_1 = Activation("softmax", name="d1", dtype="float32")(d1)
    out_2 = Activation("softmax", name="d2", dtype="float32")(d2)
    out_3 = Activation("softmax", name="d3", dtype="float32")(d3)
    out_4 = Activation("softmax", name="d4", dtype="float32")(d4)
    out_5 = Activation("softmax", name="d5", dtype="float32")(d5)
    out_6 = Activation("softmax", name="d6", dtype="float32")(d6)
    
    return tf.keras.Model(inputs=img_input,  outputs=[out_0, out_1, out_2, out_3, out_4, out_5, out_6], name="U2Net")