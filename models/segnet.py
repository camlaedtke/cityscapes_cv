import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from models.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def segnet(input_height, input_width, pool_size=(2,2), n_classes = 3):
    
    img_input = tf.keras.layers.Input(shape=(input_height, input_width, 3))

    # -------------------------- VGG Encoder 1 --------------------------
    c1 = Conv2D(64, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(img_input)
    c1 = Conv2D(64, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c1)
    p1, p1_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c1)
    # -------------------------- VGG Encoder 2 --------------------------
    c2 = Conv2D(128, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(p1)
    c2 = Conv2D(128, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c2)
    p2, p2_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c2)
    # -------------------------- VGG Encoder 3 --------------------------
    c3 = Conv2D(256, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(p2)
    c3 = Conv2D(256, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c3)
    c3 = Conv2D(256, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c3)
    p3, p3_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c3)
    # -------------------------- VGG Encoder 4 --------------------------
    c4 = Conv2D(512, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(p3)
    c4 = Conv2D(512, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c4)
    c4 = Conv2D(512, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c4)
    p4, p4_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c4)

    
    # ---------------------- Maxpool Index Decoder 1 --------------------
    u3 = MaxUnpooling2D(pool_size, dtype="float32")([p4, p4_ind])
    c6 = Conv2D(512, 3, padding='same', kernel_initializer = 'he_normal')(u3)
    c6 = BatchNormalization()(c6)
    c6 = Activation("selu")(c6)
    c6 = Conv2D(512, 3, padding='same', kernel_initializer = 'he_normal')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation("selu")(c6)
    c6 = Conv2D(256, 3, padding='same', kernel_initializer = 'he_normal')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation("selu")(c6)
    # ---------------------- Maxpool Index Decoder 2 --------------------
    u4 = MaxUnpooling2D(pool_size, dtype="float32")([c6, p3_ind])
    c7 = Conv2D(256, 3, padding='same', kernel_initializer = 'he_normal')(u4)
    c7 = BatchNormalization()(c7)
    c7 = Activation("selu")(c7)
    c7 = Conv2D(256, 3, padding='same', kernel_initializer = 'he_normal')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation("selu")(c7)
    c7 = Conv2D(128, 3, padding='same', kernel_initializer = 'he_normal')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation("selu")(c7)
    # ---------------------- Maxpool Index Decoder 3 --------------------
    u5 = MaxUnpooling2D(pool_size, dtype="float32")([c7, p2_ind])
    c8 = Conv2D(128, 3, padding='same', kernel_initializer = 'he_normal')(u5)
    c8 = BatchNormalization()(c8)
    c8 = Activation("selu")(c8)
    c8 = Conv2D(64, 3, padding='same', kernel_initializer = 'he_normal')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation("selu")(c8)
    # ---------------------- Maxpool Index Decoder 4 --------------------
    u6 = MaxUnpooling2D(pool_size, dtype="float32")([c8, p1_ind])
    c9 = Conv2D(64, 3, padding='same', kernel_initializer = 'he_normal')(u6)
    c9 = BatchNormalization()(c9)
    c9 = Activation("selu")(c9)
    c9 = Conv2D(64, 3, padding='same', kernel_initializer = 'he_normal')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation("selu")(c9)
    c9 = Conv2D(n_classes, 3, padding='same')(c9)
    
    output = Activation("softmax", dtype='float32')(c9)
    
    return tf.keras.Model(inputs=img_input, outputs=output, name="segnet")   



