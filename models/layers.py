import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer

class MaxPoolingWithArgmax2D(Layer):
    
    def __init__(self, pool_size=(2,2), strides=(2,2), padding='same',**kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides
            
    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides

        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        
        output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        
        argmax = K.cast(argmax, K.floatx()) # this might be important. What does k.floatx do?
        return [output, argmax]
    
    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]
    
    def compute_mask(self, inputs, mask=None):
        return 2 * [None]
    

class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size
        
    def call(self, inputs, output_shape=None):
        updates = inputs[0]
        mask = tf.cast(inputs[1], dtype=tf.int64) # IMPORTANT: WHAT IS AN INT64 DOING HERE?
        
        input_shape = tf.shape(updates, out_type=tf.int64)
        # now calculate the new shape
        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1]*self.size[0],
                input_shape[2]*self.size[1],
                input_shape[3])
            
        # calculation indices for batch, height, width and feature map
        one_like_mask = tf.ones_like(mask, dtype=tf.int64)
        batch_range   = tf.reshape(
                            tf.range(
                                tf.cast(output_shape[0], dtype=tf.int64),
                                dtype=mask.dtype),
                            shape=[input_shape[0], 1, 1, 1])
        
        b = one_like_mask * batch_range
        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
        
        updates_ = tf.reshape(updates, [flat_input_size])
        
        b1 = tf.reshape(b, [flat_input_size, 1])
        mask_ = tf.reshape(mask, [flat_input_size, 1])
        mask_ = tf.concat([b1, mask_], 1)
        
        ret = tf.scatter_nd(mask_, updates_, shape=tf.cast(flat_output_shape, tf.int64)) # WARNING: INT 64
        ret = tf.reshape(ret, output_shape)
        
        set_input_shape = updates.get_shape()
        set_output_shape = [set_input_shape[0],
                            set_input_shape[1] * self.size[0],
                            set_input_shape[2] * self.size[1],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret
    
    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1] * self.size[0],
                mask_shape[2] * self.size[1],
                mask_shape[3]
                )
