import tensorflow as tf
import numpy as np


class BasicConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding=0, relu=True, bn=True,data_format='channels_last'):
        super(BasicConv, self).__init__()
        self.filters = filters
        self.conv = tf.keras.layers.Conv2D(filters,kernel_size,strides,padding='valid',data_format=data_format)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if bn else None
        self.relu = relu
        if data_format=='channels_first':
            self.pads = tf.constant([[0,0],[0,0],[padding,padding],[padding,padding]])
        else:
            self.pads = tf.constant([[0,0],[padding,padding],[padding,padding],[0,0]])

    def call(self, x,training=True):
        
        x = tf.pad(x,self.pads)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = tf.nn.relu(x)
        return x



class ChannelGate(tf.keras.layers.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'],name=None,data_format='channels_last'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.data_format = data_format
        
        self.mlp = tf.keras.models.Sequential(
            [tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(gate_channels // reduction_ratio),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(gate_channels)]
            )
        self.pool_types = pool_types

    def call(self, x, training=True):
        channel_att_sum = None
        if self.data_format=='channels_last':
            nn_data_format = 'NHWC'
            axis=(1,2)
            if int(tf.__version__.split('.')[0])==1:
                height, width = x.shape[1].value, x.shape[2].value
            elif int(tf.__version__.split('.')[0])==2:
                height, width = x.shape[1], x.shape[2]
        elif self.data_format=='channels_first':
            nn_data_format = 'NCHW'
            axis=(2,3)
            if int(tf.__version__.split('.')[0])==1:
                height, width = x.shape[2].value, x.shape[3].value
            elif int(tf.__version__.split('.')[0])==2:
                height, width = x.shape[2], x.shape[3]
    

        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = tf.nn.avg_pool(x,(height,width),strides=(height,width),padding='VALID',data_format=nn_data_format)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = tf.nn.max_pool(x,(height,width),strides=(height,width),padding='VALID',data_format=nn_data_format)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = tf.expand_dims(tf.expand_dims(tf.math.sigmoid(channel_att_sum),axis=axis[0]),axis=axis[1])
        
        if scale.dtype!=x.dtype:
            scale = tf.cast(scale,x.dtype)

        return tf.math.multiply(x,scale)


class ChannelPool(tf.keras.layers.Layer):
    def __init__(self,data_format='channels_last'):
        super(ChannelPool, self).__init__()
        self.axis = 1 if data_format=='channels_first' else 3
    def call(self, x,training=True):
        return tf.concat((tf.expand_dims(tf.math.reduce_max(x,self.axis),self.axis),tf.expand_dims(tf.math.reduce_mean(x,self.axis),self.axis)),axis=self.axis)


class SpatialGate(tf.keras.layers.Layer):
    def __init__(self,data_format):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool(data_format)
        self.spatial = BasicConv(1, kernel_size, strides=1, padding=((kernel_size-1)//2), relu=False,data_format=data_format)
    def call(self, x,training=True):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = tf.math.sigmoid(x_out) # broadcasting
        if scale.dtype!=x.dtype:
            scale = tf.cast(scale,x.dtype)
        return tf.math.multiply(x,scale)


class CBAM(tf.keras.layers.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False,data_format='channels_last'):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types,data_format=data_format)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(data_format=data_format)
    def call(self, x,training=True):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out