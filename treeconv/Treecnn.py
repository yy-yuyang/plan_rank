import numpy as np
import tensorflow as tf


class BinaryTreeConv(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()

        self.__out_channels = out_channels
        self.conv = tf.keras.layers.Conv1D(filters=self.__out_channels, padding='same', strides=3, kernel_size=3)

    def call(self, input, idxes):

        if idxes != None:
            idxes = tf.cast(idxes, dtype=tf.int32)
            expanded = tf.gather_nd(params=input, indices=idxes, batch_dims=2)
            results = self.conv(expanded)
        else:
            results = self.conv(input)
        return (results, None)

class TreeActivation(tf.keras.layers.Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def call(self, input):
        return self.activation(input)

class TreeLayerNorm(tf.keras.layers.Layer):
    def call(self, input):
        data = input
        # 计算均值和标准差
        mean = tf.reduce_mean(data, axis=(2, 3), keepdims=True)
        std = tf.math.reduce_std(data, axis=(2, 3), keepdims=True)
        # 标准化数据
        epsilon = 1e-5  # 用于防止除零错误的小值
        normd = (data - mean) / (std + epsilon)

        return normd
    
class DynamicPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        # print(inputs)
        max_values = tf.reduce_max(inputs, axis=3)
        # print(max_values)
        return max_values

class TreeCNN(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # self.input_feature_dim = input_feature_dim
        self.BiTree1 = BinaryTreeConv(512)
        self.BiTree2 = BinaryTreeConv(256)
        self.BiTree3 = BinaryTreeConv(128)
        self.Norm = TreeLayerNorm()
        self.Pool = DynamicPooling()
        self.activation = TreeActivation(tf.keras.activations.relu)
        self.Layer = tf.keras.layers.Dense(128)

    def call(self, input, idx):
        input, idx = self.BiTree1(input, idx)
        input = self.Norm(input)
        input = self.activation(input)
        input, idx = self.BiTree2(input, idx)
        input = self.Norm(input)
        input = self.activation(input)
        input, idx = self.BiTree3(input, idx)
        input = self.Norm(input)
        input = self.Pool(input)
        output = self.Layer(input)
        return output