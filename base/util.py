import tensorflow as tf
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Input, Activation, Concatenate, Add, Multiply, \
    Lambda
import os
import numpy as np
import keras
from keras import backend as K
from keras.engine.topology import Layer
import re
from keras import Model

from keras import initializers, regularizers, constraints, optimizers, losses

"""
 定义了一些自定义layer 及 通用方法。供base.model 调用
"""


#######张量拆分
def split2(input, x1, y1):
    lam = Lambda(lambda x: tf.reshape(x, x1))(input)
    lam1, lam2 = Lambda(lambda x: tf.split(x, 2, axis=-1))(lam)
    lam1 = Lambda(lambda x: tf.reshape(x, y1))(lam1)
    lam2 = Lambda(lambda x: tf.reshape(x, y1))(lam2)
    return lam1, lam2;


########点乘
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


########Cogate 的实现
class CoGate(Layer):
    def __init__(self, mod=1, **kwargs):
        self.mod = mod
        super(CoGate, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[0][2], input_shape[0][2],),
                                  initializer='uniform',

                                  )

        self.U1 = self.add_weight(name='U1',
                                  shape=(input_shape[0][2], input_shape[0][2],),
                                  initializer='uniform',

                                  )

        super(CoGate, self).build(input_shape)

    def call(self, inputs):
        h1 = inputs[0]
        u1 = inputs[1]
        _h1 = dot_product(h1, self.W1)
        ##todo 这里应该实现一个首尾拼接在repeat
        # if u1.shape[0] != None:
        #     if self.mod == 1:
        #         for i in range(1, u1.shape[0]):
        #             u1[i] = u1[0]
        #     if self.mod == 2:
        #         for i in range(u1.shape[0] - 1):
        #             u1[i] = u1[u1.shape[0] - 1]

        _u1 = dot_product(u1, self.U1)

        a1 = _h1 + _u1
        a1 = K.sigmoid(a1)
        a1 = tf.multiply(h1, a1)
        return a1

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2];


###LSTM attention 的实现
class Attention(Layer):
    # Input shape 3D tensor with shape: `(samples, steps, features)`.
    # Output shape 2D tensor with shape: `(samples, features)`.

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        dim = input_shape[2] // 2

        self.W = self.add_weight(name='W',
                                 shape=(dim, dim),
                                 initializer='uniform',
                                 trainable=True,
                                 )
        self.V = self.add_weight(name='V',
                                 shape=(dim, dim),
                                 initializer='uniform',
                                 trainable=True,
                                 )

        self.features_dim = input_shape[-1] / 2

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        x = tf.reshape(x, [-1, K.int_shape(x)[1], K.int_shape(x)[-1] // 2, 2])

        x1, x2 = tf.split(x, 2, axis=-1)
        x1 = tf.squeeze(x1, [3])
        x2 = tf.squeeze(x2, [3])

        eij = dot_product(x1, self.W) + dot_product(x2, self.V)

        eij = K.tanh(eij)
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True)
                    #   + K.epsilon()
                    ,
                    K.floatx())
        ## a = K.expand_dims(a)
        weighted_input = x1 * a
        c = K.sum(weighted_input, axis=-1)
        a = K.sum(a, axis=-1)
        res = K.concatenate([c, a], axis=-1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


######计算一个概率分布
class Pgen(Layer):
    def __init__(self, **kwargs):
        super(Pgen, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[0][1], input_shape[0][1],),
                                  initializer='uniform',
                                  trainable=True,
                                  )

        self.U1 = self.add_weight(name='U1',
                                  shape=(input_shape[1][2],),
                                  initializer='uniform',
                                  trainable=True,
                                  )
        self.bias = self.add_weight(shape=(input_shape[0][1],),
                                    initializer='uniform',
                                    name='bias',
                                    trainable=True, )

        super(Pgen, self).build(input_shape)

    def call(self, inputs):
        c = inputs[0]
        s = inputs[1]
        c = K.dot(c, self.W1)
        s = dot_product(s, self.U1)
        res = K.bias_add(c + s, self.bias)
        res = K.softmax(res)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1];


####概率分布层
class LambdaT(Layer):
    def __init__(self, **kwargs):
        super(LambdaT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[2][2], 1,),
                                  initializer='uniform',
                                  trainable=True,
                                  )

        self.U1 = self.add_weight(name='U1',
                                  shape=(input_shape[2][2], input_shape[1][2],),
                                  initializer='uniform',
                                  trainable=True,
                                  )
        self.V1 = self.add_weight(name='V1',
                                  shape=(input_shape[2][2], input_shape[2][2],),
                                  initializer='uniform',
                                  trainable=True,
                                  )
        self.bias = self.add_weight(shape=(input_shape[0][1], input_shape[2][2],),
                                    initializer='uniform',
                                    name='bias',
                                    trainable=True, )

        super(LambdaT, self).build(input_shape)

    def call(self, inputs):
        c = inputs[0]
        s = inputs[1]
        y = inputs[2]
        c = K.expand_dims(c, -1);

        s = dot_product(s, self.U1)
        y = dot_product(y, self.V1)
        c = dot_product(c, self.W1)

        res = c + s + y
        res = K.bias_add(res, self.bias)

        res = K.sigmoid(res)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[2][2]


####这里完成的是lanbdaT 与 pgen 对应元素的相乘
def mul(x):
    a = x[0]
    b = x[1]
    b = K.expand_dims(b, -1)
    b = K.repeat_elements(b, a.shape[-1], -1)

    return a * b



####这里是要解决 embedding 向量 转换 one-hot 向量 x2 为词向量矩阵
@tf.function
def mul1(x, x2):
    x2 = tf.convert_to_tensor(x2)
    x = tf.transpose(x, [1, 2, 0])
    res = K.dot(x2, x)

    return tf.transpose(res, [2, 1, 0])
