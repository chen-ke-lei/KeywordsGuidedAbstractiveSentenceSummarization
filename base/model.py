from base.util import *

"""
 这个模块 主要是定义了一些模型的组件，组件的调用见base.run 具体的功能见base.run
"""


def embeddingAndWeight(input, libSize, embeddingSize):
    embedding = Embedding(libSize, embeddingSize)(input)
    model = Model(inputs=[input], outputs=[embedding])
    weight = model.layers[1].get_weights()[0]
    return embedding, weight


def biLstm(input, lenth, hiddenSize):
    bi = Bidirectional(LSTM(hiddenSize, return_sequences=True))(input)
    return split2(bi, [-1, lenth, hiddenSize, 2], [-1, lenth, hiddenSize])


def calEncode(input, mod):
    return CoGate(mod=mod)(input)


def attention(input, maxlen, hiddenSize):
    s = LSTM(hiddenSize, return_sequences=True)(input)
    attention = Concatenate(axis=-1)([input, s])
    attention = Attention()(attention)
    c, a = split2(attention, [-1, maxlen, 2], [-1, maxlen])
    return c, a, s


def GateFusion(cr, ck, hiddenSize):
    gt = Activation("sigmoid")(Add()([Dense(hiddenSize)(cr), Dense(hiddenSize)(ck)]))
    gt_1 = Lambda(lambda x: 1 - x)(gt)
    return Add()([Multiply()([gt, cr]), Multiply()([gt_1, ck])]);


def calPgen(c, s):
    return Pgen()([c, s])


def calA(a, b):
    # a = Lambda(lambda x: K.sum(x, axis=-1))(a)
    # b = Lambda(lambda x: K.sum(x, axis=-1))(b)
    return Lambda(lambda x: 0.5 * x)(Add()([a, b]))


def test(input, output):
    model = Model(input=input, output=output);
    model.summary();


def genertorY(s, w,embedingSize):
    s = Dense(embedingSize)(s)
    return Lambda(mul1, arguments={'x2': w})(s)


def calLambda(c, s, y):
    return LambdaT()([c, s, y])


def calPw(A, lambdaT, pgen):
    lambdaT_1 = Lambda(lambda x: 1 - x)(lambdaT)
    a = Lambda(mul)([lambdaT, pgen])
    b = Lambda(mul)([lambdaT_1, A])
    return Add()([a, b]);


def compile(input, output):
    model = Model(input=input, output=output)
    model.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=[-1.0, 1.0]),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    return model

