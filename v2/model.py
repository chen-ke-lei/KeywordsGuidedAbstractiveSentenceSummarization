from v2.util import *
from bert4keras.models import build_transformer_model

"""
 这个模块 主要是定义了一些模型的组件，组件的调用见base.run 具体的功能见base.run
"""


def embeddingAndWeight(input, libSize, embeddingSize, weights=None):
    if weights is None:
        embedding = Embedding(libSize, embeddingSize)(input)
    else:
        embedding = Embedding(libSize, embeddingSize, weights=[weights])(input)
    model = Model(inputs=[input], outputs=[embedding])
    weight = model.layers[1].get_weights()[0]
    return embedding, weight


def biLstm(input, lenth, hiddenSize):
    # btext = Bidirectional(LSTM(hiddenSize))(input)
    # model = Model(inputs=[input], outputs=[btext])
    # model.summary()

    bi = Bidirectional(LSTM(hiddenSize, return_sequences=True))(input)
    return split2(bi, [-1, lenth, hiddenSize, 2], [-1, lenth, hiddenSize])


def calEncode(input, mod):
    return CoGate(mod=mod)(input)


def attention(input, hiddenSize):
    s = LSTM(hiddenSize, return_sequences=True)(input)
    attention = Attention()([input, s])
    a, c = split2(attention, [-1, -1, 2], [-1, -1])
    return a, c, s


def GateFusion(cr, ck, hiddenSize):
    cr=Lambda(lambda x:x[0:hiddenSize])(cr)
    ck = Lambda(lambda x: x[0:hiddenSize])(ck)
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


def genertorY(s, w, embedingSize):
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


def calS0(input1, input2, hiddenSize):
    h0 = Lambda(lambda x: tf.split(x, K.int_shape(x)[1], axis=1)[0])(input1)
    hn = Lambda(lambda x: tf.split(x, K.int_shape(x)[1], axis=1)[-1])(input2)
    s0 = Add()([h0, hn])
    s0 = Dense(hiddenSize)(s0)
    s0 = Activation('tanh')(s0)
    return s0


def calS(s, s0):
    s = Lambda(lambda x: tf.reshape(x, [K.int_shape(x)[1], -1, K.int_shape(x)[2]]))(s)
    s = Lambda(lambda x: x[0: -1])(s)
    s0 = Lambda(lambda x: tf.reshape(x, [K.int_shape(x)[1], -1, K.int_shape(x)[2]]))(s0)
    s = Lambda(lambda x: tf.concat(x, axis=0))([s0, s])
    s = Lambda(lambda x: tf.reshape(x, [-1, K.int_shape(x)[0], K.int_shape(x)[2]]))(s)
    return s


def createModel(maxLen, libSize, embeddingSize, hiddenSize):
    ################embedding 获取embeding 结果与词向量矩阵########################
    sentenceIn = Input(shape=(maxLen,), name='sentence')
    keyWordsIn = Input(shape=(maxLen,), name='keyWords')
    sentenceEmbedding, sentenceWeight = embeddingAndWeight(sentenceIn, libSize, embeddingSize)
    keyWordsEmbedding, keyWordsWeight = embeddingAndWeight(keyWordsIn, libSize, embeddingSize, weights=sentenceWeight)

    ####################双向LSTM######################
    ####1是正向 2 是负向
    sentenceH1, sentenceH2 = biLstm(sentenceEmbedding, maxLen, hiddenSize)
    keyWordsH1, keyWordsH2 = biLstm(keyWordsEmbedding, maxLen, hiddenSize)

    ######################过Cogate 获取encode结果
    sentenceH1Cogate = calEncode([sentenceH1, keyWordsH1], 1)
    sentenceH2Cogate = calEncode([sentenceH2, keyWordsH2], 2)
    keyWordsH1Cogate = calEncode([keyWordsH1, sentenceH1], 1)
    keyWordsH2Cogate = calEncode([keyWordsH2, sentenceH2], 2)

    ############生成s0####################
    sentenceS0 = calS0(sentenceH1, sentenceH2, hiddenSize)
    keyWordsS0 = calS0(keyWordsH1, keyWordsH2, hiddenSize)

    #######################对两个方向的结果相加合并
    sentenceCogate = Add()([sentenceH1Cogate, sentenceH2Cogate])
    keyWordsCogate = Add()([keyWordsH1Cogate, keyWordsH2Cogate])

    sentenceCogate = calS(sentenceCogate, sentenceS0)
    keyWordsCogate = calS(keyWordsCogate, keyWordsS0)

    # test([sentenceIn, keyWordsIn], [sentenceCogate, keyWordsCogate])
    ###########################带context的Attention 获取context Vector  a向量 及 s
    sentenceC, sentenceA, sentenceS = attention(sentenceCogate, maxLen, hiddenSize)
    keyWordsC, keyWordsA, keyWordsS = attention(keyWordsCogate, maxLen, hiddenSize)

    #################通过cogateFusion 对 两个输入的c进行合并
    c = GateFusion(sentenceC, keyWordsC, maxLen)

    ###########简单对a合并并求和
    A = calA(sentenceA, keyWordsA)

    ##生成Y向量  y为sentence 的Attention 结果转one hot
    y = genertorY(sentenceS, keyWordsWeight, embeddingSize)

    ###############产生概率分布P
    lambdaT = calLambda(c, sentenceS, y)
    pgen = calPgen(c, sentenceS)
    p = calPw(A, lambdaT, pgen)

    ################编译模型 训练
    model = Model(inputs=[sentenceIn, keyWordsIn], outputs=[p])
    return model


def loadBertModl(basePath):
    config_path = basePath + '/bert_config.json'
    checkpoint_path = basePath + '/bert_model.ckpt'
    model = build_transformer_model(config_path, checkpoint_path)
    # model.summary()
    # print(model.layers)

    input = model.input
    output = model.output
    output = Activation('softmax')(output)
    hiddenState = model.layers[-2].output
    a, c, s = attention(output, 128)
    print(hiddenState)
    a1, c1, s1 = attention(hiddenState, 128)
    print(K.int_shape(c1))
    c = GateFusion(c1, c, 64)
    # pgen = calPgen(c, s)
    test(input, [c])


loadBertModl('./model/uncased_L-2_H-128_A-2')
