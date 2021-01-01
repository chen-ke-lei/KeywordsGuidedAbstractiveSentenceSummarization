from base.model import *
from base.dataProcess import *


import sys
import time


####################这个类为整个模型的执行流程 依赖base.model 和 base.dataProcess#######################


def app(inputPath,
        sumPath,
        batchSize=64,
        embeddingSize=300,
        lineNum=-1,
        hiddenSize=512,
        maxLen=64,
        epochs=32
        ):
    saveingPath = '../save/model' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".model"

    stopWordsPath = '../data/stopwords.txt'

    ########################################################################
    print("sentences Path: " + inputPath)
    print("summary Path: " + sumPath)
    print("batchSize: " + str(batchSize))
    print("embeddingSize :  " + str(embeddingSize))
    print("hiddenSize: " + str(hiddenSize))
    print("sentence length: " + str(maxLen))
    print("epochSize: " + str(epochs))
    if lineNum != -1:
        print("max number of lines: " + str(lineNum))
    ##############获得经过处理的np 数组###########################
    libSize, oneHotKeyWords, lines, keyWords = getTrainData(inputPath, sumPath, stopWordsPath, maxLen, lineNum=lineNum)
    print("read training data complete!!!!")
    print("voc length: " + str(libSize))
    ################embedding 获取embeding 结果与词向量矩阵########################
    sentenceIn = Input(shape=(maxLen,), name='sentence')
    keyWordsIn = Input(shape=(maxLen,), name='keyWords')
    sentenceEmbedding, sentenceWeight = embeddingAndWeight(sentenceIn, libSize, embeddingSize)
    keyWordsEmbedding, keyWordsWeight = embeddingAndWeight(keyWordsIn, libSize, embeddingSize)

    ####################双向LSTM######################
    ####1是正向 2 是负向
    sentenceH1, sentenceH2 = biLstm(sentenceEmbedding, maxLen, hiddenSize)
    keyWordsH1, keyWordsH2 = biLstm(keyWordsEmbedding, maxLen, hiddenSize)

    ######################过Cogate 获取encode结果
    sentenceH1Cogate = calEncode([sentenceH1, keyWordsH1], 1)
    sentenceH2Cogate = calEncode([sentenceH2, keyWordsH2], 2)
    keyWordsH1Cogate = calEncode([keyWordsH1, sentenceH1], 1)
    keyWordsH2Cogate = calEncode([keyWordsH2, sentenceH2], 2)

    #######################对两个方向的结果相加合并
    sentenceCogate = Add()([sentenceH1Cogate, sentenceH2Cogate])
    keyWordsCogate = Add()([keyWordsH1Cogate, keyWordsH2Cogate])

    ###########################带context的Attention 获取context Vector  a向量 及 s
    sentenceC, sentenceA, sentenceS = attention(sentenceCogate, maxLen, hiddenSize)
    keyWordsC, keyWordsA, keyWordsS = attention(keyWordsCogate, maxLen, hiddenSize)

    #################通过cogateFusion 对 两个输入的c进行合并
    c = GateFusion(sentenceC, keyWordsC, maxLen)

    ###########简单对a合并并求和
    A = calA(sentenceA, keyWordsA)

    ##生成Y向量 和学长沟通后 y为sentence 的Attention 结果转one hot
    y = genertorY(sentenceS, sentenceWeight, embeddingSize)

    ###############产生概率分布P
    lambdaT = calLambda(c, sentenceS, y)
    pgen = calPgen(c, sentenceS)
    p = calPw(A, lambdaT, pgen)

    ################编译模型 训练
    model = Model(inputs=[sentenceIn, keyWordsIn], outputs=[p])
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    ############ 和学长沟通后 y_label 为 ground-Truth 的one-hot 形式
    model.fit({'sentence': lines, 'keyWords': keyWords}
              , [oneHotKeyWords]
              , batch_size=batchSize
              , epochs=epochs
              , validation_split=0.1
              )

    model.save(saveingPath)


if __name__ == "__main__":
    inputPath = "../data/test.src.txt"
    sumPath = '../data/test.tgt.txt'
    lineNum = -1
    if len(sys.argv) == 2:
        inputPath = sys.argv[0]
        sumPath = sys.argv[1]
    if len(sys.argv) > 2:
        lineNum = int(sys.argv[2])
    app(inputPath, sumPath, lineNum=12)
