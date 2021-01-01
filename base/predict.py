from keras.models import load_model
from base.run import *
from base.util import *


def load():
    model = load_model('../save/model.h5', custom_objects={'CoGate': CoGate
        , 'Attention': Attention
        , 'mul1': mul1
        , 'mul': mul
        , 'Pgen': Pgen
        , 'LambdaT': LambdaT})
    lines, summary, keyWords, dict, index = getInputData('../data/test.src.txt'
                                                         , '../data/test.tgt.txt'
                                                         , '../data/stopwords.txt'
                                                         , 64
                                                         , lineNum=1000)
    libSize, oneHotKeyWords, lines, keyWords = getTrainData('../data/dev.src.txt'
                                                            , '../data/dev.tgt.txt'
                                                            , '../data/stopwords.txt'
                                                            , 64
                                                            , lineNum=100)
    pre = model.predict([lines, keyWords])
    print(pre)


load()
