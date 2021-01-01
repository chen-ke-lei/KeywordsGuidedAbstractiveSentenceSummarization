from version1.dataProcess import *
from version1.model import *


def load(maxLen, libSize, embeddingSize, hiddenSize, weightPath='../save/model.w'):
    # model = load_model('../save/model.h5', custom_objects={'CoGate': CoGate
    #     , 'Attention': Attention
    #     , 'mul1': mul1
    #     , 'mul': mul
    #     , 'Pgen': Pgen
    #     , 'LambdaT': LambdaT, 'tf': tf})
    model = createModel(maxLen, libSize, embeddingSize, hiddenSize)
    model.load_weights(weightPath)
    dict = loadDic()
    stopWords = laodStopWords('../data/stopwords.txt')
    return model, dict, stopWords


stopWords = []
model = None;
model, dict, stopWords = load()
dict2ID = index2dic(dict)
inputShape = 64


def predict(str):
    line = line2Word(stopWords, str)
    lines = [];
    lines.append(dict['#START#'])
    wordNum = 1
    for x in line:
        if wordNum >= inputShape - 1:
            break
        if x in dict:
            lines.append(dict[x])
        else:
            lines.append(dict['#UK#'])
        wordNum += 1
    lines.append(dict['#END#'])
    wordNum += 1
    while wordNum < inputShape:
        lines.append(0)
        wordNum += 1
    lines = np.array([lines])
    key = np.zeros((1, inputShape))
    res = ''
    end = False
    i = 0
    while not end:
        pre = model.predict([lines, key])
        x = np.argmax(pre[0][i])
        print(pre.shape)
        print(x)
        print(pre)
        key[0][i] = x
        print(key.shape)
        if x == 0:
            break
        word = dict2ID[x]
        res += (' ' + word)
        i += 1
        if i >= inputShape or word == '#END#':
            end = True
    print(res)
