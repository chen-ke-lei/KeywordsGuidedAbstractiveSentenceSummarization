import re;
import numpy as np
from bert4keras.tokenizers import Tokenizer

######字典
dic = None
####分词器
tokenizer = None


def initTokenizer(dicPath='../data/dic.txt'):
    tokenizer = Tokenizer(dicPath, do_lower_case=True)
    dic = tokenizer._token_dict
    return tokenizer, dic


def line2WordHandle(line):
    return [x for x in re.split(r'\s+', line)]


def readFile(path, handle):
    lines = []
    if handle is None:
        handle = lambda x: x
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            lines.append(handle(line.strip()))
            line = f.readline()
    return lines


def findCrossWords(sentence, keywords):
    res = []
    sentence = set(sentence)
    for keyword in keywords:
        if keyword in sentence:
            res.append(keyword)
    return res


def readTrainData(sentecePath,
                  keyWordPath,
                  maxlen,
                  sentenceHandel,
                  keyWordHadel,
                  ):
    sentences = readFile(sentecePath, sentenceHandel)
    keyWords = readFile(keyWordPath, keyWordHadel)
    tokenizer, dic = initTokenizer()
    token_ids = []
    segment_ids = []
    if len(sentences) != len(keyWords):
        return
    for i in range(len(sentences)):
        cross = findCrossWords(sentences[i], keyWords[i])
        line = ''
        for x in sentences[i]:
            line += x + " "
        grouth = ''
        for x in cross:
            grouth += x + " "
        token_id, segment_id = tokenizer.encode(line, grouth, maxlen=(maxlen + 1) * 2)
        print(token_id)
        token_ids.append(token_id)
        segment_ids.append(segment_id)


# dict = loadDic()


# print(dict)

readTrainData('../data/train/src/train.src_00'
              , '../data/train/tgt/train.tgt_00'
              , 64
              , line2WordHandle
              , line2WordHandle)


def oneHot(arr, maxLen):
    res = []
    for i in range(len(arr)):
        res.append([])
        for j in arr[i]:
            zero = ceateZero(maxLen)
            zero[j] = 1
            res[i].append(zero)
    return res

    ######创建全0向量


def ceateZero(maxlen):
    a = np.zeros((maxlen,))
    return a
