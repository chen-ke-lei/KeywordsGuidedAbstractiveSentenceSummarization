import re;
import numpy as np


def readFile(path):
    lines = []
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            lines.append(line.strip())
            line = f.readline()
    return lines


##加载停用词表
def laodStopWords(path):
    return set(readFile(path))


def line2Word(stopWords, line):
    line = re.sub(r"[^a-zA-Z\- ]", "", line)
    return [x for x in re.split(r'\s+', line) if x not in stopWords and x != 'UNK' and x != '']


def writeLine(f, str):
    f.write(str)
    f.write('\n')


def createDic(filePath, dicpath):
    stopWordPath = '../data/stopwords.txt'
    stop = laodStopWords(stopWordPath);
    dict = {}
    for path in filePath:
        with open(path, encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line2Word(stop, line)
                for x in line:
                    if x not in dict:
                        dict[x] = 1
                    else:
                        dict[x] += 1
                line = f.readline()
    dictList = []
    for x in dict:
        dictList.append((x, dict[x]))
    dictList.sort(key=lambda b: -b[1])
    with open(dicpath, 'w', encoding='utf-8') as f:
        writeLine(f, '#UK#')
        writeLine(f, '#START#')
        writeLine(f, '#END#')
        for x in dictList:
            writeLine(f, x[0])


# createDic(['../data/train.src.txt', '../data/train.tgt.txt', '../data/test.src.txt', '../data/test.tgt.txt',
#            '../data/dev.src.txt', '../data/dev.tgt.txt'],
#           '../data/dic.txt')
# createDic(['../data/train.src.txt', '../data/train.tgt.txt'], '../data/dic.txt')

def loadDic(path='../data/dic.txt', length=10000):
    dict = {}
    index = 1
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line and index < length:
            dict[re.sub(r"[^a-zA-Z\-# ]", "", line)] = index
            index += 1
            line = f.readline()
    return dict


# dict = loadDic()
# print(dict)


def readInputFile(path
                  , dict
                  , stopWords
                  , inputShape=64
                  , lineNum=-1):
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        index = 0
        res = []
        while line and (lineNum == -1 or index < lineNum):
            line = line2Word(stopWords, line)
            wordNum = 1;
            lines = [];
            lines.append(dict['#START#'])
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
            res.append(lines)
            line = f.readline()
            index += 1
    return res

    ####转化one-hot 向量


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


def getTrainData(sentencePath
                 , keyWordPath
                 , dicPath='../data/dic.txt'
                 , stopWordsPath='../data/stopwords.txt'
                 , dicLength=3000
                 , inputShape=64
                 , lineNum=-1):
    dict = loadDic(path=dicPath, length=dicLength)
    stopWords = laodStopWords(stopWordsPath)
    sentence = readInputFile(sentencePath
                             , dict
                             , stopWords
                             , inputShape=inputShape
                             , lineNum=lineNum)
    keyWords = readInputFile(keyWordPath
                             , dict
                             , stopWords
                             , inputShape=inputShape
                             , lineNum=lineNum)
    sentence = np.array(sentence)
    keyWords = np.array(keyWords)
    label = oneHot(keyWords, len(dict) + 1)
    return sentence, keyWords, label, dict


def index2dic(dic):
    indexDic = {}
    for key in dic:
        indexDic[dic[key]] = key
    return indexDic
