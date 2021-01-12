import json
import re
from bert4keras.tokenizers import Tokenizer, load_vocab
import numpy as np
import pickle


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


def repliceSpaceHadle(line):
    line = line.replace(' ', '')
    return line


def splitHadle(line):
    return line.split()


def JsonHandleBulider(attr, type):
    if type == 'str':
        def JsonHandle(line):
            jsonData = json.loads(line)
            return repliceSpaceHadle(jsonData[attr])
    elif type == 'int':
        def JsonHandle(line):
            jsonData = json.loads(line)
            return jsonData[attr]
    elif type == 'arr':
        def JsonHandle(line):
            jsonData = json.loads(line)
            return splitHadle(jsonData[attr])
    return JsonHandle


def loadLawData(path):
    factList = readFile(path, JsonHandleBulider('fact_cut', 'str'))
    accuList = readFile(path, JsonHandleBulider('accu', 'int'))
    lawList = readFile(path, JsonHandleBulider('law', 'int'))
    termList = readFile(path, JsonHandleBulider('term', 'arr'))
    return factList, np.array(accuList), np.array(lawList), np.array(termList)


def loadDic(dicPath):
    token_dict, keep_tokens = load_vocab(
        dict_path=dicPath,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)
    return token_dict, keep_tokens, tokenizer


def data2index(tokenizer, maxLen, txt):
    token_ids, segment_ids = [], []
    for x in txt:
        token_id, segment_id = tokenizer.encode(
            x, maxlen=maxLen
        )
        if (len(token_id) < maxLen):
            sub = maxLen - len(token_id)
            token_id.extend([0] * sub)
            segment_id.extend([0] * sub)
        token_ids.append(np.array(token_id))
        segment_ids.append(np.array(segment_id))
    # print(len(token_ids))
    # print(len(segment_ids))
    return np.array(token_ids), np.array(segment_ids)


# loadLawData('../law_data/test_cs.json')
def tranferKeyWord(keywords, maxlen):
    res = []
    for x in keywords:
        sentece = []
        for i in range(maxlen):
            sentece.append([1, 0])
        for y in x:
            if y[1] >= maxlen:
                break
            start = y[0]
            end = y[1]
            while start <= end:
                sentece[start][1] = 1
                sentece[start][0] = 0
                start += 1

        res.append(np.array(sentece))
    return np.array(res)


def loadLawDataContainKeyWordsAndSummary(path, maxlen):
    factList, accuList, lawList, termList = loadLawData(path)
    summaryList = readFile(path, JsonHandleBulider('summary', 'str'))
    keyWordsList = tranferKeyWord(readFile(path, JsonHandleBulider('keywords_index', 'int')),
                                  maxlen)
    return factList, summaryList, accuList, lawList, termList, keyWordsList


# list = readFile('../law_data/law_mark.txt', JsonHandleBulider('keywords_index', 'arr'))
# list = tranferKeyWord(list, 512)

def loadDic(dicPath="../law_data/w2id_thulac.pkl"):
    fp = open(dicPath, "rb+")
    dic = pickle.load(fp)
    ###最后一个是padding
    ###倒数第二个是unk
    return dic


def dataPadding(dic, maxlen, factList):
    res = []
    pad = len(dic) - 1
    for line in factList:
        tmp = []
        i = 0
        for word in line:
            if i >= maxlen:
                break
            i += 1
            if word in dic:
                tmp.append(dic[word])
            else:
                tmp.append(len(dic) - 2)
        if len(tmp) < maxlen:
            tmp.extend([pad] * (maxlen - len(tmp)))
        res.append(tmp)
    res = np.array(res)
    return res


def loadBaseData(path
                 , dic
                 , maxlen=512):
    factList = readFile(path, JsonHandleBulider('fact_cut', 'arr'))
    accuList = readFile(path, JsonHandleBulider('accu', 'int'))
    lawList = readFile(path, JsonHandleBulider('law', 'int'))
    termList = readFile(path, JsonHandleBulider('term', 'int'))

    factList = dataPadding(dic, maxlen, factList)
    return factList, np.array(accuList), np.array(lawList), np.array(termList)


def loadEmbeddingWeight(path='../law_data/cail_thulac.npy'):
    a = np.load(path)
    print(a.shape)


# loadEmbeddingWeight()
##loadBaseData('../law_data/train_cs.json')

dic = loadDic()

loadBaseData('../law_data/test_cs.json', dic)
