import json
from bert4keras.tokenizers import Tokenizer, load_vocab
import numpy as np
import pickle
import jieba


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

def loadDicByPKL(dicPath="../law_data/w2id_thulac.pkl"):
    fp = open(dicPath, "rb+")
    dic = pickle.load(fp)
    ###最后一个是padding
    ###倒数第二个是unk
    ###57543 EOS
    ###83367 BOS
    return dic


def dataPadding(dic,
                maxlen,
                factList,
                startToken=None,
                endToken=None):
    res = []
    seq = maxlen
    if startToken is not None:
        seq -= 1
    if endToken is not None:
        seq -= 1
    pad = len(dic) - 1
    for line in factList:
        tmp = []
        if startToken is not None:
            tmp.append(dic[startToken])
        i = 0
        for word in line:
            if i >= seq:
                break
            i += 1
            if word in dic:
                tmp.append(dic[word])
            else:
                tmp.append(len(dic) - 2)
        if endToken is not None:
            tmp.append(dic[endToken])
        if len(tmp) < maxlen:
            tmp.extend([pad] * (maxlen - len(tmp)))
        res.append(tmp)
    res = np.array(res)
    return res


def loadBaseDataByWords(path
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
    return a


def str2Indes(dic, str):
    if str in dic:
        return dic[str]
    else:
        return len(dic) - 2


def cutOneLine(factStr, keyWordArray, maxlen, dic):
    start = 0
    length = 0
    fact = []
    keyWords = []
    for keyWord in keyWordArray:
        if keyWord[0] > start:
            tmp = jieba.lcut(factStr[start:keyWord[0]])
            # print(tmp)
            for x in tmp:
                if length >= maxlen:
                    return fact, keyWords
                length += 1
                fact.append(str2Indes(dic, x))
                keyWords.append(np.array([0, 1]))
        if length >= maxlen:
            return fact, keyWords
        length += 1
        # print(factStr[keyWord[0]:keyWord[1]])
        fact.append(str2Indes(dic, factStr[keyWord[0]:keyWord[1]]))
        start = keyWord[1]
        keyWords.append(np.array([1, 0]))
    if start < len(factStr):
        tmp = jieba.lcut(factStr[start:len(factStr)])
        # print(tmp)
        for x in tmp:
            if length >= maxlen:
                return fact, keyWords
            length += 1
            fact.append(str2Indes(dic, x))
            keyWords.append(np.array([0, 1]))
    if len(fact) < maxlen:
        sub = maxlen - len(fact)
        fact.extend([len(fact) - 1] * sub)
        keyWords.extend([np.array([0, 1])] * sub)
    return np.array(fact), np.array(keyWords)


def cutWordsByKeyWords(factList, keyWordsList, maxlen, dic):
    if len(factList) != len(keyWordsList):
        raise Exception("数据长度不正确")
    factsIndex = []
    keyWordsIndex = []
    for i in range(len(factList)):
        factStr = factList[i]
        keyWordArray = keyWordsList[i]
        fact, keyWords = cutOneLine(factStr, keyWordArray, maxlen, dic)
        factsIndex.append(fact)
        keyWordsIndex.append(keyWords)
    return np.array(factsIndex), np.array(keyWordsIndex)


def loadAllDataByWords(path
                       , dic
                       , maxlen=512
                       , summaryLen=64):
    accuList = np.array(readFile(path, JsonHandleBulider('accu', 'int')))
    lawList = np.array(readFile(path, JsonHandleBulider('law', 'int')))
    termList = np.array(readFile(path, JsonHandleBulider('term', 'int')))
    factList = readFile(path, JsonHandleBulider('fact_cut', 'str'))
    summaryList = readFile(path, JsonHandleBulider('summary', 'str'))
    keyWordsList = readFile(path, JsonHandleBulider('keywords_index', 'int')),
    summaryListInput = dataPadding(dic, summaryLen, summaryList, startToken='BOS')
    summaryListOutput = dataPadding(dic, summaryLen, summaryList, endToken='EOS')
    factList, keyWordsList = cutWordsByKeyWords(factList, keyWordsList[0], maxlen, dic)
    return accuList, lawList, termList, summaryListInput, summaryListOutput, factList, keyWordsList


jieba.load_userdict(loadDicByPKL())
dict = loadDicByPKL()
loadAllDataByWords('../law_data/law_mark_new.json', dict)
