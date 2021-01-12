import re;
import numpy as np
from bert4keras.tokenizers import Tokenizer, load_vocab
import json

######字典
dic = None
####分词器
tokenizer = None


def englishToken(word):
    for x in word:
        if ord(x) >= 128:
            return False
    return True


def EnglishDicHandle(dic, dicLenth=0):
    index = 0
    res = {}
    for key in dic:
        if englishToken(key):
            res[key] = index
            index += 1
        if dicLenth != 0 and dicLenth <= index:
            break
    return res


def initTokenizer(dicPath='../data/dic.txt', diclenth=1000, handle=EnglishDicHandle):
    token_dict, keep_tokens = load_vocab(
        dict_path=dicPath,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    token_dict = handle(token_dict, diclenth)
    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    return tokenizer, token_dict


def line2WordHandle(line):
    return [x for x in re.split(r'\s+', line)]


def factJsonHandle(line):
    jsonData = json.loads(line)
    return repliceSpaceHadle(jsonData['fact_cut'])


def repliceSpaceHadle(line):
    line = re.sub(r" ", "", line)
    return line;


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


def oneHot(arr, maxLen):
    a = np.array(arr)
    res = np.zeros((len(arr), maxLen))
    res[:, a] = 1
    return res

    ######创建全0向量


def readTrainData(sentecePath,
                  keyWordPath,
                  maxlen,
                  tokenizer,
                  dic,
                  sentenceHandel=line2WordHandle,
                  keyWordHadel=line2WordHandle,
                  ):
    sentences = readFile(sentecePath, sentenceHandel)
    keyWords = readFile(keyWordPath, keyWordHadel)

    token_ids = []
    segment_ids = []
    one_Hot = []
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
        token_id, segment_id = tokenizer.encode(line, grouth, maxlen=(maxlen) * 2)
        if len(token_id) < (maxlen) * 2:
            token_id.extend([0] * ((maxlen) * 2 - len(token_id)))
            segment_id.extend([1] * ((maxlen) * 2 - len(segment_id)))

        base, _ = tokenizer.encode(grouth, maxlen=maxlen)
        if len(base) < (maxlen):
            base.extend([0] * (maxlen - len(base)))
        base = base[1:];

        base.extend([0] * (maxlen + 1))

        # print(len(base))
        one_Hot.append(oneHot(base, len(dic)))
        token_ids.append(token_id)
        segment_ids.append(segment_id)
    token_ids = np.array(token_ids)
    segment_ids = np.array(segment_ids)
    one_Hot = np.array(one_Hot)
    return token_ids, segment_ids, one_Hot


def loadLawData(path):
    list = readFile(path, factJsonHandle)
    return list
