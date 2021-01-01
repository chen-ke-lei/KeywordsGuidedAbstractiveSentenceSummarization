from base.util import *

"""
这个文件是数据处理 从加载数据到将数据转化成可以被训练的np数组
"""


####普通读文件
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
    return set(readFile(path));


#####对数据做padding 截断或补0
def padding(array, maxlen):
    if len(array) == maxlen:
        return array
    if (len(array) > maxlen):
        return array[0:maxlen - 1];
    while (len(array) < maxlen):
        array.append(0)
    return array


####加载文件返回python数组

def getInputData(inputPath, summaryPath, stopWordsPath, maxlen, lineNum=-1):
    """

    :param inputPath:
    :param summaryPath:
    :param stopWordsPath:
    :param maxlen:
    :param lineNum: -1 为取所有数据 或者可以取指定行
    :return:
    """

    baseLines = readFile(inputPath)
    baseSum = readFile(summaryPath)
    stopWord = laodStopWords(stopWordsPath)
    lines = []
    summary = []
    keyWords = []
    dict = {}
    index = 1
    for i in range(len(baseLines)):
        if lineNum >= 0 and i > lineNum:
            break
        line = re.sub(r"[^a-zA-Z\- ]", "", baseLines[i])
        sum = re.sub(r"[^a-zA-Z\- ]", "", baseSum[i])
        lineWords = [x for x in re.split(r'\s+', line) if x not in stopWord]
        sumWords = [x for x in re.split(r'\s+', sum) if x not in stopWord]
        line2Id = []
        sum2Id = []

        keyWord2ID = []
        for x in lineWords:
            if x not in dict:
                dict[x] = index
                index += 1
            line2Id.append(dict[x])
        lineWords = set(lineWords)
        for x in sumWords:
            if x not in dict:
                dict[x] = index
                index += 1
            sum2Id.append(dict[x])
            if x in lineWords:
                keyWord2ID.append(dict[x])
        keyWords.append(padding(keyWord2ID, maxlen))
        lines.append(padding(line2Id, maxlen))
        summary.append(padding(sum2Id, maxlen))

    return lines, summary, keyWords, dict, index + 1


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


############获取训练数据
def getTrainData(inputPath, sumPath, stopWordsPath, maxlen, lineNum):
    lines, summary, keyWords, dict, dictLength = getInputData(inputPath, sumPath, stopWordsPath, maxlen,
                                                              lineNum=lineNum)
    oneHotKeyWords = oneHot(keyWords, dictLength)
    lines = np.array(lines)
    oneHotKeyWords = np.array(oneHotKeyWords)
    keyWords = np.array(keyWords)
    return dictLength, oneHotKeyWords, lines, keyWords
