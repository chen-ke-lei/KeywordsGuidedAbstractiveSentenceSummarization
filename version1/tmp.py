import os;


def spliteTrainDataFile(basePath):
    print(basePath)
    if not os.path.exists('../data/train'):
        os.makedirs('../data/train')
    outSrcPath = '../data/train/' + basePath
    i = 0
    wf = None
    with open('../data/train.' + basePath + '.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            print(i)
            if i % 10000 == 0:
                if wf is not None:
                    wf.close()
                print("已完成" + str(i % 10000) + '个文件')
                wf = open('../data/train/' + basePath + '_' + str(i % 10000) + '.txt', 'w', encoding='utf-8')
            wf.write(line + '\n')
            line = f.readline()
            i += 1


def createTrainSubData():
    spliteTrainDataFile('src')
    spliteTrainDataFile('tgt')


createTrainSubData()
