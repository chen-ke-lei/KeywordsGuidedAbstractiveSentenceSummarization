from v2.model import *
from v2.dataProcess import *
import gc


####################这个类为整个模型的执行流程 依赖base.model 和 base.dataProcess#######################


# def app(inputPath,
#         sumPath,
#         batchSize=64,
#         embeddingSize=300,
#         lineNum=-1,
#         hiddenSize=512,
#         maxLen=64,
#         epochs=32,
#         weightPath=None,
#         dicLength=10000
#         ):
#     if weightPath is None:
#         weightPath = '../save/model.w'
#
#     ########################################################################
#     print("sentences Path: " + inputPath)
#     print("summary Path: " + sumPath)
#     print("batchSize: " + str(batchSize))
#     print("embeddingSize :  " + str(embeddingSize))
#     print("hiddenSize: " + str(hiddenSize))
#     print("sentence length: " + str(maxLen))
#     print("epochSize: " + str(epochs))
#     if lineNum != -1:
#         print("max number of lines: " + str(lineNum))
#     ##############获得经过处理的np 数组###########################
#     # lines, keyWords, oneHotKeyWords, dict = getTrainData(inputPath
#     #                                                      , sumPath
#     #                                                      , dicLength=dicLength
#     #                                                      , inputShape=maxLen
#     #                                                      , lineNum=lineNum
#     #
#     #                                                      )
#     # libSize = len(dict) + 1
#     # print("read training data complete!!!!")
#     # print("voc length: " + str(libSize))
#     # model = createModel(maxLen, libSize, embeddingSize, hiddenSize)
#     # if weightPath is not None and os.path.isfile(weightPath):
#     #     model.load_weights(weightPath)
#     # # model.summary()
#     # model.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
#     #               loss="categorical_crossentropy",
#     #               metrics=['accuracy'])
#     # ############ 和学长沟通后 y_label 为 ground-Truth 的one-hot 形式
#     # model.fit({'sentence': lines, 'keyWords': keyWords}
#     #           , [oneHotKeyWords]
#     #           , batch_size=batchSize
#     #           , epochs=epochs
#     #           , validation_split=0.1
#     #           )
#     #
#     # model.save_weights(weightPath)
#
#
# ###############可以支持分文件训练#############
# def runByMul(dirPath='../data/train',
#              srcPre='tgt',
#              tgtPre='src',
#              fileCount=100,
#              batchSize=64,
#              embeddingSize=300,
#              lineNum=-1,
#              hiddenSize=512,
#              maxLen=64,
#              epochs=32,
#              dicLength=10000,
#              startIndex=0
#              ):
#     srcPath = dirPath + "/" + srcPre
#     tgtPath = dirPath + "/" + tgtPre
#     if not os.path.exists(srcPath):
#         return
#     if not os.path.exists(tgtPath):
#         return
#     srcFiles = os.listdir(dirPath + "/" + srcPre)
#     tgtFiles = os.listdir(dirPath + "/" + tgtPre)
#     if len(srcFiles) != len(tgtFiles):
#         return
#     for i in range(len(srcFiles)):
#         if i < startIndex:
#             continue
#         if fileCount != 0 and fileCount < i:
#             break
#         print(srcPath + '/' + srcFiles[i])
#         print(tgtPath + '/' + tgtFiles[i])
#         gc.collect()
#         app(srcPath + '/' + srcFiles[i]
#             , tgtPath + '/' + tgtFiles[i],
#             batchSize=batchSize,
#             embeddingSize=embeddingSize,
#             lineNum=lineNum,
#             hiddenSize=hiddenSize,
#             maxLen=maxLen,
#             epochs=epochs,
#             dicLength=dicLength
#             )
#

def extractionBybert(
        sentecePath,
        keyWordPath,
        modelPath,
        hiddensize,
        maxlen=64,
        epochs=32,
        batchSize=64,
        weightPath='../save/model.w',
):
    print("模型初始化开始")
    tokenizer, dic = initTokenizer()
    print("分词器初始化完成")
    print(len(dic))
    model = loadBertModl(modelPath, hiddensize, maxlen * 2, len(dic))
    model.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    print("模型加载完成")
    token_ids, sentences, one_hot = readTrainData(sentecePath,
                                                  keyWordPath,
                                                  maxlen,
                                                  tokenizer,
                                                  dic)
    if weightPath is not None and os.path.isfile(weightPath):
        model.load_weights(weightPath)
    model.fit([token_ids, sentences]
              , [one_hot]
              , batch_size=batchSize
              , epochs=epochs
              , validation_split=0.1
              )
    model.save_weights(weightPath)


# if __name__ == '__main__':
# for i in range(10):
#     runByMul(epochs=1, startIndex=9);
extractionBybert('../data/train/src/train.src',
                 '../data/train/tgt/train.tgt',
                 './model/uncased_L-2_H-128_A-2',
                 128
                 )
