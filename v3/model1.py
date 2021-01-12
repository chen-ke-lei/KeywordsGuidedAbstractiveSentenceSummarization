from v3.dataProcess import *
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from keras.layers import *
from bert4keras.tokenizers import Tokenizer, load_vocab
from keras.utils import to_categorical
from keras.optimizers import *
import os
from keras import regularizers


def model1(data_path='../law_data/train_cs.json',
           modelName='uncased_L-2_H-128_A-2',
           maxlen=512,
           batch_size=128,
           testPath='../law_data/test_cs.json',
           epochs=20,
           baseModelPath='./model/',
           baseSavePath='../save/'
           ):
    dict_path = baseModelPath + modelName + '/vocab.txt'
    config_path = baseModelPath + modelName + '/bert_config.json'
    checkpoint_path = baseModelPath + modelName + '/bert_model.ckpt'
    weightPath = baseSavePath + modelName + '_l2' + '.weights'
    token_dict, keep_tokens, tokenizer = loadDic(dict_path)

    bertModel = build_transformer_model(
        config_path,
        checkpoint_path,
        model='bert',
        sequence_length=maxlen,
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    hiddenSate = bertModel.outputs[0]
    # hiddenSate = bertModel.outputs
    # accTmp = Dense(100, activation='relu')(hiddenSate)
    # lawTmp = Dense(100, activation='relu')(hiddenSate)
    # termTmp = Dense(100, activation='relu')(hiddenSate)
    # hiddenSate = LSTM(128)(hiddenSate)
    hiddenSate = Lambda(lambda x: K.sum(x, axis=1))(hiddenSate)
    # hiddenSate = Lambda(lambda x: x[:, 0:1, :])(hiddenSate)
    # print("hiddenSate.shape " + str(K.int_shape(hiddenSate)))
    #   hiddenSate = Dropout(0.2)(hiddenSate)
    # accTmp = Lambda(lambda x: K.sum(x, axis=1))(accTmp)
    # lawTmp = Lambda(lambda x: K.sum(x, axis=1))(lawTmp)
    # termTmp = Lambda(lambda x: K.sum(x, axis=1))(termTmp)
    accuOut = Dense(117, name='acc', activation='softmax', kernel_regularizer=regularizers.l2())(hiddenSate)
    lawOut = Dense(101, name='law', activation='softmax', kernel_regularizer=regularizers.l2())(hiddenSate)
    termOut = Dense(11, name='term', activation='softmax', kernel_regularizer=regularizers.l2())(hiddenSate)

    model = Model(inputs=bertModel.inputs, outputs=[accuOut, lawOut, termOut])
    if weightPath is not None and os.path.isfile(weightPath):
        model.load_weights(weightPath)
    model.summary()
    model.compile(optimizer=Adam(lr=0.0005
                                 , beta_1=0.9
                                 , beta_2=0.999
                                 , epsilon=1e-08
                                 ),
                  loss={
                      'acc': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'law': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'term': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),

                  },
                  loss_weights={
                      'acc': 1.1,
                      'law': 1.,
                      'term': 1.
                  },
                  metrics=['sparse_categorical_accuracy',
                           ]
                  )
    ########加载训练数据
    factList, accuList, lawList, termList = loadLawData(data_path)
    token_ids, segment_ids = data2index(tokenizer, maxlen, factList)

    ##########加载测试数据
    factTest, accuTest, lawTest, termTest = loadLawData(testPath)
    token_test, segment_test = data2index(tokenizer, maxlen, factTest)

    model.fit([token_ids, segment_ids],
              {'acc': accuList,
               'law': lawList,
               'term': termList,
               },

              validation_data=([token_test, segment_test], {'acc': accuTest,
                                                            'law': lawTest,
                                                            'term': termTest,
                                                            }),
              batch_size=batch_size,
              epochs=epochs,
              validation_freq=4,
              # validation_split=0.1,
              # steps_per_epoch=steps_per_epoch
              )
    model.save_weights(weightPath)


# model1()


def model1_LSTM_ENCODE(data_path='../law_data/train_cs.json',
                       maxlen=512,
                       batch_size=256,
                       testPath='../law_data/test_cs.json',
                       epochs=20,
                       baseSavePath='../save/'
                       ):
    dic = loadDicByPKL()
    embeddingWeight = loadEmbeddingWeight()
    weightPath = baseSavePath + 'LSTM' + '_ebdwgt' + '.weights'
    dataInput = Input(shape=(maxlen,))
    embedding = Embedding(len(dic), 200, weights=[embeddingWeight])(dataInput)
    hiddenSate = LSTM(128)(embedding)
    accTmp = Dense(100, activation='tanh')(hiddenSate)
    lawTmp = Dense(100, activation='tanh')(hiddenSate)
    termTmp = Dense(100, activation='tanh')(hiddenSate)
    accuOut = Dense(117, name='acc', activation='softmax')(accTmp)
    lawOut = Dense(101, name='law', activation='softmax')(lawTmp)
    termOut = Dense(11, name='term', activation='softmax')(termTmp)

    model = Model(inputs=[dataInput], outputs=[accuOut, lawOut, termOut])
    if weightPath is not None and os.path.isfile(weightPath):
        model.load_weights(weightPath)
    model.summary()
    model.compile(optimizer=Adam(lr=0.0005
                                 , beta_1=0.9
                                 , beta_2=0.999
                                 , epsilon=1e-08
                                 ),
                  loss={
                      'acc': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'law': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'term': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),

                  },
                  loss_weights={
                      'acc': 1.1,
                      'law': 1.,
                      'term': 1.
                  },
                  metrics=['sparse_categorical_accuracy',
                           ]
                  )
    ########加载训练数据
    factList, accuList, lawList, termList = loadBaseDataByWords(data_path, dic, maxlen=maxlen)

    ###########加载测试数据
    factTest, accuTest, lawTest, termTest = loadBaseDataByWords(testPath, dic, maxlen=maxlen)

    model.fit([factList],
              {'acc': accuList,
               'law': lawList,
               'term': termList,
               },

              validation_data=([factTest], {'acc': accuTest,
                                            'law': lawTest,
                                            'term': termTest,
                                            }),
              batch_size=batch_size,
              epochs=epochs,
              validation_freq=4,
              # validation_split=0.1,
              # steps_per_epoch=steps_per_epoch
              )
    model.save_weights(weightPath)


def model2_lstm_encoder(data_path='../law_data/law_mark_new.json',
                        maxlen=512,
                        batch_size=16,
                        decodeHiddenState=128,
                        summaryLen=32,
                        epochs=30,
                        baseSavePath='../save/'
                        ):
    dic = loadDicByPKL()
    embeddingWeight = loadEmbeddingWeight()
    weightPath = baseSavePath + 'LSTM' + '_ebdwgt' + '.weights'
    dataInput = Input(shape=(maxlen,))
    summaryInput = Input(shape=(summaryLen, len(dic)))
    embedding = Embedding(len(dic), 200, weights=[embeddingWeight])(dataInput)
    hiddenSateAll = LSTM(128, return_sequences=True)(embedding)
    hiddenSate = Lambda(lambda x: x[:, -2:-1, :])(hiddenSateAll)
    accTmp = Dense(100, activation='tanh')(hiddenSate)
    lawTmp = Dense(100, activation='tanh')(hiddenSate)
    termTmp = Dense(100, activation='tanh')(hiddenSate)
    accuOut = Dense(117, name='acc', activation='softmax')(accTmp)
    lawOut = Dense(101, name='law', activation='softmax')(lawTmp)
    termOut = Dense(11, name='term', activation='softmax')(termTmp)

    basemodel = Model(inputs=[dataInput], outputs=[accuOut, lawOut, termOut])
    if weightPath is not None and os.path.isfile(weightPath):
        print("加载weight成功:  " + weightPath)
        basemodel.load_weights(weightPath)

    encoder_outputs, state_h, state_c = LSTM(decodeHiddenState, return_state=True)(hiddenSate)
    encoder_states = [state_h, state_c]

    summaryPredicts, _, _ = LSTM(decodeHiddenState, return_sequences=True, return_state=True)(summaryInput,
                                                                                              initial_state=encoder_states)
    summaryPredicts = Dense(len(dic), activation='softmax', name='summary')(summaryPredicts)

    keyWordOut = Dense(2, name='keyWord', activation='softmax')(hiddenSate)

    model = Model(inputs=[dataInput, summaryInput]
                  , outputs=[accuOut, lawOut, termOut, keyWordOut, summaryPredicts])

    model.compile(optimizer=Adam(lr=0.0005
                                 , beta_1=0.9
                                 , beta_2=0.999
                                 , epsilon=1e-08
                                 ),
                  loss={
                      'acc': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'law': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'term': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'keyWord': tf.keras.losses.CategoricalCrossentropy(),
                      'summary': tf.keras.losses.CategoricalCrossentropy(),
                  },
                  loss_weights={
                      'acc': 1.1,
                      'law': 1.,
                      'term': 1.,
                      'keyWord': 1.,
                      'summary': 1.
                  },
                  metrics={'acc': 'sparse_categorical_accuracy',
                           'law': 'sparse_categorical_accuracy',
                           'term': 'sparse_categorical_accuracy',
                           'keyWord': 'accuracy',
                           'summary': 'accuracy',
                           }
                  )
    model.summary()
    accuList, lawList, termList, summaryListInput, summaryListOutput, factList, keyWordsList = loadAllDataByWords(
        path=data_path,
        dic=dic,
        maxlen=maxlen,
        summaryLen=summaryLen)
    summaryListInput = to_categorical(summaryListInput, num_classes=len(dic))
    summaryListOutput = to_categorical(summaryListOutput, num_classes=len(dic))
    model.fit([factList, summaryListInput],
              {'acc': accuList,
               'law': lawList,
               'term': termList,
               'keyWord': keyWordsList,
               'summary': summaryListOutput,
               },
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              # steps_per_epoch=steps_per_epoch
              )
    model.save_weights(weightPath)


def model2(data_path='../law_data/law_mark_new.json',
           modelName='uncased_L-12_H-128_A-2',
           maxlen=512,
           batch_size=16,
           decodeHiddenState=128,
           summaryLen=32,
           epochs=30,
           baseModelPath='./model/',
           baseSavePath='../save/'
           ):
    dict_path = baseModelPath + modelName + '/vocab.txt'
    config_path = baseModelPath + modelName + '/bert_config.json'
    checkpoint_path = baseModelPath + modelName + '/bert_model.ckpt'
    weightPath = baseSavePath + modelName + '.weights'
    ##加载字典
    token_dict, keep_tokens, tokenizer = loadDic(dict_path)
    #########数据加载#######
    factList, summaryList, accuList, lawList, termList, keyWordsList = loadLawDataContainKeyWordsAndSummary(data_path,
                                                                                                            maxlen)
    token_ids, segment_ids = data2index(tokenizer, maxlen, factList)
    summaryIds, _ = data2index(tokenizer, summaryLen, summaryList)
    summaryOut = []
    for x in summaryIds:
        y = x[1:]
        y = np.append(y, 1)
        summaryOut.append(y)

    summaryOut = np.array(summaryOut)
    summaryOut = to_categorical(summaryOut, num_classes=len(token_dict))
    summaryIds = to_categorical(summaryIds, num_classes=len(token_dict))
    ########################

    bertModel = build_transformer_model(

        config_path,
        checkpoint_path,
        sequence_length=maxlen,
        # application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表

    )
    summaryInput = Input(shape=(None, len(token_dict)))
    hiddenSate = bertModel.outputs[0]

    ###########一个简单的summary 生成模型
    encoder_outputs, state_h, state_c = LSTM(decodeHiddenState, return_state=True)(hiddenSate)
    encoder_states = [state_h, state_c]

    summaryPredicts, _, _ = LSTM(decodeHiddenState, return_sequences=True, return_state=True)(summaryInput,
                                                                                              initial_state=encoder_states)
    summaryPredicts = Dense(len(token_dict), activation='softmax', name='summary')(summaryPredicts)

    keyWordOut = Dense(2, name='keyWord', activation='softmax')(hiddenSate)
    hiddenSate = bertModel.outputs[0]

    # accTmp = Dense(100, activation='relu')(hiddenSate)
    # lawTmp = Dense(100, activation='relu')(hiddenSate)
    # termTmp = Dense(100, activation='relu')(hiddenSate)
    hiddenSate = Lambda(lambda x: K.sum(x, axis=1))(hiddenSate)
    # accTmp = Lambda(lambda x: K.sum(x, axis=1))(accTmp)
    # lawTmp = Lambda(lambda x: K.sum(x, axis=1))(lawTmp)
    # termTmp = Lambda(lambda x: K.sum(x, axis=1))(termTmp)
    accuOut = Dense(117, name='acc', activation='softmax')(hiddenSate)
    lawOut = Dense(101, name='law', activation='softmax')(hiddenSate)
    termOut = Dense(11, name='term', activation='softmax')(hiddenSate)
    baseModel = Model(inputs=bertModel.inputs, outputs=[accuOut, lawOut, termOut])
    if weightPath is not None and os.path.isfile(weightPath):
        baseModel.load_weights(weightPath)
    model = Model(inputs=[bertModel.inputs[0], bertModel.inputs[1], summaryInput]
                  , outputs=[accuOut, lawOut, termOut, keyWordOut, summaryPredicts])

    model.summary()
    model.compile(optimizer=Adam(lr=0.001
                                 , beta_1=0.9
                                 , beta_2=0.999
                                 , epsilon=1e-08
                                 ),
                  loss={
                      'acc': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'law': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'term': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      'keyWord': tf.keras.losses.CategoricalCrossentropy(),
                      'summary': tf.keras.losses.CategoricalCrossentropy(),
                  },
                  loss_weights={
                      'acc': 100.,
                      'law': 100.,
                      'term': 100.,
                      'keyWord': 1.,
                      'summary': 1.
                  },
                  metrics={'acc': 'sparse_categorical_accuracy',
                           'law': 'sparse_categorical_accuracy',
                           'term': 'sparse_categorical_accuracy',
                           'keyWord': 'accuracy',
                           'summary': 'accuracy',
                           }
                  )
    model.fit([token_ids, segment_ids, summaryIds],
              {'acc': accuList,
               'law': lawList,
               'term': termList,
               'keyWord': keyWordsList,
               'summary': summaryOut,
               },
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              # steps_per_epoch=steps_per_epoch
              )
    model.save_weights(weightPath)

for x in range(100):
    model2_lstm_encoder()
# # for x in range(100):
# #     model1_LSTM_ENCODE()
