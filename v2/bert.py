from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array

import numpy as np

config_path = './model/uncased_L-2_H-128_A-2/bert_config.json'
checkpoint_path = './model/uncased_L-2_H-128_A-2/bert_model.ckpt'
dict_path = './model/uncased_L-2_H-128_A-2/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
dic=tokenizer._token_dict;
print(dic)
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
model.summary()
# model.output
# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
token_ids, segment_ids = to_array([token_ids], [segment_ids])
print(token_ids.shape)
#
# print('\n ===== predicting =====\n')
# print(model.predict([token_ids, segment_ids]))
