#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933

from __future__ import print_function
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from v2.dataProcess import *
import os

# 基本参数
maxlen = 512
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
modelName = 'uncased_L-12_H-128_A-2'
config_path = './model/' + modelName + '/bert_config.json'
checkpoint_path = './model/' + modelName + '/bert_model.ckpt'
dict_path = '../data/dic.txt'
data_path = '../law_data/test_cs.json'
tmpWeight = '../save/' + modelName + '_tmp.weights'
# 训练样本。THUCNews数据集，每个样本保存为一个txt。
txts = loadLawData(data_path)

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, txt in self.sample(random):
            if len(txt) > 1:
                token_ids, segment_ids = tokenizer.encode(
                    txt, maxlen=maxlen
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoSummary(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autoSummary = AutoSummary(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


def just_show():
    s1 = u'2015年11月10日晚9时许，被告人李某的妹妹李某某与被害人华某某在桦川县悦来镇石锅烤肉吃饭时发生口角，华某某殴打李某某被他人拉开。后李某某打电话将此事告知李某。李某便开车接上李某某在悦来镇“0454饮吧”找到华某某并质问其因何殴打李某某，之后二人厮打在一起。李某用拳头、巴掌连续击打华某某脸部，致华受伤住院治疗。经桦川县公安局司法鉴定，华某某所受伤为轻伤二级。'
    s2 = u'2014年5月6日14时许，被告人叶某某驾车途径赤壁市赵李桥镇胜利街涵洞时，被在此处饭店外的朱某某等人挡住去路，叶某某与朱某某为此发生争吵。随后，叶某某到赵李桥镇街道胡某某茶馆准备打牌，将自己的小车停在茶馆门前。朱某某的丈夫叶某甲带着外甥肖某回家时，发现叶某某的车子停在胡某某门外，肖某便用手拍打汽车，扬言要打叶某某，后被胡某某劝离。叶某某随后邀约余某某、黎某某、黄某某、陈某某（均另案处理）等人来到叶某甲楼下，与叶某甲、肖某及叶某甲另一个外甥刘某某发生厮打，被告人一伙手持木棍、砍刀、砖头将叶某甲、肖某、刘某某打伤。经鉴定：叶某甲为轻伤二级，肖某、刘某某均为轻微伤，叶某某亦受轻微伤。2015年1月26日18时许，被告人叶某某在赤壁市赵李桥镇紫阳酒店被公安民警抓获归案。同时查明，当事人双方已就本案民事赔偿问题自愿达成如下协议：即由被告人叶某某一次性赔偿被害人叶某甲、肖某、刘某某各项经济损失4万元，被害人表示不追究叶某某等人的法律责任。上述事实，被告人叶某某在开庭过程中亦无异议，且有被害人叶某甲、肖某、刘某某的陈述、证人朱某某、胡某某、甘某某等人的证言、辨认笔录、鉴定意见、调解协议、谅解书、户籍证明、到案经过等证据证实，'
    for s in [s1, s2]:
        print(u'生成标题:', autoSummary.generate(s))
    print()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('../save/best_model.weights')
        # 演示效果
        model.save_weights(tmpWeight)
        just_show()


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(txts, batch_size)
    if tmpWeight is not None and os.path.isfile(tmpWeight):
        model.load_weights(tmpWeight)
    model.fit(
        train_generator.forfit(),

        steps_per_epoch=steps_per_epoch,
        epochs=epochs,

        callbacks=[evaluator]
    )

else:
    model.load_weights('../save/best_model.weights')
