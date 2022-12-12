import os

from models.zoo.bert.bert_modeling import BertConfig, BertModel
from models.zoo.bert.tokenization import FullTokenizer
from models.zoo.bert.bert_utils import restore_model_ckpt
from models.abstract_model import *


def build_BertNer(bert_model, num_labels, max_seq_length):
    float_type = tf.float32
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
    valid_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='valid_ids')

    if type(bert_model) == str:
        bert_config = BertConfig.from_json_file(os.path.join(bert_model, "bert_config.json"))
    elif type(bert_model) == dict:
        bert_config = BertConfig.from_dict(bert_model)

    bert_layer = BertModel(config=bert_config, float_type=float_type)
    pooled_output, sequence_output = bert_layer(input_word_ids, input_mask, input_type_ids)

    # bn = tf.keras.layers.BatchNormalization(momentum=0.99, scale=True, center=True)(pooled_output)
    classifier = tf.keras.layers.Dense(num_labels, activation='softmax', dtype=float_type)(pooled_output)

    bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids, valid_ids], outputs=[classifier])

    return bert


def create_model(bert_config, seq_len=10, lr=0.001, decay=0, do_compile=True, default_weights=True):
    bert_path = 'models/zoo/bert/models/' + bert_config
    tokenizer = FullTokenizer(os.path.join(bert_path, "vocab.txt"), True)
    model = build_BertNer(bert_path, len(list(tokenizer.vocab.keys())), seq_len)

    if do_compile:
        compile_model(model, lr, decay)

    if default_weights == 'pretrained':
        model = restore_model_ckpt(model, bert_path)
    elif default_weights == 'pretrained-frozen':
        model = restore_model_ckpt(model, bert_path)
        for layer in model.layers:
            if isinstance(layer, BertModel):
                layer.trainable = False
    elif default_weights is True:
        assign_default_weights(model, 'nwp-bert' + str(bert_config))

    return model


def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


def load(model_path):
    return load('bert_nwp/' + model_path)


def save(model, model_path):
    save(model, 'bert_nwp/' + model_path)
