import os

from models.zoo.bert.tokenization import FullTokenizer
from models.abstract_model import *

from models.zoo.bert.bert_model import new_bert, restore_pretrained_weights
from models.zoo.bert.metrics import MaskedSparseCategoricalCrossentropy, MaskedSparseCategoricalAccuracy, MaskedSparseTopKCategoricalAccuracy

COUNT = 0


def build_bert_nwp(bert_model, num_labels, max_seq_length):
    (input_word_ids, input_mask, input_type_ids, valid_ids), (pooled_output, sequence_output), bert_config = new_bert(bert_model, max_seq_length)

    classifier = tf.keras.layers.Dense(num_labels, activation='softmax', dtype=tf.float32)(sequence_output)

    bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids, valid_ids], outputs=[classifier])

    return bert


def create_model(bert_config, seq_len=12, lr=5e-4, decay=0, do_compile=True, default_weights=True):
    bert_path = 'models/zoo/bert/models/' + bert_config
    tokenizer = FullTokenizer(os.path.join(bert_path, "vocab.txt"), True)
    model = build_bert_nwp(bert_path, len(list(tokenizer.vocab.keys())), seq_len)
    global COUNT
    model._name = "bert-nwp_{}".format(COUNT)
    COUNT += 1

    if do_compile:
        compile_model(model, lr, decay)

    if type(default_weights) == str:
        if default_weights == 'global':
            assign_default_weights(model.layers[3], 'global-bert' + str(bert_config))
        elif 'pretrained' in default_weights:
            model = restore_pretrained_weights(model, bert_path, 'frozen' in default_weights)
        else:
            assign_default_weights(model, "{}-{}".format(default_weights, str(bert_config)))
    elif default_weights is True:
        assign_default_weights(model, 'nwp-bert' + str(bert_config))

    return model


# Sequence output
def compile_model(model, lr=0.001, decay=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
        loss=MaskedSparseCategoricalCrossentropy(),
        metrics=[
            MaskedSparseCategoricalAccuracy(),
            MaskedSparseTopKCategoricalAccuracy(k=3)
        ]
    )


# Pooled output

"""
def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
"""
