import os
import tensorflow as tf
from tensorflow import keras

from common.ner.bert.bert_modeling import BertConfig, BertModel
# import common.abstract_model as ab_mod
from common.abstract_model import *
from common.ner.bert.metrics import MaskedSparseCategoricalCrossentropy
from seqeval.metrics import classification_report, accuracy_score

import data.ner.clients_data as cd


class ValidationLayer(keras.layers.Layer):
    def call(self, sequence_output, valid_ids):
        sq = sequence_output
        vi = valid_ids

        def val_fn(i):
            cond = tf.equal(vi[i], tf.constant(1, dtype=tf.int32))
            temp = tf.squeeze(tf.gather(sq[i], tf.where(cond)))
            r = tf.tile(tf.zeros(tf.shape(sq[i])[1]), [tf.math.subtract(tf.shape(sq[i])[0], tf.shape(temp)[0])])
            r = tf.reshape(r, [-1, tf.shape(sq[i])[1]])
            n = tf.concat([temp, r], 0)
            return n

        n_vo = tf.map_fn(val_fn, tf.range(tf.shape(sq)[0]), dtype=tf.float32)
        return n_vo


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
    _, sequence_output = bert_layer(input_word_ids, input_mask, input_type_ids)

    val_layer = ValidationLayer()(sequence_output, valid_ids)

    dropout = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(val_layer)

    initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)

    classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)(dropout)

    bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids, valid_ids], outputs=[classifier])

    return bert


def create_model(bert_config, seq_len=128, lr=0.001, decay=0, do_compile=True, default_weights=True):
    processor = cd.PROCESSOR
    model = build_BertNer('common/ner/bert/models/' + bert_config, processor.label_len(), seq_len)

    if do_compile:
        compile_model(model, lr, decay)

    if default_weights:
        assign_default_weights(model, 'mnist' + str(bert_config))

    return model


def compile_model(model, lr=0.001, decay=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
        loss=MaskedSparseCategoricalCrossentropy()
    )


def evaluate(model, batched_eval_data, label_map, out_ind, pad_ind, do_print=False):
    y_true, y_pred = [], []
    # for (input_ids, input_mask, segment_ids, valid_ids, label_ids) in batched_eval_data:
    for (input_ids, input_masks, segment_ids, valid_ids), (label_ids, label_mask) in batched_eval_data:
        logits = model((input_ids, input_masks, segment_ids, valid_ids), training=False)
        logits = tf.argmax(logits, axis=2)
        for i in range(label_ids.shape[0]):
            lbl_ids = label_ids[i]
            pred_ids = logits[i]
            cond = tf.not_equal(lbl_ids, tf.constant(pad_ind, dtype=tf.int64))
            lbl_ids = tf.squeeze(tf.gather(lbl_ids, tf.where(cond)))[1:-1]
            pred_ids = tf.squeeze(tf.gather(pred_ids, tf.where(cond)))[1:-1]

            # temp_1 = list(map(lambda x: label_map[x.numpy()], lbl_ids))
            temp_1 = list(map(
                lambda x: label_map[x.numpy()] if label_map[x.numpy()] not in ['[SEP]', '[PAD]', '[CLS]'] else
                label_map[out_ind], lbl_ids))

            temp_2 = list(map(
                lambda x: label_map[x.numpy()] if label_map[x.numpy()] not in ['[SEP]', '[PAD]', '[CLS]'] else
                label_map[out_ind], pred_ids))

            y_true.append(temp_1)
            y_pred.append(temp_2)

    # accuracy = accuracy_score(y_true, y_pred)

    if do_print:
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    class_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    # class_dict['accuracy'] = accuracy
    out_dict = {}
    for k, v in class_dict.items():
        for dk, dv in v.items():
            out_dict[k.replace('-', '_').replace(' ', '_') + '_' + dk.replace('-', '_').replace(' ', '_')] = dv
    return out_dict


def eval_model_metrics(m, dataset):
    processor = cd.PROCESSOR
    return evaluate(m, dataset,
                    processor.get_label_map(),
                    processor.token_ind('O'),
                    processor.token_ind('[SEP]'),
                    processor.token_ind('[PAD]'))


def eval_ensemble_metrics(models, dataset, metrics, weights=None):
    if weights is None:
        weights = len(models) * [1 / len(models)]
    processor = cd.PROCESSOR
    return evaluate_models(models, weights, dataset,
                           processor.get_label_map(),
                           processor.token_ind('O'),
                           processor.token_ind('[SEP]'),
                           processor.token_ind('[PAD]'))


def evaluate_models(models, weights, batched_eval_data, label_map, out_ind, pad_ind, do_print=False):
    y_true, y_pred = [], []
    for (input_ids, input_masks, segment_ids, valid_ids), (label_ids, label_mask) in batched_eval_data:
        logits_list = [model((input_ids, input_masks, segment_ids, valid_ids), training=False) for model in models]
        logits = sum((m_logits * w for m_logits, w in zip(logits_list, weights)))
        logits = tf.argmax(logits, axis=2)
        for i in range(label_ids.shape[0]):
            lbl_ids = label_ids[i]
            pred_ids = logits[i]
            cond = tf.not_equal(lbl_ids, tf.constant(pad_ind, dtype=tf.int64))
            lbl_ids = tf.squeeze(tf.gather(lbl_ids, tf.where(cond)))[1:-1]
            pred_ids = tf.squeeze(tf.gather(pred_ids, tf.where(cond)))[1:-1]

            # temp_1 = list(map(lambda x: label_map[x.numpy()], lbl_ids))
            temp_1 = list(map(
                lambda x: label_map[x.numpy()] if label_map[x.numpy()] not in ['[SEP]', '[PAD]', '[CLS]'] else
                label_map[out_ind], lbl_ids))

            temp_2 = list(map(
                lambda x: label_map[x.numpy()] if label_map[x.numpy()] not in ['[SEP]', '[PAD]', '[CLS]'] else
                label_map[out_ind], pred_ids))

            y_true.append(temp_1)
            y_pred.append(temp_2)

    # accuracy = accuracy_score(y_true, y_pred)

    if do_print:
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    class_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    # class_dict['accuracy'] = accuracy
    out_dict = {}
    for k, v in class_dict.items():
        for dk, dv in v.items():
            out_dict[k.replace('-', '_').replace(' ', '_') + '_' + dk.replace('-', '_').replace(' ', '_')] = dv
    return out_dict
