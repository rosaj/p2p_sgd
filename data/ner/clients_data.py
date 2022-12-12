import os
import numpy as np
from data.ner.dataset_loader import CoNLLProcessor, FewNERDProcessor
from models.zoo.bert.tokenization import FullTokenizer

global PROCESSOR


def load_clients_data(num_clients, dataset='conll', seq_len=128, vocab_path='data/ner'):
    if dataset == 'conll':
        processor = CoNLLProcessor('data/ner/conll')
    else:
        processor = FewNERDProcessor('data/ner/few_nerd')
    global PROCESSOR
    PROCESSOR = processor
    tokenizer = FullTokenizer(os.path.join(vocab_path, "vocab.txt"), True)

    train_features = processor.get_train_as_features(seq_len, tokenizer)
    val_features = processor.get_dev_as_features(seq_len, tokenizer)
    test_features = processor.get_test_as_features(seq_len, tokenizer)

    data = {
        "train": split_features(train_features, num_clients),
        "val": split_features(val_features, num_clients),
        "test": split_features(test_features, num_clients),
    }
    return data


def split_features(features, num_clients):
    features_split = np.array_split(features, num_clients)
    ds_list = []
    for client_data in features_split:
        c_input_ids = [f.input_ids for f in client_data]
        c_input_mask = [f.input_mask for f in client_data]
        c_segment_ids = [f.segment_ids for f in client_data]
        c_valid_ids = [f.valid_ids for f in client_data]
        c_label_id = [f.label_id for f in client_data]
        c_label_mask = [f.label_mask for f in client_data]

        ds_list.append(
            (
                (np.asarray(c_input_ids), np.asarray(c_input_mask), np.asarray(c_segment_ids), np.asarray(c_valid_ids)),
                (np.asarray(c_label_id), np.asarray(c_label_mask))
            )
        )
    return ds_list[:num_clients]
