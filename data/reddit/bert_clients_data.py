import string
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.zoo.bert.tokenization import FullTokenizer
import h5py


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def convert_nwp_examples_to_features(paragraphs, seq_len, tokenizer):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def tokenize(sent):
        return tokenizer.tokenize(
            sent.strip().translate(translator).replace(' ' * 4, '').replace(' ' * 3, '').replace(' ' * 2, ' '))

    sentence_tokens = []
    for p in paragraphs:
        sent_tok = [tokenize(s) for s in p.split('.')]
        sentence_tokens.extend(sent_tok)

    sentence_tokens = [st for st in sentence_tokens if len(st) > 0]
    valid_tokens_sentences = [np.logical_not(np.char.startswith(st, '##')).astype(np.uint8) for st in sentence_tokens]
    sentences = [tokenizer.convert_tokens_to_ids(st) for st in sentence_tokens]

    seqs = []
    valid_seq_ind = []
    cls = tokenizer.vocab['[CLS]']
    # sep = tokenizer.vocab['[SEP]']
    for sentence, valid_tokens in zip(sentences, valid_tokens_sentences):
        for i in range(1, len(sentence)):
            # skip predicting non-valid tokens
            if valid_tokens[i] == 1:
                start_ind = max(i - seq_len + 2, 0)
                words = sentence[start_ind: i + 1]
                valid_word_ind = valid_tokens[start_ind: i + 1]
                if valid_tokens[start_ind] != 1:
                    prev_valid_tokens = np.ndarray.flatten(np.argwhere(valid_tokens[:start_ind] == 1))
                    first_valid_word_ind = np.argwhere(valid_word_ind == 1)[0][0]
                    for je, ji in enumerate(range(first_valid_word_ind - 1,
                                                  first_valid_word_ind - min(len(prev_valid_tokens),
                                                                             first_valid_word_ind) - 1, -1)):
                        words[ji] = sentence[prev_valid_tokens[-je - 1]]
                        valid_word_ind[ji] = 1
                    while valid_word_ind[0] != 1:
                        words.pop(0)
                        valid_word_ind = valid_word_ind[1:]
                words.insert(0, cls)
                # words.append(sep)
                valid_word_ind = np.insert(valid_word_ind, 0, 1)
                # valid_word_ind = np.append(valid_word_ind, 1)
                seqs.append(words)
                valid_seq_ind.append(valid_word_ind)
    text_seqs = np.array(
        pad_sequences(np.array(seqs, dtype=object), maxlen=seq_len, padding='post', value=tokenizer.vocab['[PAD]']),
        dtype=np.int32)
    valid_ids = np.array(pad_sequences(np.array(valid_seq_ind, dtype=object), maxlen=seq_len, padding='post', value=1),
                         dtype=np.uint8)
    t_x = text_seqs[:, :]
    label_ids = np.full(t_x.shape, 0, dtype=np.int32)
    mask_ind = tokenizer.vocab['[MASK]']
    input_mask = np.full(t_x.shape, 0, dtype=np.uint8)
    for i in range(len(t_x)):
        ind = np.where(t_x[i] == 0)
        if len(ind[0]) > 0:
            ind = ind[0][0] - 1
        else:
            ind = len(t_x[i]) - 1
        label_ids[i][ind] = t_x[i][ind]
        t_x[i][ind] = mask_ind
        input_mask[i][:ind + 1] = 1

    input_word_ids = t_x
    input_type_ids = np.full(input_word_ids.shape, 0, dtype=np.uint8)
    label_mask = np.array(label_ids > 0, dtype=np.uint8)

    features = [
        InputFeatures(input_ids=input_word_ids[i],
                      input_mask=input_mask[i],
                      segment_ids=input_type_ids[i],
                      label_id=label_ids[i],
                      valid_ids=valid_ids[i],
                      label_mask=label_mask[i]) for i in range(len(input_word_ids))
    ]
    return features


def clean_text(text):
    text = text.replace('<EOS>', '').replace('<BOS>', '').replace('<PAD>', '').strip() \
        .replace('\n', ' ').replace('\r', ' ').replace('\ufeff', '').strip()
    return text


def parse_reddit_file(reddit_filename='reddit_0_train.json', seq_len=10, tokenizer_path='data/ner/vocab.txt'):
    tokenizer = FullTokenizer(tokenizer_path, True)
    reddit_filepath = 'data/reddit/source/data/reddit_leaf/' + reddit_filename.split('.')[0].split('_')[-1] + '/' + reddit_filename
    with open(reddit_filepath, 'r') as inf:
        cdata = json.load(inf)
    json_data = cdata['user_data']
    j_agents_x = []
    j_agents_y = []
    for key in list(json_data.keys()):
        j_agent_data_x = []
        j_agent_data_y = []
        for sx, sy in zip(json_data[str(key)]['x'], json_data[str(key)]['y']):
            s = ' '.join([' '.join(l) for l in sx])
            s = clean_text(s)
            features = convert_nwp_examples_to_features(s.split('.'), seq_len=seq_len, tokenizer=tokenizer)
            j_agent_data_x.extend(features)
            j_agent_data_y.extend([sy['subreddit']] * len(features))

        j_agents_x.append(j_agent_data_x)
        j_agents_y.append(j_agent_data_y)
    return j_agents_x, j_agents_y


def parse_and_save_reddit_file(reddit_filename='reddit_0_train.json', seq_len=10, tokenizer_path='data/ner/vocab.txt', max_client_num=1_000):
    agents_x, agents_y = parse_reddit_file(reddit_filename, seq_len, tokenizer_path)
    agents = [(x, y) for x, y in zip(agents_x, agents_y)]
    pre_filename = reddit_filename.split('.')[0]
    prev_ind, cur_ind = 0, max_client_num
    part = 0
    os.makedirs('data/reddit/bert_clients/', exist_ok=True)
    while True:
        save_agents = agents[prev_ind:min(cur_ind, len(agents))]
        save_to_file(save_agents,
                     'data/reddit/bert_clients/{}_{}SL_{}CN_{}PT.h5'.format(pre_filename, seq_len, max_client_num, part))
        if len(agents) <= cur_ind:
            break
        prev_ind = cur_ind
        cur_ind += max_client_num
        part += 1


def save_to_file(data_clients, filename):
    hf = h5py.File(filename, 'w')
    for i, c in enumerate(data_clients):
        num_examples = len(c[0])
        hf.create_dataset("{}".format(i), data=num_examples)
        hf.create_dataset('{}-input_ids'.format(i), data=[f.input_ids for f in c[0]], dtype=np.int32)
        hf.create_dataset('{}-input_mask'.format(i), data=[f.input_mask for f in c[0]], dtype=np.uint8)
        hf.create_dataset('{}-segment_ids'.format(i), data=[f.segment_ids for f in c[0]], dtype=np.uint8)
        hf.create_dataset('{}-label_id'.format(i), data=[f.label_id for f in c[0]], dtype=np.int32)
        hf.create_dataset('{}-valid_ids'.format(i), data=[f.valid_ids for f in c[0]], dtype=np.uint8)
        hf.create_dataset('{}-label_mask'.format(i), data=[f.label_mask for f in c[0]], dtype=np.uint8)
        hf.create_dataset('{}-y'.format(i), data=c[1])
    hf.close()


def load_from_file(filename):
    data_clients = []
    hf = h5py.File(filename, 'r')
    i = 0
    while hf.get("{}".format(i)):
        num = hf.get("{}".format(i))[()]
        input_ids = hf['{}-input_ids'.format(i)][:]
        input_mask = hf['{}-input_mask'.format(i)][:]
        segment_ids = hf['{}-segment_ids'.format(i)][:]
        label_id = hf['{}-label_id'.format(i)][:]
        valid_ids = hf['{}-valid_ids'.format(i)][:]
        label_mask = hf['{}-label_mask'.format(i)][:]
        y = hf['{}-y'.format(i)][:]

        features = [InputFeatures(input_ids=input_ids[j],
                                  input_mask=input_mask[j],
                                  segment_ids=segment_ids[j],
                                  label_id=label_id[j],
                                  valid_ids=valid_ids[j],
                                  label_mask=label_mask[j]) for j in range(num)]
        data_clients.append([features, y])
        i += 1
    hf.close()
    return data_clients


def load_clients(data_type, client_num, seq_len=10, max_client_num=1_000):
    reddit_index, part = 0, 0
    clients = []

    def parsed_name():
        return 'data/reddit/bert_clients/clients_reddit_{}_{}_{}SL__{}CN_{}PT.h5'\
            .format(reddit_index, data_type, seq_len, max_client_num, part)

    while len(clients) < client_num:
        # print("Loading", parsed_name())
        file_clients = load_from_file(parsed_name())
        part += 1
        if not os.path.exists(parsed_name()):
            reddit_index += 1
            part = 0
        clients.extend(file_clients)
    return clients


def load_client_datasets(num_clients=1_000):
    train = load_clients('train', num_clients)
    val = load_clients('val', num_clients)
    test = load_clients('test', num_clients)
    metadata = []
    for tr, v, ts in zip(train, val, test):
        subreddits = [d.decode() for d in tr[2]] + [d.decode() for d in v[2]] + [d.decode() for d in ts[2]]
        metadata.append(np.unique(subreddits))

    train = [el[:2] for el in train]
    val = [el[:2] for el in val]
    test = [el[:2] for el in test]
    return train, val, test, metadata


def load_clients_data(num_clients=100, starting_client=0):
    tr, val, test, metadata = load_client_datasets(num_clients + starting_client)
    data = {
        "train": tr[starting_client:num_clients + starting_client],
        "val": val[starting_client:num_clients + starting_client],
        "test": test[starting_client:num_clients + starting_client],
        "metadata-subreddits": metadata[starting_client:num_clients + starting_client]
    }
    return data


if __name__ == '__main__':
    parse_and_save_reddit_file('reddit_0_train.json')
    parse_and_save_reddit_file('reddit_0_val.json')
    parse_and_save_reddit_file('reddit_0_test.json')
