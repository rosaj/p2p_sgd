from data.reddit.preparation import process_agent_data, load_reddit_json, parse_json_agents, Tokenizer
from data.reddit.clients_data import load_client_datasets
from collections import Counter
import numpy as np


def parse_per_subreddit(seq_len=10, max_client_num=1_000, vocab_size=10_002, categories=['gaming', 'politics']):
    train_json_data = load_reddit_json('data/reddit/source/data/reddit_leaf/train/reddit_0_train.json')
    val_json_data = load_reddit_json('data/reddit/source/data/reddit_leaf/val/reddit_0_val.json')
    test_json_data = load_reddit_json('data/reddit/source/data/reddit_leaf/test/reddit_0_test.json')

    train_agents_x, train_agents_y = parse_json_agents(train_json_data)
    val_agents_x, val_agents_y = parse_json_agents(val_json_data)
    test_agents_x, test_agents_y = parse_json_agents(test_json_data)

    indices = []
    for i, (train_y, val_y, test_y) in enumerate(zip(train_agents_y, val_agents_y, test_agents_y)):
        c_train = [cat in train_y for cat in categories]
        if sum(c_train) == 1:
            check_cat = np.array(categories)[np.logical_not(np.array(c_train))]
            s_val = sum([cat in val_y for cat in check_cat])
            s_test = sum([cat in test_y for cat in check_cat])
            if s_val == 0 and s_test == 0:
                indices.append(i)

    print("Len train", len(train_agents_x), "Len indices:", len(indices))
    train_agents_x, train_agents_y = [train_agents_x[i] for i in indices], [train_agents_y[i] for i in indices]
    val_agents_x, val_agents_y = [val_agents_x[i] for i in indices], [val_agents_y[i] for i in indices]
    test_agents_x, test_agents_y = [test_agents_x[i] for i in indices], [test_agents_y[i] for i in indices]

    c = Counter()
    c.update(' '.join([' '.join(a) for a in train_agents_x]).split())
    words = np.array(c.most_common(vocab_size))[:, 0]
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(words)

    process_agent_data(train_agents_x, train_agents_y, tokenizer, seq_len, vocab_size, max_client_num, pre_filename='clients_reddit_0_train', directory='clients_category/' + '_'.join(categories))
    process_agent_data(val_agents_x, val_agents_y, tokenizer, seq_len, vocab_size, max_client_num, pre_filename='clients_reddit_0_val', directory='clients_category/' + '_'.join(categories))
    process_agent_data(test_agents_x, test_agents_y, tokenizer, seq_len, vocab_size, max_client_num, pre_filename='clients_reddit_0_test', directory='clients_category/' + '_'.join(categories))

    for cat in categories:
        indices = [i for i in range(len(train_agents_y)) if cat in train_agents_y[i]]
        train_x, train_y = [train_agents_x[i] for i in indices], [train_agents_y[i] for i in indices]

        c = Counter()
        c.update(' '.join([' '.join(a) for a in train_x]).split())
        words = np.array(c.most_common(vocab_size))[:, 0]
        tokenizer = Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(words)

        process_agent_data(train_x, train_y, tokenizer, seq_len, vocab_size, max_client_num, pre_filename='clients_reddit_0_train', directory='clients_category/{}'.format(cat))
        val_x, val_y = [val_agents_x[i] for i in indices], [val_agents_y[i] for i in indices]
        process_agent_data(val_x, val_y, tokenizer, seq_len, vocab_size, max_client_num, pre_filename='clients_reddit_0_val', directory='clients_category/{}'.format(cat))
        test_x, test_y = [test_agents_x[i] for i in indices], [test_agents_y[i] for i in indices]
        process_agent_data(test_x, test_y, tokenizer, seq_len, vocab_size, max_client_num, pre_filename='clients_reddit_0_test', directory='clients_category/{}'.format(cat))


def load_clients_data(num_clients=100, seed=608361, train_examples_range=(700, 20_000), categories=['gaming', 'politics']):
    if not isinstance(num_clients, list) and not isinstance(num_clients, tuple):
        num_clients = int(num_clients / len(categories))
        num_clients = [num_clients] * len(categories)

    train, val, test, metadata = load_client_datasets(num_clients=3_500, directory='clients_category/' + '_'.join(categories))
    indices = []
    for cat, n_cli in zip(categories, num_clients):
        choices = [i for i, tr in enumerate(train) if train_examples_range[0] <= len(tr[0]) <= train_examples_range[1]
                   and cat in metadata[i]]
        if seed is not None:
            from numpy.random import MT19937
            from numpy.random import RandomState, SeedSequence
            rs = RandomState(MT19937(SeedSequence(seed)))
            clients_ids = rs.choice(choices, size=n_cli, replace=False)
        else:
            clients_ids = np.random.choice(choices, size=n_cli, replace=False)
        indices.append(clients_ids)

    data = {
        "train": [],
        "val": [],
        "test": [],
        "metadata-subreddits": [],
        "dataset_name": []
    }

    for cli_ind in indices:
        data["train"].extend([el for ei, el in enumerate(train) if ei in cli_ind])
        data["val"].extend([el for ei, el in enumerate(val) if ei in cli_ind])
        data["test"].extend([el for ei, el in enumerate(test) if ei in cli_ind])
        metadata_subreddits = [el for ei, el in enumerate(metadata) if ei in cli_ind]
        data["metadata-subreddits"].extend(metadata_subreddits)
        dataset_names = ['reddit-nwp-{}'.format(np.array(categories)[np.array([cat in mt_sr for cat in categories])][0]) for mt_sr in metadata_subreddits]
        data["dataset_name"].extend(dataset_names)

    """
    metadata_subreddits = [el for ei, el in enumerate(metadata) if ei in indices]
    data = {
        "train": [el for ei, el in enumerate(train) if ei in indices],
        "val": [el for ei, el in enumerate(val) if ei in indices],
        "test": [el for ei, el in enumerate(test) if ei in indices],
        "metadata-subreddits": metadata_subreddits,
        "dataset_name": ['reddit-nwp-{}'.format(np.array(categories)[np.array([cat in mt_sr for cat in categories])][0]) for mt_sr in metadata_subreddits]
        # "dataset_name": ['reddit-nwp'] * len(indices)
    }
    """
    return data


if __name__ == '__main__':
    parse_per_subreddit()
