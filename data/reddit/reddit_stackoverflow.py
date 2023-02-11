from collections import Counter
from data.reddit.preparation import load_reddit_json, parse_json_agents as parse_reddit_json_agents
from data.stackoverflow.preparation import load_stackoverflow_json, parse_json_agents as parse_so_json_agents

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle


def create_tokenizer(vocab_size=10_000):
    c = Counter()
    j_data = load_reddit_json('data/reddit/source/data/reddit_leaf/train/reddit_0_train.json')
    agents, _ = parse_reddit_json_agents(j_data)
    c.update(' '.join([' '.join(a) for a in agents]).split())

    j_data = load_stackoverflow_json('data/stackoverflow/users/stackoverflow_0.json')
    agents = parse_so_json_agents(j_data)
    c.update(''.join([''.join(a) for a in agents]).split())

    words = np.array(c.most_common(vocab_size))[:, 0]
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(words)

    pickle.dump(tokenizer, open('data/reddit/reddit_stackoverflow_tokenizer.pkl', 'wb'))
    return tokenizer


if __name__ == '__main__':
    create_tokenizer()
