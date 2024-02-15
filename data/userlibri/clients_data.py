from data.userlibri.data import *
root_path = 'data/userlibri/UserLibri/audio_data/test-clean/'

with tf.io.gfile.GFile('data/userlibri/sp_model_unigram_16K_libri.model', "rb") as f:
    tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)


def parse_cli_dir(cli_dir, test_split=0.3):
    cli_txts = [f for f in os.listdir(root_path + cli_dir) if f.endswith('.txt')]

    cli_data = []

    for txt in cli_txts:
        audios = sorted([a for a in os.listdir(root_path + cli_dir) if txt.split('.')[0] in a and txt != a])
        with open(root_path + cli_dir + '/' + txt, "r") as t:
            desc = t.read()
        rows = desc.strip().split('\n')
        for r in rows:
            f_name = r.split(' ')[0]
            y = r.split(f_name)[-1].strip()
            if f_name + '.flac' in audios:
                # cli_data.append([root_path + cli_dir + '/' + f_name + '.flac', y])
                cli_data.append([f_name + '.flac', y])
            else:
                print(f"File {f_name} not in audios")

    def write_tsv(tsv_data, mode):
        with open(f"{root_path + cli_dir}/{cli_dir}-{mode}.tsv", 'w+') as cli_paths:
            cli_paths.write("FilePath\tText\n")
            for ci, c in enumerate(tsv_data):
                cli_paths.write(f"{c[0]}\t{c[1]}" + ('\n' if ci != len(tsv_data)-1 else ""))

    # tr_ind = random.choices(range(len(cli_data)), k=int(len(cli_data)*(1-test_split)))
    # write_tsv([cli for ci, cli in enumerate(cli_data) if ci in tr_ind], "train")
    # write_tsv([cli for ci, cli in enumerate(cli_data) if ci not in tr_ind], "test")

    write_tsv(cli_data, "all")
    ds = get_dataset(f"{root_path + cli_dir}/{cli_dir}-all.tsv", 'flac', sample_rate=16000, tokenizer=tokenizer)

    np_iter = ds.as_numpy_iterator()
    data_x, data_y = [], []

    while True:
        da = next(np_iter, None)
        if da is None:
            break
        data_x.append(da[0])
        data_y.append(da[1])
    assert len(data_x) == len(data_y) == len(cli_data), "Something went wrong when parsing audios"

    tr_ind = random.choices(range(len(cli_data)), k=int(len(cli_data)*(1-test_split)))
    train = [[x for xi, x in enumerate(data_x) if xi in tr_ind],
             [y for yi, y in enumerate(data_y) if yi in tr_ind]]
    test = [[x for xi, x in enumerate(data_x) if xi not in tr_ind],
           [y for yi, y in enumerate(data_y) if yi not in tr_ind]]

    return train, test


def load_clients_data(num_clients=50):
    assert num_clients <= 55
    data = {
        "train": [],
        "val": [([], []) for _ in range(num_clients)],
        "test": [],
        "dataset_name": ['userlibri'] * num_clients,
    }
    for cli_dir in os.listdir(root_path)[:num_clients]:
        train, test = parse_cli_dir(cli_dir)
        data['train'].append(train)
        data['test'].append(test)
    return data
