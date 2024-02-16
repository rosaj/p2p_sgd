from data.userlibri.data import *
import h5py
import numpy as np

root_path = 'data/userlibri/UserLibri/audio_data/test-clean/'


def get_tokenizer():
    with tf.io.gfile.GFile('data/userlibri/sp_model_unigram_16K_libri.model', "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    return tokenizer


def save_to_file(data_clients, filename):
    hf = h5py.File(filename, 'w')

    dt = h5py.vlen_dtype(np.dtype('float32'))
    max_len = max([len(c[0]) for c in data_clients])
    x = hf.create_dataset('x', (max_len,), dtype=dt)

    for i, c in enumerate(data_clients):
        num_examples = len(c[0])
        hf.create_dataset("{}".format(i), data=num_examples)
        # hf.create_dataset('{}-x'.format(i), data=c[0], dtype=np.float32)
        x[i] = c[0]
        hf.create_dataset('{}-y'.format(i), data=c[1])
    hf.close()


def load_from_file(filename):
    data_clients = []
    hf = h5py.File(filename, 'r')
    i = 0
    while hf.get("{}".format(i)):
        x = hf['{}-x'.format(i)][:]
        y = hf['{}-y'.format(i)][:]
        data_clients.append([x, y])
        i += 1
    hf.close()
    return data_clients


def parse_cli_dir(cli_dir, tokenizer):
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

    write_tsv(cli_data, "all")
    ds = get_dataset(f"{root_path + cli_dir}/{cli_dir}-all.tsv", 'flac', sample_rate=16000, tokenizer=tokenizer)
    ds = ds.map(make_log_mel_spectrogram(
                sample_rate=16000,
                frame_length=320,
                frame_step=160,
                fft_length=320,
                num_mel_bins=80,
                lower_edge_hertz=80.0,
                upper_edge_hertz=7600.0,
                ), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    slice_fn = slice_example(2048, 128)
    ds = ds.apply(slice_fn)

    def make_example(audio: tf.Tensor, tokens: tf.Tensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        return audio, (tokens[:-1], tokens[1:])
    ds = ds.map(make_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(1, (([2048, 80, 1]), ([127], [127])))

    # ds = ds.padded_batch(1, ([2048, 80, 1], [128]))
    np_iter = ds.as_numpy_iterator()
    data_x, data_y = [], []
    while True:
        da = next(np_iter, None)
        if da is None:
            break
        # data_x.append(np.pad(da[0], ((2048,), (80,), (1,))))
        # data_y.append(np.pad(da[1], 128))
        # data_x.append([np.reshape(da[0][0], (2048, 80, 1)), np.reshape(da[0][1], (127,))])
        data_x.append(np.reshape(da[0], (2048, 80, 1)))
        data_y.append((np.reshape(da[1][0], (127,)), np.reshape(da[1][0], (127,))))
    assert len(data_x) == len(data_y) == len(cli_data), "Something went wrong when parsing audios"
    """
    tr_ind = random.choices(range(len(cli_data)), k=int(len(cli_data)*(1-test_split)))
    train = [[x for xi, x in enumerate(data_x) if xi in tr_ind],
             [y for yi, y in enumerate(data_y) if yi in tr_ind]]
    test = [[x for xi, x in enumerate(data_x) if xi not in tr_ind],
           [y for yi, y in enumerate(data_y) if yi not in tr_ind]]
    """

    return data_x, data_y


def parse_clients():
    tokenizer = get_tokenizer()
    clients = []
    for cli_dir in os.listdir(root_path):
        x, y = parse_cli_dir(cli_dir, tokenizer)
        clients.append((x, y))
    save_to_file(clients, 'clients.h5')


def load_clients_data(num_clients=50, test_split=0.3):
    assert num_clients <= 55
    data = {
        "train": [],
        "val": [([], []) for _ in range(num_clients)],
        "test": [],
        "dataset_name": ['userlibri'] * num_clients,
    }
    # clients = load_from_file('clients.h5')[:num_clients]
    tokenizer = get_tokenizer()
    for ci, cli_dir in enumerate(os.listdir(root_path)[:num_clients]):
        print(ci, cli_dir)
        x, y, = parse_cli_dir(cli_dir, tokenizer)
        tr_ind = random.choices(range(len(x)), k=int(len(x)*(1-test_split)))
        train = ([x for xi, x in enumerate(x) if xi in tr_ind],
                 [y for yi, y in enumerate(y) if yi in tr_ind])
        test = ([x for xi, x in enumerate(x) if xi not in tr_ind],
                [y for yi, y in enumerate(y) if yi not in tr_ind])

        data['train'].append(train)
        data['test'].append(test)

    return data


def post_process_dataset(tf_dataset, data_pars):
    if next(iter(tf_dataset), None) is None:
        return tf_dataset

    def make_example(audio: tf.Tensor, tokens: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        # print(audio.shape, tokens[0].shape, tokens[1].shape)
        return (audio, tokens[0]), tokens[1]
    tf_dataset = tf_dataset.map(make_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    padded_shape = (([2048, 80, 1], [127]), [127])
    tf_dataset = tf_dataset.shuffle(data_pars['batch_size']).tf_dataset.padded_batch(data_pars['batch_size'], padded_shape)
    # tf_dataset = tf_dataset.padded_batch(data_pars['batch_size'], (([2048, 80, 1], [127]), [127]))
    return tf_dataset
