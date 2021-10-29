from common import set_seed
from data import clients_data
from fl.federated_averaging import train_fed_avg


def do_train(client_num, num_train_clients, examples_range, batch_size, epochs, client_pars, server_pars, model_v, client_weighting, round_num, seed, accuracy_step):
    tr_cli, val_cli, test_cli = clients_data.filtered_clients(client_num, examples_range)
    set_seed(seed)
    train_fed_avg(tr_cli, val_cli, test_cli, num_train_clients, batch_size, epochs, client_pars, server_pars, model_v, client_weighting, round_num, accuracy_step)

