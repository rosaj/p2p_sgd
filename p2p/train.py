from p2p import init_agents, load_agents, train_loop, train_fixed_neighbors, single_model, set_mode
from data import clients_data
from common import set_seed


def do_train(client_num, num_neighbors, examples_range,
             batch_size, private_ds_size, shared_pars, private_pars,
             seed, mode, share_method, epochs, accuracy_step='epoch', resume_agent_id=-1):

    train_cli, val_cli, test_cli = clients_data.filtered_clients(client_num, examples_range)

    if num_neighbors < 1:
        set_seed(seed)
        single_model(train_cli, val_cli, test_cli, shared_pars, batch_size, epochs)
    else:
        set_mode(mode)
        if resume_agent_id < 0:
            set_seed(seed)
            agents = init_agents(train_cli, val_cli, test_cli, batch_size, private_ds_size, shared_pars, private_pars)
        else:
            agents = load_agents(train_cli, val_cli, test_cli, batch_size, private_ds_size, resume_agent_id)
        train_loop(agents, num_neighbors, epochs, share_method, accuracy_step)
