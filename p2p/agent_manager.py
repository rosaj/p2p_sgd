from p2p.p2p_utils import *
import time


def init_agents(agent_class, clients_data_pars, model_pars=None, agent_pars=None):
    start_time = time.time()

    agent_pars = agent_pars or {}
    train_clients, val_clients, test_clients = clients_data_pars['clients_data']\
        .load_clients(**{k: v for k, v in clients_data_pars.items() if k not in ['clients_data', 'batch_size']})
    batch_size = clients_data_pars['batch_size']

    num_agents = len(train_clients)
    print("{}: {} agents, batch size: {}, model_pars: {}, agent_pars: {}".format(
        agent_class.__name__, num_agents, batch_size, model_pars, agent_pars))

    # model_pars['model_mod'].clear_def_weights_cache()
    clear_def_weights_cache()

    pbar = tqdm(total=num_agents, position=0, leave=False, desc='Init agents')
    devices = environ.get_devices()
    agents = []

    for agent_id, (train, val, test) in enumerate(zip(train_clients, val_clients, test_clients)):
        device = resolve_agent_device(agents, None, devices)
        with tf.device(device or 'CPU'):
            clear_session()
            agent_pars['train'] = train
            agent_pars['val'] = val
            agent_pars['test'] = test
            agent_pars['data_pars'] = clients_data_pars
            # agent_pars['batch_size'] = batch_size
            # agent_pars['model'] = model_mod.create_model(**m_pars)
            agent_pars['model'] = model_pars
            a = agent_class(**agent_pars)
            a.device = device
            a.id = agent_id
            agents.append(a)
        update_pb(pbar, agents, 1, start_time)
    pbar.close()
    print("Init agents: {} minutes".format(round((time.time() - start_time) / 60)))
    # environ.save_env_vars()
    return agents


def load_agents(agent_class, train_clients, val_clients, test_clients, batch_size, first_agent_id):
    environ.set_agent_id(first_agent_id)
    agents = []
    for train, val, test in zip(train_clients, val_clients, test_clients):
        a = agent_class(train=train,
                        val=val,
                        test=test,
                        batch_size=batch_size,
                        model=None,
                        )
        agents.append(a)
    return agents
