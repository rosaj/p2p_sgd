from p2p.p2p_utils import *
import time


def init_agents(agent_pars, agent_data_pars, model_pars=None):
    start_time = time.time()

    train_clients, val_clients, test_clients = agent_data_pars['agents_data']\
        .load_clients_data(**{k: v for k, v in agent_data_pars.items() if k not in ['agents_data', 'batch_size']})
    num_agents = len(train_clients)

    clear_def_weights_cache()

    pbar = tqdm(total=num_agents, position=0, leave=False, desc='Init agents')
    devices = environ.get_devices()
    agents = []
    agent_class = agent_pars['agent_class']

    for agent_id, (train, val, test) in enumerate(zip(train_clients, val_clients, test_clients)):
        device = resolve_agent_device(agents, None, devices)
        with tf.device(device or 'CPU'):
            clear_session()

            a_p = {k: v for k, v in agent_pars.items() if k not in ['agent_class']}
            a_p['train'], a_p['val'], a_p['test'] = train, val, test
            a_p['data_pars'], a_p['model'] = agent_data_pars, model_pars

            a = agent_class(**a_p)
            a.device = device
            a.id = agent_id
            agents.append(a)
        update_pb(pbar, agents, 1, start_time)
    pbar.close()
    print("Init agents: {} minutes".format(round((time.time() - start_time) / 60)))
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
