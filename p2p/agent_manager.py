from p2p.p2p_utils import *
import time


def init_agents(agent_class, train_clients, val_clients, test_clients, batch_size, model_pars=None):
    start_time = time.time()
    model_default = {"v": 1, "lr": 0.001, "decay": 0, "default_weights": False}
    model_pars = model_default if model_pars is None else {**model_default, **model_pars}

    num_agents = len(train_clients)
    print("{} agents, batch size: {}, model_pars: {}".format(num_agents, batch_size, model_pars))

    if model_pars["default_weights"]:
        model_pars["default_weights"] = create_keras_model(model_v=model_pars["v"], lr=model_pars["lr"],
                                                           decay=model_pars["decay"]).get_weights()

    pbar = tqdm(total=num_agents, position=0, leave=False, desc='Init agents')
    devices = environ.get_devices()
    agents = []

    for train, val, test in zip(train_clients, val_clients, test_clients):
        device = resolve_agent_device(agents, None, devices)
        with tf.device(device or 'CPU'):
            clear_session()
            a_model = create_keras_model(model_v=model_pars["v"], lr=model_pars["lr"], decay=model_pars["decay"])
            if model_pars["default_weights"]:
                a_model.set_weights(model_pars["default_weights"])

            a = agent_class(
                train=train,
                val=val,
                test=test,
                batch_size=batch_size,
                model=a_model,
            )
            a.device = device
            agents.append(a)
        update_pb(pbar, agents, start_time)
    pbar.close()
    print("Init agents: {} minutes".format(round((time.time() - start_time) / 60)))
    environ.save_env_vars()
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
