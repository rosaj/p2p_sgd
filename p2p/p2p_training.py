from common import *
from p2p.agent import *
import numpy as np
import tensorflow as tf
import os
import time

import environ

if environ.is_collab():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


MODE = 'RAM'
PRINT_MODE = 'SHORT'
NUM_CACHED_AGENTS = 200


def print_accs(accs, info):
    accs = accs or [0]
    accs = np.array(accs)
    if PRINT_MODE == 'SHORT':
        print("{}:\tMean: {:.3%}\tMedian: {:.3%}".format(info, np.average(accs), np.median(accs)))
    else:
        print("{}:\tMean: {:.3%}\tMedian: {:.3%}\t{}".format(info, np.average(accs), np.median(accs), '\t'.join(["{:.3%}".format(i) for i in list(accs)])))


def print_f_val_accs(e, agents):
    accs = [agent.val_ensemble_acc if agent.has_complex else agent.val_base_acc for agent in agents]
    print_accs(accs, e)


def print_f_test_accs(e, agents):
    accs = [agent.test_ensemble_acc if agent.has_complex else agent.test_base_acc for agent in agents]
    print_accs(accs, e)


def print_f_train_accs(e, agents):
    accs = [agent.train_ensemble_acc if agent.has_complex else agent.train_base_acc for agent in agents]
    print_accs(accs, e)


def print_all_accs(agents, e, breakdown=True):
    print("Epoch:", e)
    devices = environ.get_devices()
    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Evaluating agents')
    for a in agents:
        device = resolve_agent_device(agents, a, devices)
        if device is None:
            a.calc_new_accs()
        else:
            with tf.device(device):
                a.calc_new_accs()
        pbar.update()
        pbar.set_postfix(memory_info())
    pbar.close()

    print_f_train_accs("\tTrain", agents)
    if breakdown:
        print_accs([agent.train_base_acc for agent in agents], "\t\tB")
        print_accs([agent.train_complex_acc for agent in agents if agent.has_complex], "\t\tC")
        print_accs([agent.train_ensemble_acc for agent in agents if agent.has_complex], "\t\tE")
    print_f_val_accs("\tVal", agents)
    if breakdown:
        print_accs([agent.val_base_acc for agent in agents], "\t\tB")
        print_accs([agent.val_complex_acc for agent in agents if agent.has_complex], "\t\tC")
        print_accs([agent.val_ensemble_acc for agent in agents if agent.has_complex], "\t\tE")
    print_f_test_accs("\tTest", agents)
    if breakdown:
        print_accs([agent.test_base_acc for agent in agents], "\t\tB")
        print_accs([agent.test_complex_acc for agent in agents if agent.has_complex], "\t\tC")
        print_accs([agent.test_ensemble_acc for agent in agents if agent.has_complex], "\t\tE")


def get_sample_neighbors(agents, num_clients, self_ind):
    client_num = len(agents)
    c_list = list(range(client_num))
    c_list.remove(self_ind)
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]


def single_model(train_data, val_data, test_data, model_pars=None, batch_size=50, epochs=20):
    print("Training single model")
    x, y = [], []
    for tx, ty in train_data:
        x.extend(tx)
        y.extend(ty)
    print("Examples:", len(y))
    vx, vy = [], []
    for val_x, val_y in val_data:
        vx.extend(val_x)
        vy.extend(val_y)

    if model_pars is None:
        model_pars = {"v": 1, "lr": 0.005, "decay": 0}
    model = create_keras_model(model_v=model_pars['v'], lr=model_pars['lr'], decay=model_pars['decay'])

    x, y = np.array(x), np.array(y)
    for e in range(epochs):
        model.fit(x, y, epochs=1, batch_size=batch_size, validation_data=(np.array(vx), np.array(vy)))
        print("Val: {:.3%}".format(np.average(
            np.array([model.evaluate(val_data[key][0], val_data[key][1], verbose=0) for key in range(len(val_data))])[:, 1])))
        print("Test: {:.3%}".format(np.average(
            np.array([model.evaluate(test_data[key][0], test_data[key][1], verbose=0) for key in range(len(test_data))])[:, 1])))


def set_mode(mode):
    global MODE
    MODE = mode


def resolve_agent_device(agents, agent, devices):
    if len(devices) == 0:
        return None
    if agent is None:
        free_mem, mem_dev = 0, ''
        for device in devices:
            available_mem = available_device_memory(device)
            if available_mem > free_mem:
                free_mem = available_mem
                mem_dev = device
        return mem_dev

    if agent.device is None:
        free_mem, mem_dev = 0, ''
        for device in devices:
            available_mem = available_device_memory(device)
            if available_mem > free_mem:
                free_mem = available_mem
                mem_dev = device

        agent.device = mem_dev
        with tf.device(agent.device):
            agent.deserialize()
        if available_device_memory(agent.device) > agent.memory_footprint * 3:
            return agent.device

        for a in agents:
            if a != agent and a.device == agent.device:
                with tf.device(a.device):
                    a.serialize()
                    a.device = None
                break
    return agent.device


def init_agents(train_clients,
                val_clients,
                test_clients,
                batch_size,
                complex_ds_size=1000,
                base_pars=None,
                complex_pars=None):

    start_time = time.time()
    base_default = {"v": 1, "lr": 0.01, "decay": 0, "default_weights": False}
    complex_default = {"v": 2, "lr": 0.01, "decay": 0, "default_weights": False}
    base_pars = base_default if base_pars is None else {**base_default, **base_pars}
    complex_pars = complex_default if complex_pars is None else {**complex_default, **complex_pars}

    num_agents = len(train_clients)
    print("{} agents, batch size: {}, complex_ds_size: {}, base_pars: {}, complex_pars: {}, mode: {}"
          .format(num_agents, batch_size, complex_ds_size, base_pars, complex_pars, MODE))

    if base_pars["default_weights"]:
        base_pars["default_weights"] = create_keras_model(model_v=base_pars["v"], lr=base_pars["lr"], decay=base_pars["decay"]).get_weights()

    if complex_pars["default_weights"]:
        complex_pars["default_weights"] = create_keras_model(model_v=complex_pars["v"], lr=complex_pars["lr"], decay=complex_pars["decay"]).get_weights()

    pbar = tqdm(total=num_agents, position=0, leave=False, desc='Init agents')
    devices = environ.get_devices()
    agents = []

    def create_agent():
        clear_session()
        base_model = create_keras_model(model_v=base_pars["v"], lr=base_pars["lr"], decay=base_pars["decay"])
        if base_pars["default_weights"]:
            base_model.set_weights(base_pars["default_weights"])
            # print("Setting default weights")

        complex_model = None
        if 0 < complex_ds_size <= len(train[0]):
            complex_model = create_keras_model(model_v=complex_pars["v"], lr=complex_pars["lr"],
                                               decay=complex_pars["decay"])
            if complex_pars["default_weights"]:
                complex_model.set_weights(complex_pars["default_weights"])

        a = Agent(train=train,
                  val=val,
                  test=test,
                  batch_size=batch_size,
                  base_model=base_model,
                  complex_model=complex_model
                  )
        a.device = device
        agents.append(a)
        if MODE != 'RAM':
            a.serialize()

    for train, val, test in zip(train_clients, val_clients, test_clients):
        device = resolve_agent_device(agents, None, devices)
        if device is None:
            create_agent()
        else:
            with tf.device(device):
                create_agent()
        pbar.update()
        pbar.set_postfix(memory_info())
    pbar.close()
    print("Init agents: {} minutes".format(round((time.time() - start_time)/60)))
    environ.save_env_vars()
    return agents


def load_agents(train_clients,
                val_clients,
                test_clients,
                batch_size,
                complex_ds_size,
                first_agent_id):

    environ.set_agent_id(first_agent_id)
    agents = []
    for train, val, test in zip(train_clients, val_clients, test_clients):
        complex_model = True if 0 < complex_ds_size <= len(train[0]) else None
        a = Agent(train=train,
                  val=val,
                  test=test,
                  batch_size=batch_size,
                  base_model=None,
                  complex_model=complex_model
                  )
        agents.append(a)
    return agents


def abstract_train_loop(agents, num_neighbors, epochs, share_method, train_loop_fn):

    start_time = time.time()
    examples = sum([a.train_len for a in agents])
    print("Training {} agents, num neighbors: {}, examples: {}, share method: {}".format(len(agents), num_neighbors, examples, share_method), flush=True)

    devices = environ.get_devices()
    single_device = len(devices) < 2

    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Pretraining')
    for i, a in enumerate(agents):
        a.train_rounds = 1
        a.fit()
        # a.train_rounds = 1
        a.can_msg = True
        pbar.update()
    pbar.close()
    # print_all_accs(agents, 0)

    max_examples = epochs * examples
    total_examples, round_num, num_cached = 0, 0, 0
    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Training')
    msgs = {"total_useful": 0, "total_useless": 0, "useful": 0, "useless": 0}

    while total_examples < max_examples:
        possible_a = [i for i in range(len(agents)) if agents[i].can_msg]

        if len(possible_a) == 0:
            print("No agents to train")
            break
            # possible_a.append(sorted(agents, key=lambda a: a.updates, reverse=True)[0])
        a_i = possible_a[choose(-1, len(possible_a))]
        agent = agents[a_i]

        pbar.update()
        postfix = memory_info()
        for dev in devices:
            postfix["N_{}".format(dev)] = sum([1 for a in agents if a.device == dev])
        pbar.set_postfix(postfix)

        total_examples += agent.train_len * agent.train_rounds
        if MODE != 'RAM' and agent.base_model is None and single_device:
            agent.deserialize()
            num_cached += 1
        clear_session()

        device = resolve_agent_device(agents, agent, devices)
        if device is None:
            neighbors = train_loop_fn(a_i, agent)
        else:
            with tf.device(device):
                neighbors = train_loop_fn(a_i, agent)

        for a_j in neighbors:
            agent_j = agents[a_j]
            if MODE != 'RAM' and agent_j.base_model is None and single_device:
                agent_j.deserialize()
                num_cached += 1

            device_j = resolve_agent_device(agents, agent_j, devices)
            if device_j is None:
                msg = agent_j.receive_model(agent, share_method)
                agent_j.can_msg = True
            else:
                with tf.device(device_j):
                    msg = agent_j.receive_model(agent, share_method)
                agent_j.can_msg = True

            msgs["useful" if msg else "useless"] += 1
            if MODE != 'RAM' and NUM_CACHED_AGENTS < num_cached and single_device:
                agent_j.serialize()
                num_cached -= 1

        agent.can_msg = False

        if MODE != 'RAM' and single_device:
            agent.serialize()
            num_cached -= 1
        num_train = sum([1 for a in agents if a.trainable])
        if pbar.total == pbar.n:
            pbar.close()
            print("Training:", num_train, "Useful:", msgs["useful"], "Useless:", msgs["useless"], flush=True)
            msgs["total_useful"] += msgs["useful"]
            msgs["useful"] = 0
            msgs["total_useless"] += msgs["useless"]
            msgs["useless"] = 0

            round_num += 1
            print("Round: {}\t".format(round_num), end='')
            print_all_accs(agents, int(total_examples / examples))
            print('', end='', flush=True)

            if MODE != 'RAM' and round_num % 10 == 0:
                for live_agent in agents:
                    if live_agent.device is not None:
                        with tf.device(live_agent.device):
                            live_agent.serialize(True)
            pbar = tqdm(total=len(agents), position=0, leave=False, desc='Training')

    print("Total useful:", msgs["total_useful"], "Total useles's:", msgs["total_useless"], flush=True)
    print("Train agents: {} minutes".format(round((time.time() - start_time)/60)), flush=True)


def train_loop(agents, num_neighbors, epochs, share_method):
    def train_fn(a_i, agent):
        agent.fit()
        possible_a = [i for i in range(len(agents)) if not agents[i].can_msg]
        if a_i in possible_a:
            possible_a.remove(a_i)
        if len(possible_a) >= num_neighbors:
            random_indices = np.random.choice(len(possible_a), size=num_neighbors, replace=False)
            return np.array(possible_a)[random_indices]
        return get_sample_neighbors(agents, num_neighbors, a_i)

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn)


def train_fixed_neighbors(agents, num_neighbors, epochs, share_method):
    neighbors = [get_sample_neighbors(agents, num_neighbors, a_i) for a_i in range(len(agents))]

    def train_fn(a_i, agent):
        agent.fit()
        return neighbors[a_i]

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn)


def train_loopy(agents, num_neighbors, epochs, share_method):
    for a in agents:
        a.train_rounds = 0
    agents[0].train_rounds = 1

    def train_fn(a_i, agent):
        agent.fit()
        return get_sample_neighbors(agents, 1, a_i)

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn)
