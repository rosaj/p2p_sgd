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


def choose(self_rank, high):
    """
    choose a dest_rank from range(size) to push to

    """

    dest_rank = self_rank

    while dest_rank == self_rank:
        dest_rank = np.random.randint(low=0, high=high)

    return dest_rank


def draw(p):
    """
    draw from Bernoulli distribution

    """
    # Bernoulli distribution is a special case of binomial distribution with n=1
    a_draw = np.random.binomial(n=1, p=p, size=None)

    success = (a_draw == 1)

    return success


def avg_agents_acc(agents, test_data):
    accs = []
    t_accs = []
    for i, a in enumerate(agents):
        accs.append(a.model.evaluate(test_data[i][0], test_data[i][1], verbose=0)[1])
        if a.teacher is not None:
            t_accs.append(a.teacher.evaluate(test_data[i][0], test_data[i][1], verbose=0)[1])
    return np.average(np.array(accs)), np.average(np.array(t_accs)) if len(t_accs) > 0 else 0


def print_accs(accs, info):
    # accs = [ac[0] if len(ac) > 0 else 0 for ac in accs]
    accs = accs or [0]
    if PRINT_MODE == 'SHORT':
        print("{}:\tMean: {:.3%}\tMedian: {:.3%}".format(info, np.average(np.array(accs)), np.median(np.array(accs))))
    else:
        print("{}:\tMean: {:.3%}\tMedian: {:.3%}\t{}".format(info, np.average(np.array(accs)), np.median(np.array(accs)),
                                                             '\t'.join(["{:.3%}".format(i) for i in list(np.array(accs))])))


def print_b_accs(e, agents):
    # accs = [agent.model_val_acc() for agent in agents]
    accs = [agent.val_base_acc for agent in agents]
    print_accs(accs, e)


def print_c_accs(e, agents):
    # accs = [agent.teacher_val_acc() if agent.has_teacher else [0] for agent in agents]
    accs = [agent.val_complex_acc for agent in agents if agent.has_complex]
    print_accs(accs, e)


def print_e_accs(e, agents):
    # accs = [agent.ensemble_val_acc() if agent.has_teacher else [0] for agent in agents]
    accs = [agent.val_ensemble_acc for agent in agents if agent.has_complex]
    print_accs(accs, e)


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
    # print_b_accs("\tB", agents)
    # print_c_accs("\tC", agents)
    # print_e_accs("\tE", agents)
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
    print_f_train_accs("\tTrain", agents)
    if breakdown:
        print_accs([agent.train_base_acc for agent in agents], "\t\tB")
        print_accs([agent.train_complex_acc for agent in agents if agent.has_complex], "\t\tC")
        print_accs([agent.train_ensemble_acc for agent in agents if agent.has_complex], "\t\tE")


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

    if model_pars is None:
        model_pars = {"v": 1, "lr": 0.005, "decay": 0}
    model = create_keras_model(model_v=model_pars['v'], lr=model_pars['lr'], decay=model_pars['decay'])

    x, y = np.array(x), np.array(y)
    for e in range(epochs):
        model.fit(x, y, epochs=1, batch_size=batch_size)
        print("Val: {:.3%}".format(np.average(
            np.array([model.evaluate(val_data[key][0], val_data[key][1], verbose=0) for key in range(len(val_data))])[:, 1])))
        print("Test: {:.3%}".format(np.average(
            np.array([model.evaluate(test_data[key][0], test_data[key][1], verbose=0) for key in range(len(test_data))])[:, 1])))


def set_mode(mode):
    global MODE
    MODE = mode


def init_agents(train_clients,
                val_clients,
                test_clients,
                batch_size,
                complex_ds_size=1000,
                base_pars=None,
                complex_pars=None):

    start_time = time.time()
    base_default = {"v": 1, "lr": 0.01, "decay": 0, "default_weights": False}
    complex_default = {"v": 2, "lr": 0.01, "decay": 0}
    base_pars = base_default if base_pars is None else {**base_default, **base_pars}
    complex_pars = complex_default if complex_pars is None else {**complex_default, **complex_pars}

    print("{} agents, batch size: {}, complex_ds_size: {}, base_pars: {}, complex_pars: {}, mode: {}"
          .format(len(train_clients), batch_size, complex_ds_size, base_pars, complex_pars, MODE))

    if base_pars["default_weights"]:
        base_pars["default_weights"] = create_keras_model(model_v=base_pars["v"], lr=base_pars["lr"], decay=base_pars["decay"]).get_weights()

    pbar = tqdm(total=len(train_clients), position=0, leave=False, desc='Init agents')
    agents = []
    for train, val, test in zip(train_clients, val_clients, test_clients):
        base_model = create_keras_model(model_v=base_pars["v"], lr=base_pars["lr"], decay=base_pars["decay"])
        if base_pars["default_weights"]:
            base_model.set_weights(base_pars["default_weights"])
            print("Setting default weights")

        complex_model = None
        if 0 < complex_ds_size <= len(train[0]):
            complex_model = create_keras_model(model_v=complex_pars["v"], lr=complex_pars["lr"],
                                               decay=complex_pars["decay"])

        a = Agent(train=train,
                  val=val,
                  test=test,
                  batch_size=batch_size,
                  base_model=base_model,
                  complex_model=complex_model
                  )
        agents.append(a)
        if MODE != 'RAM':
            a.serialize()
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

    max_examples = epochs * examples
    total_examples = 0
    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Training')
    msgs = {"total_useful": 0, "total_useless": 0, "useful": 0, "useless": 0}

    devices = environ.get_logical_devices()
    agents_device = list(None for _ in range(len(agents)))
    single_device = len(devices) < 2

    def resolve_agent_device(ag, ind):
        if agents_device[ind] is None:
            for device in devices:
                if available_mb_gpu_memory(device) > ag.memory_footprint * 3:
                    agents_device[ind] = device
                    break

            for i in range(len(agents_device)):
                if agents_device[i] is not None:
                    agents_device[ind] = agents_device[i]
                    agents_device[i] = None
                    agents[i].serialize()
                    agent.deserialize()
                    break
        return agents_device[ind]

    round_num = 0
    num_cached = 0
    while total_examples < max_examples:
        possible_a = [i for i in range(len(agents)) if agents[i].trainable]
        a_i = possible_a[choose(-1, len(possible_a))]
        agent = agents[a_i]
        if agent.trainable:
            pbar.update()
            postfix = memory_info()
            postfix["MEM_AG"] = sum([1 for i in agents_device if i is not None])
            pbar.set_postfix(postfix)

        total_examples += agent.train_len * agent._train_rounds
        if MODE != 'RAM' and agent.base_model is None and single_device:
            agent.deserialize()
            num_cached += 1
        clear_session()

        if single_device:
            neighbors = train_loop_fn(a_i, agent)
        else:
            a_device = resolve_agent_device(agent, a_i)
            with tf.device(a_device):
                neighbors = train_loop_fn(a_i, agent)

        for a_j in neighbors:
            agent_j = agents[a_j]
            if MODE != 'RAM' and agent_j.base_model is None and single_device:
                agent_j.deserialize()
                num_cached += 1
            if single_device:
                if not agent_j.receive_model(agent, mode=share_method, only_improvement=False):
                    msgs["useless"] += 1
                else:
                    msgs["useful"] += 1
            else:
                a_device = resolve_agent_device(agent, a_i)
                with tf.device(a_device):
                    if not agent_j.receive_model(agent, mode=share_method, only_improvement=False):
                        msgs["useless"] += 1
                    else:
                        msgs["useful"] += 1
            if MODE != 'RAM' and NUM_CACHED_AGENTS < num_cached and single_device:
                agent_j.serialize()
                num_cached -= 1
        if MODE != 'RAM' and single_device:
            agent.serialize()
            num_cached -= 1
        num_train = sum([1 for a in agents if a.trainable])
        if pbar.total == pbar.n:
            print("Training:", num_train, "Useful:", msgs["useful"], "Useless:", msgs["useless"], flush=True)
            msgs["total_useful"] += msgs["useful"]
            msgs["useful"] = 0
            msgs["total_useless"] += msgs["useless"]
            msgs["useless"] = 0

            round_num += 1
            print("Round: {}\t".format(round_num), end='')
            print_all_accs(agents, int(total_examples / examples))
            print('', end='', flush=True)
            pbar.close()
            pbar = tqdm(total=len(agents), position=0, leave=False, desc='Training')
        if num_train == 0:
            pbar.close()
            break
    print("Total useful:", msgs["total_useful"], "Total useles's:", msgs["total_useless"], flush=True)
    print("Train agents: {} minutes".format(round((time.time() - start_time)/60)), flush=True)


def train_loop(agents, num_neighbors, epochs, share_method):
    def train_fn(a_i, agent):
        agent.fit()
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
        a._train_rounds = 0
    agents[0]._train_rounds = 1

    def train_fn(a_i, agent):
        agent.fit()
        return get_sample_neighbors(agents, 1, a_i)

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn)
