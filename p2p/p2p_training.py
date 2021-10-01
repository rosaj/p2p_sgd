from p2p.agent import *
from p2p.p2p_utils import *
import time

MODE = 'RAM'
NUM_CACHED_AGENTS = 200
FILE_NAME = 'log/p2p_{}'


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
    complex_default = {"v": 2, "lr": 0.01, "decay": 0, "default_weights": False}
    base_pars = base_default if base_pars is None else {**base_default, **base_pars}
    complex_pars = complex_default if complex_pars is None else {**complex_default, **complex_pars}

    num_agents = len(train_clients)
    print("{} agents, batch size: {}, complex_ds_size: {}, base_pars: {}, complex_pars: {}, mode: {}"
          .format(num_agents, batch_size, complex_ds_size, base_pars, complex_pars, MODE))

    global FILE_NAME
    FILE_NAME = FILE_NAME + '_{}B_{}CDS_{}Vb_{}Vc'.format(batch_size, complex_ds_size, base_pars['v'], complex_pars['v'])

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


def abstract_train_loop(agents, num_neighbors, epochs, share_method, train_loop_fn, accuracy_step='epoch'):

    start_time = time.time()
    examples = sum([a.train_len for a in agents])
    print("Training {} agents, num neighbors: {}, examples: {}, share method: {}".format(len(agents), num_neighbors, examples, share_method), flush=True)

    devices = environ.get_devices()
    single_device = len(devices) < 2
    max_examples = epochs * examples
    total_examples, round_num, num_cached = 0, 0, 0

    if 'epoch' in accuracy_step:
        accuracy_step = len(agents)
    elif 'iter' in accuracy_step:
        accuracy_step = int(accuracy_step.replace('iter', '').strip() or 1)

    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Pretraining')
    for i, a in enumerate(agents):
        total_examples += a.train_len * a.train_rounds
        a.fit()
        a.can_msg = True
        pbar.update()
    pbar.close()

    pbar = tqdm(total=accuracy_step, position=0, leave=False, desc='Training')
    while total_examples < max_examples:
        possible_a = [i for i in range(len(agents)) if agents[i].can_msg]

        if len(possible_a) == 0:
            print("No agents to train")
            break
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
                agent_j.receive_model(agent, share_method)
            else:
                with tf.device(device_j):
                    agent_j.receive_model(agent, share_method)

            if MODE != 'RAM' and NUM_CACHED_AGENTS < num_cached and single_device:
                agent_j.serialize()
                num_cached -= 1

        agent.can_msg = False

        if MODE != 'RAM' and single_device:
            agent.serialize()
            num_cached -= 1

        if pbar.total == pbar.n:
            pbar.close()
            print("Training:", sum([1 for a in agents if a.trainable]))

            round_num += 1
            print("Round: {}\t".format(round_num), end='')
            print_all_acc(agents, round(total_examples / examples))

            if MODE != 'RAM' and round_num % 10 == 0:
                for live_agent in agents:
                    if live_agent.device is not None:
                        with tf.device(live_agent.device):
                            live_agent.serialize(True)
            pbar = tqdm(total=accuracy_step, position=0, leave=False, desc='Training')

    pbar.close()
    print("Train agents: {} minutes".format(round((time.time() - start_time)/60)), flush=True)

    global FILE_NAME
    FILE_NAME = FILE_NAME.format("{}A_{}N_{}E_{}".format(len(agents), num_neighbors, epochs, share_method)) + '.json'
    dump_acc_hist(FILE_NAME, agents)


def train_loop(agents, num_neighbors, epochs, share_method, accuracy_step):
    def train_fn(a_i, agent):
        agent.fit()
        possible_a = [i for i in range(len(agents)) if not agents[i].can_msg]
        if a_i in possible_a:
            possible_a.remove(a_i)
        if len(possible_a) >= num_neighbors:
            random_indices = np.random.choice(len(possible_a), size=num_neighbors, replace=False)
            return np.array(possible_a)[random_indices]
        return get_sample_neighbors(agents, num_neighbors, a_i)

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn, accuracy_step)


def train_fixed_neighbors(agents, num_neighbors, epochs, share_method, accuracy_step):
    neighbors = [get_sample_neighbors(agents, num_neighbors, a_i) for a_i in range(len(agents))]

    def train_fn(a_i, agent):
        agent.fit()
        return neighbors[a_i]

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn, accuracy_step)


def train_loopy(agents, num_neighbors, epochs, share_method, accuracy_step):
    for a in agents:
        a.train_rounds = 0
    agents[0].train_rounds = 1

    def train_fn(a_i, agent):
        agent.fit()
        return get_sample_neighbors(agents, 1, a_i)

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn, accuracy_step)


def get_sample_neighbors(agents, num_clients, self_ind):
    client_num = len(agents)
    c_list = list(range(client_num))
    c_list.remove(self_ind)
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]
