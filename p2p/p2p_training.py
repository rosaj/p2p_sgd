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
                private_ds_size=1000,
                shared_pars=None,
                private_pars=None):

    start_time = time.time()
    shared_default = {"v": 1, "lr": 0.01, "decay": 0, "default_weights": False}
    private_default = {"v": 2, "lr": 0.01, "decay": 0, "default_weights": False}
    shared_pars = shared_default if shared_pars is None else {**shared_default, **shared_pars}
    private_pars = private_default if private_pars is None else {**private_default, **private_pars}

    num_agents = len(train_clients)
    print("{} agents, batch size: {}, private_ds_size: {}, shared_pars: {}, private_pars: {}, mode: {}"
          .format(num_agents, batch_size, private_ds_size, shared_pars, private_pars, MODE))

    global FILE_NAME
    FILE_NAME = FILE_NAME + '_{}B_{}CDS_{}Vb_{}Vc'.format(batch_size, private_ds_size, shared_pars['v'], private_pars['v'])

    if shared_pars["default_weights"]:
        shared_pars["default_weights"] = create_keras_model(model_v=shared_pars["v"], lr=shared_pars["lr"], decay=shared_pars["decay"]).get_weights()

    if private_pars["default_weights"]:
        private_pars["default_weights"] = create_keras_model(model_v=private_pars["v"], lr=private_pars["lr"], decay=private_pars["decay"]).get_weights()

    pbar = tqdm(total=num_agents, position=0, leave=False, desc='Init agents')
    devices = environ.get_devices()
    agents = []

    def create_agent():
        clear_session()
        shared_model = create_keras_model(model_v=shared_pars["v"], lr=shared_pars["lr"], decay=shared_pars["decay"])
        if shared_pars["default_weights"]:
            shared_model.set_weights(shared_pars["default_weights"])
            # print("Setting default weights")

        private_model = None
        if 0 < private_ds_size <= len(train[0]):
            private_model = create_keras_model(model_v=private_pars["v"], lr=private_pars["lr"],
                                               decay=private_pars["decay"])
            if private_pars["default_weights"]:
                private_model.set_weights(private_pars["default_weights"])

        a = Agent(train=train,
                  val=val,
                  test=test,
                  batch_size=batch_size,
                  shared_model=shared_model,
                  private_model=private_model
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
                private_ds_size,
                first_agent_id):

    environ.set_agent_id(first_agent_id)
    agents = []
    for train, val, test in zip(train_clients, val_clients, test_clients):
        private_model = True if 0 < private_ds_size <= len(train[0]) else None
        a = Agent(train=train,
                  val=val,
                  test=test,
                  batch_size=batch_size,
                  shared_model=None,
                  private_model=private_model
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
        if MODE != 'RAM' and agent.shared_model is None and single_device:
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
            if MODE != 'RAM' and agent_j.shared_model is None and single_device:
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
        return possible_a + list(get_sample_neighbors(agents, num_neighbors - len(possible_a), a_i))

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn, accuracy_step)


def train_fixed_neighbors(agents, num_neighbors, epochs, share_method, accuracy_step):
    neighbors = [get_sample_neighbors(agents, num_neighbors, a_i) for a_i in range(len(agents))]

    def train_fn(a_i, agent):
        agent.fit()
        return neighbors[a_i]

    abstract_train_loop(agents, num_neighbors, epochs, share_method, train_fn, accuracy_step)


def get_sample_neighbors(agents, num_clients, self_ind):
    client_num = len(agents)
    c_list = list(range(client_num))
    c_list.remove(self_ind)
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]
