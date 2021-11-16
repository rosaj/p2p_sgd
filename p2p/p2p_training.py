from p2p.graph_manager import GraphManager
from p2p.p2p_utils import *
from p2p.agent_manager import init_agents
import time


def parse_acc_step(accuracy_step, examples):
    if 'epoch' in accuracy_step:
        accuracy_step = examples
    elif 'iter' in accuracy_step:
        accuracy_step = int(accuracy_step.replace('iter', '').strip() or 1)
    return accuracy_step


def p2p_fn(agents, pbar, devices, start_time):
    possible_a = [i for i in range(len(agents)) if agents[i].can_msg]
    if len(possible_a) == 0:
        print("No agents to train")
        return False
    a_i = possible_a[choose(-1, len(possible_a))]
    agent = agents[a_i]
    clear_session()
    device = resolve_agent_device(agents, agent, devices) or 'CPU'
    with tf.device(device):
        pbar.update(agent.train_fn())
    # update_pb(pbar, agents, start_time)
    return True


def push_sum_fn(agents, pbar, devices, start_time):
    for agent in agents:
        clear_session()
        device = resolve_agent_device(agents, agent, devices)
        with tf.device(device or 'CPU'):
            pbar.update(agent.train_fn())
        # update_pb(pbar, agents, start_time)
    for agent in agents:
        agent.update_local_parameters()


train_fn_class = {
    'P2PAgent': p2p_fn,
    'GoSGDAgent': None,
    'PushSumAgent': push_sum_fn,
}


def train_loop(agent_class, train, val, test, batch_size, model_pars, graph_pars, agent_pars=None, epochs=1,
               accuracy_step='epoch'):
    agents = init_agents(agent_class, train, val, test, batch_size, model_pars, agent_pars)
    graph_manager = GraphManager(nodes=agents, **graph_pars)
    for a in agents:
        a.graph = graph_manager
    graph_manager.print_info()

    start_time = time.time()
    agent_num = len(agents)
    examples = sum([a.train_len for a in agents])

    devices = environ.get_devices()
    # accuracy_step = parse_acc_step(accuracy_step, examples)
    max_examples = epochs * examples
    total_examples, round_num = 0, 0

    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Starting agents')
    for a in agents:
        a.start()
        update_pb(pbar, agents, start_time)
    pbar.close()

    agent_train_fn = train_fn_class[agent_class.__name__]

    pbar = new_progress_bar(examples, 'Training')
    while total_examples < max_examples:

        if agent_train_fn is None:
            for agent in agents:
                clear_session()
                device = resolve_agent_device(agents, agent, devices)
                with tf.device(device or 'CPU'):
                    pbar.update(agent.train_fn())
        else:
            if not agent_train_fn(agents, pbar, devices, start_time):
                break

        if pbar.n >= pbar.total:
            pbar.close()
            graph_manager.check_time_varying(round_num)
            total_examples = sum([a.trained_examples for a in agents])

            round_num += 1
            msg_count = sum([a.hist_total_messages for a in agents])
            print("\nMsgs: {}\tRound: {}\t".format(msg_count, round_num), end='')
            print_all_acc(agents, round(total_examples / examples), False)

            pbar = new_progress_bar(examples, 'Training')

    pbar.close()
    print("Train time: {}".format(time_elapsed_info(start_time)), flush=True)

    filename = "{}_{}A_{}E_{}B_{}V_{}({})_{}N_{}TV".format(
        agent_class.__name__, len(agents), epochs, batch_size, model_pars['v'], graph_manager.graph_type,
        'directed' if graph_manager.directed else 'undirected', graph_manager.num_neighbors, graph_manager.time_varying)
    dump_acc_hist(filename + '.json', agents)
