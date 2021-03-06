from p2p.graph_manager import GraphManager
from p2p.p2p_utils import *
from p2p.agent_manager import init_agents
from p2p.agents import SyncAgent

from datetime import datetime
import time


def print_pars_info(**kwargs):
    for k, v in kwargs.items():
        print(k+":", v)


def parse_acc_step(accuracy_step, examples):
    if 'epoch' in accuracy_step:
        accuracy_step = examples
    elif 'iter' in accuracy_step:
        accuracy_step = int(accuracy_step.replace('iter', '').strip() or 1)
    return accuracy_step


def train_loop(agent_pars, agent_data_pars, model_pars, graph_pars, sim_pars):
    print_pars_info(agent_pars=agent_pars, agent_data_pars=agent_data_pars,
                    model_pars=model_pars, graph_pars=graph_pars, sim_pars=sim_pars)
    set_seed(sim_pars.get('seed'))

    agents = init_agents(agent_pars, agent_data_pars, model_pars)
    graph_manager = GraphManager(nodes=agents, **graph_pars)
    for a in agents:
        a.graph = graph_manager

    start_time = time.time()
    examples = sum([a.train_len for a in agents])

    devices = environ.get_devices()
    # accuracy_step = parse_acc_step(accuracy_step, examples)
    epochs = sim_pars.get('epochs', 1)
    agent_class = agent_pars['agent_class']
    max_examples = epochs * examples
    total_examples, round_num = 0, 0

    pbar = new_progress_bar(examples, 'Training')
    for a in agents:
        device = resolve_agent_device(agents, a, devices)
        with tf.device(device or 'CPU'):
            n = a.start()
        update_pb(pbar, agents, n, start_time)
    _, pbar, round_num, total_examples = checkpoint(pbar, agents, round_num, examples, total_examples)

    while total_examples < max_examples:
        if issubclass(agent_class, SyncAgent):
            for agent in agents:
                clear_session()
                device = resolve_agent_device(agents, agent, devices)
                with tf.device(device or 'CPU'):
                    pbar.update(agent.train_fn())
            for agent in agents:
                agent.sync_parameters()
        else:
            possible_a = [a for a in agents if a.can_be_awaken()]
            if len(possible_a) == 0:
                print("No agents to train")
                break
            agent = np.random.choice(possible_a, 1)[0]
            clear_session()
            device = resolve_agent_device(agents, agent, devices)
            with tf.device(device or 'CPU'):
                pbar.update(agent.train_fn())

        is_check, pbar, round_num, total_examples = checkpoint(pbar, agents, round_num, examples, total_examples)
        if is_check:
            graph_manager.check_time_varying(round_num)

    pbar.close()
    print("Train time: {}".format(time_elapsed_info(start_time)), flush=True)

    filename = "{}_{}A_{}E_{}B_{}_{}".format(
        agent_class.__name__, len(agents), epochs, agent_data_pars['batch_size'],
        graph_pars['graph_type'] + '(' + ('' if graph_pars['directed'] else 'un') + 'directed-' + str(graph_pars['num_neighbors']) + ')',
        datetime.now().strftime("%d-%m-%Y_%H_%M"))

    sim_pars['sim_time'] = time.time() - start_time
    dump_acc_hist('log/' + filename + '.json',
                  agents,
                  graph_manager.as_numpy_array(),
                  {'agent_pars': agent_pars,
                   'agent_data_pars': agent_data_pars,
                   'model_pars': model_pars,
                   'graph_pars': graph_pars,
                   'sim_pars': sim_pars})


def checkpoint(pbar, agents, round_num, examples, total_examples):
    if pbar.n >= pbar.total:
        diff = pbar.n - pbar.total
        pbar.close()
        total_examples = sum([a.trained_examples for a in agents])

        round_num += 1
        msg_count = sum([a.hist_total_messages for a in agents])
        print("\nMsgs: {}\tRound: {}\t".format(msg_count, round_num), end='')
        calc_agents_metrics(agents, round(total_examples / examples))

        pbar = new_progress_bar(examples, 'Training')
        pbar.update(diff)
        return True, pbar, round_num, total_examples
    return False, pbar, round_num, total_examples
