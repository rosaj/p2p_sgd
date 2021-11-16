from common import *
import numpy as np
import tensorflow as tf
import environ


def print_acc(accs, info):
    accs = np.array(accs or [0])
    print("{}:\tMean: {:.3%}\tMedian: {:.3%}".format(info, np.average(accs), np.median(accs)))


def print_all_acc(agents, e, breakdown=False):
    print("Epoch:", e)
    devices = environ.get_devices()
    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Evaluating agents')
    for a in agents:
        device = resolve_agent_device(agents, a, devices)
        with tf.device(device or 'CPU'):
            a.calc_new_metrics()
        update_pb(pbar, agents)
    pbar.close()

    print_acc([agent.hist_train_model_metric for agent in agents], "\tTrain")
    print_acc([agent.hist_val_model_metric for agent in agents], "\tVal")
    print_acc([agent.hist_test_model_metric for agent in agents], "\tTest")

    """
    print_acc([agent.hist_train_ensemble_metric if agent.has_private else agent.hist_train_model_metric for agent in agents], "\tTrain")
    if breakdown:
        print_acc([agent.hist_train_model_metric for agent in agents], "\t\tS")
        print_acc([agent.hist_train_private_metric for agent in agents if agent.has_private], "\t\tP")
        print_acc([agent.hist_train_ensemble_metric for agent in agents if agent.has_private], "\t\tE")
    print_acc([agent.hist_val_ensemble_metric if agent.has_private else agent.hist_val_model_metric for agent in agents], "\tVal")
    if breakdown:
        print_acc([agent.hist_val_model_metric for agent in agents], "\t\tS")
        print_acc([agent.hist_val_private_metric for agent in agents if agent.has_private], "\t\tP")
        print_acc([agent.hist_val_ensemble_metric for agent in agents if agent.has_private], "\t\tE")
    print_acc([agent.hist_test_ensemble_metric if agent.has_private else agent.hist_test_model_metric for agent in agents], "\tTest")
    if breakdown:
        print_acc([agent.hist_test_model_metric for agent in agents], "\t\tS")
        print_acc([agent.hist_test_private_metric for agent in agents if agent.has_private], "\t\tP")
        print_acc([agent.hist_test_ensemble_metric for agent in agents if agent.has_private], "\t\tE")
    """
    print('', end='', flush=True)


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


def update_pb(pbar, agents, start_time=None):
    pbar.update()
    devices = environ.get_devices()
    postfix = memory_info()
    for dev in devices:
        postfix["N_{}".format(dev)] = sum([1 for a in agents if a.device == dev])
    if start_time:
        postfix["TP"] = time_elapsed_info(start_time)
    pbar.set_postfix(postfix)


def dump_acc_hist(filename, agents):
    save_json(filename,  {a.id: a.hist for a in agents})



