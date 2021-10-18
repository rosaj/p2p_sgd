from common import *
import numpy as np
import tensorflow as tf
import environ


def print_acc(accs, info):
    accs = np.array(accs or [0])
    print("{}:\tMean: {:.3%}\tMedian: {:.3%}".format(info, np.average(accs), np.median(accs)))


def print_all_acc(agents, e, breakdown=True):
    print("Epoch:", e)
    devices = environ.get_devices()
    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Evaluating agents')
    for a in agents:
        device = resolve_agent_device(agents, a, devices)
        if device is None:
            a.calc_new_acc()
        else:
            with tf.device(device):
                a.calc_new_acc()
        pbar.update()
        pbar.set_postfix(memory_info())
    pbar.close()

    print_acc([agent.train_ensemble_acc if agent.has_private else agent.train_shared_acc for agent in agents], "\tTrain")
    if breakdown:
        print_acc([agent.train_shared_acc for agent in agents], "\t\tS")
        print_acc([agent.train_private_acc for agent in agents if agent.has_private], "\t\tP")
        print_acc([agent.train_ensemble_acc for agent in agents if agent.has_private], "\t\tE")
    print_acc([agent.val_ensemble_acc if agent.has_private else agent.val_shared_acc for agent in agents], "\tVal")
    if breakdown:
        print_acc([agent.val_shared_acc for agent in agents], "\t\tS")
        print_acc([agent.val_private_acc for agent in agents if agent.has_private], "\t\tP")
        print_acc([agent.val_ensemble_acc for agent in agents if agent.has_private], "\t\tE")
    print_acc([agent.test_ensemble_acc if agent.has_private else agent.test_shared_acc for agent in agents], "\tTest")
    if breakdown:
        print_acc([agent.test_shared_acc for agent in agents], "\t\tS")
        print_acc([agent.test_private_acc for agent in agents if agent.has_private], "\t\tP")
        print_acc([agent.test_ensemble_acc for agent in agents if agent.has_private], "\t\tE")
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


def dump_acc_hist(filename, agents):
    save_json(filename,  {a.id: a.hist for a in agents})


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
        print('Epoch: {}'.format(e + 1))
        val_acc = np.array(
            [model.evaluate(val_data[key][0], val_data[key][1], verbose=0) for key in range(len(val_data))])[:, 1]
        print("\tVal\n\t\tMean: {:.3%}\tMedian: {:.3%}".format(np.average(val_acc), np.median(val_acc)))
        test_acc = np.array(
            [model.evaluate(test_data[key][0], test_data[key][1], verbose=0) for key in range(len(test_data))])[:, 1]
        print("\tTest\n\t\tMean: {:.3%}\tMedian: {:.3%}".format(np.average(test_acc), np.median(test_acc)))
