from p2p.p2p_training import train_loop
from data import clients_data
from p2p.agents import P2PAgent, GoSGDAgent, PushSumAgent

if __name__ == '__main__':
    train, val, test = clients_data.filtered_clients(client_num=50, examples_range=(0, 900000))
    train_loop(P2PAgent,
               train, val, test, 50,
               {"v": 4, "lr": 0.005, "decay": 0, "default_weights": True},
               agent_pars=None,
               graph_pars={'graph_type': 'sparse', 'num_neighbors': 2, 'directed': True, 'time_varying': -1},
               epochs=40, accuracy_step='epoch')


"""
if __name__ == '__main__':
    do_train(client_num=50,
             num_neighbors=2,
             examples_range=(0, 90000000),
             batch_size=50,
             private_ds_size=-1,
             model_pars={"v": 4, "lr": 0.005, "decay": 0, "default_weights": True},
             private_pars={"v": 1, "lr": 0.005, "decay": 0, "default_weights": True},
             seed=123,
             share_method='average',
             epochs=60,
             accuracy_step='epoch',
             resume_agent_id=-1
             )"""

# /Users/robert/.local/share/virtualenvs/p2p_sgd-oreLDV97/bin/python3.7 test_p2p.py > log/p2p_test.txt
