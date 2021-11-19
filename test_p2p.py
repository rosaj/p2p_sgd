from p2p.train import do_train
from p2p.agents import P2PAgent, GoSGDAgent, SGPPushSumAgent, D2Agent

if __name__ == '__main__':
    do_train(D2Agent,
             num_clients=50,
             batch_size=50,
             model_pars={"model_v": 4, "lr": 0.005, "default_weights": True},
             agent_pars=None,
             graph_pars={'graph_type': 'sparse', 'num_neighbors': 2, 'directed': True, 'time_varying': 1},
             epochs=40, accuracy_step='epoch')

# /Users/robert/.local/share/virtualenvs/p2p_sgd-oreLDV97/bin/python3.7 test_p2p.py > log/p2p_test.txt
