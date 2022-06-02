from p2p.train import do_train
from p2p.agents import P2PAgent
"""
from data.reddit import clients_data
from common.nwp import model
do_train(P2PAgent,
         clients_data,
         num_clients=5,
         batch_size=50,
         model_pars={"model_mod": model, "model_v": 3, "lr": 0.005, "default_weights": True},
         agent_pars=None,
         graph_pars={'graph_type': 'sparse', 'num_neighbors': 3, 'directed': True, 'time_varying': -1},
         epochs=100, seed=123,
         starting_client=0)

"""


from data.ner import clients_data
from common.ner.bert import model

do_train(P2PAgent,
         clients_data_pars={"clients_data": clients_data, "num_clients": 5, "batch_size": 50},
         model_pars={"model_mod": model, "bert_config": 'uncased_L-2_H-128_A-2', "lr": 5e-5, "default_weights": True},
         agent_pars=None,
         graph_pars={'graph_type': 'sparse', 'num_neighbors': 3, 'directed': True, 'time_varying': -1},
         epochs=100, seed=123,
         )
# """
