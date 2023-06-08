from p2p.agents.p2p_agent import *
import numpy as np


class PENSAgent(P2PAgent):
    def __init__(self, pens_pars={"rounds": 100, "n_sampled": 6, "top_m": 3, "n_peers": 3}, **kwargs):
        if 'early_stopping' not in kwargs:
            kwargs['early_stopping'] = False
        super(PENSAgent, self).__init__(**kwargs)
        self.pens_pars = pens_pars
        self.iteration = 0
        self.saved_models = {}
        self.selected_peers = {}

    def send_to_peers(self):
        if self.iteration < self.pens_pars['rounds']:
            peers = np.random.choice(self.graph.nodes, self.pens_pars['n_sampled'], replace=False)
            for peer in peers:
                peer.receive_message(self)
        else:
            expected_samples = (self.pens_pars['top_m']/self.graph.nodes_num) * self.pens_pars['rounds']
            peers = [k for k, v in self.selected_peers if v > expected_samples]
            peers = np.random.choice(peers, size=self.pens_pars['n_peers'], replace=False)
            for p_id in peers:
                self.graph.get_node(p_id).receive_message(self)

        self.iteration += 1

    def receive_message(self, other_agent):
        super(P2PAgent, self).receive_message(other_agent)

        if self.iteration < self.pens_pars['rounds']:
            metrics = self.eval_model(other_agent.model, self.train)
            print("METRICS-----", metrics)
            self.saved_models[metrics['loss']] = [other_agent, other_agent.get_model_weights()]

            if len(self.saved_models) >= self.pens_pars['n_sampled']:
                top_m = list(dict(sorted(self.saved_models.items())).values())[:self.pens_pars['top_m']]

                weights = self.get_model_weights()
                for peer, m in top_m:
                    weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, m)
                    peer.selected_peers[self.id] += 1

                self.set_model_weights(weights)
                self.saved_models.clear()

                self.received_msg = True
                self.train_rounds = 1
        else:
            self.saved_models[other_agent.id] = [other_agent, other_agent.get_model_weights()]

            if len(self.saved_models) >= self.pens_pars['n_peers']:
                weights = self.get_model_weights()
                for peer, m in list(self.saved_models.values()):
                    weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, m)

                self.set_model_weights(weights)
                self.saved_models.clear()

                self.received_msg = True
                self.train_rounds = 1





