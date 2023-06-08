from p2p.agents.sync_agent import *
import numpy as np


class PENSAgent(SyncAgent):
    def __init__(self, pens_pars={"rounds": 100, "n_sampled": 6, "top_m": 3, "n_peers": 3}, **kwargs):
        if 'early_stopping' not in kwargs:
            kwargs['early_stopping'] = False
        super(PENSAgent, self).__init__(**kwargs)
        self.pens_pars = pens_pars
        self.iteration = 0
        # self.saved_models = {}
        self.new_weights = None
        self.selected_peers = {}

    def train_fn(self):
        self.fit()
        self.pull_models()
        self.iteration += 1
        return self.train_len

    def get_peers(self):
        if self.iteration < self.pens_pars['rounds']:
            return np.random.choice(self.graph.nodes, self.pens_pars['n_sampled'], replace=False), True
        else:
            expected_samples = (self.pens_pars['top_m'] / self.graph.nodes_num) * self.pens_pars['rounds']
            peers = [k for k, v in self.selected_peers if v > expected_samples]
            return np.random.choice(peers, size=self.pens_pars['n_peers'], replace=False), False

    def pull_models(self):
        peers, is_pens_round = self.get_peers()

        if is_pens_round:
            saved_models = {}
            for peer in peers:
                loss = self.eval_model_loss(peer.model, self.train)
                saved_models[loss] = peer.id
            top_m = list(dict(sorted(saved_models.items())).values())[:self.pens_pars['top_m']]
            weights = self.get_model_weights()
            for peer in top_m:
                weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, peer.get_model_weights())
                if peer.id not in self.selected_peers:
                    self.selected_peers[peer.id] = 0
                self.selected_peers[peer.id] += 1

            # self.set_model_weights(weights)
            self.new_weights = weights
            saved_models.clear()

        else:
            weights = self.get_model_weights()
            for peer in peers:
                weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, peer.get_model_weights())
            # self.set_model_weights(weights)
            self.new_weights = weights

    def sync_parameters(self):
        self.set_model_weights(self.new_weights)

"""
    def send_to_peers(self):
        if self.iteration < self.pens_pars['rounds']:
            peers = np.random.choice(self.graph.nodes, self.pens_pars['n_sampled'], replace=False)
            for peer in peers:
                peer.receive_message(self)
        else:
            expected_samples = (self.pens_pars['top_m'] / self.graph.nodes_num) * self.pens_pars['rounds']
            peers = [k for k, v in self.selected_peers if v > expected_samples]
            peers = np.random.choice(peers, size=self.pens_pars['n_peers'], replace=False)
            for p_id in peers:
                self.graph.get_node(p_id).receive_message(self)

        self.iteration += 1

    def receive_message(self, other_agent):
        super(SyncAgent, self).receive_message(other_agent)

        if self.iteration < self.pens_pars['rounds']:
            loss = self.eval_model_loss(other_agent.model, self.train)
            self.saved_models[loss] = [other_agent, other_agent.get_model_weights()]

            if len(self.saved_models) >= self.pens_pars['n_sampled']:
                top_m = list(dict(sorted(self.saved_models.items())).values())[:self.pens_pars['top_m']]

                weights = self.get_model_weights()
                for peer, m in top_m:
                    weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, m)
                    if self.id not in peer.selected_peers:
                        peer.selected_peers[self.id] = 0
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
    """
