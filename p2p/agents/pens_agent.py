from p2p.agents.p2p_agent import *
import numpy as np


class PensAgent(P2PAgent):
    def __init__(self, rounds=100, n_sampled=6, top_m=3, n_peers=3, fixed_comm=False, **kwargs):
        super(PensAgent, self).__init__(**kwargs)
        self.rounds = rounds
        self.n_sampled = n_sampled
        self.top_m = top_m
        self.n_peers = n_peers
        self.fixed_comm = fixed_comm
        self.iteration = 0
        self.saved_models = {}
        self.selected_peers = {}

    def train_fn(self):
        self.iteration += 1
        return super(PensAgent, self).train_fn()

    def send_to_peers(self):
        if self.iteration < self.rounds:
            peers = np.random.choice(list(set(self.graph.nodes) - {self}), self.n_sampled, replace=False)
        else:
            expected_samples = (self.top_m / self.graph.nodes_num) * self.rounds
            peers = [k for k, v in self.selected_peers.items() if v > expected_samples]
            peers = np.random.choice(peers, size=self.n_peers, replace=False)
            if self.fixed_comm:
                graph_peers = self.graph.get_peers(self.id)
                # If the comm matrix is built, use that peers in fixed communication
                if len(graph_peers) != 0:
                    peers = graph_peers
        for peer in peers:
            peer.receive_message(self)

    def receive_message(self, other_agent):
        if self.iteration < self.rounds:
            loss = self.eval_model_loss(other_agent.model, self.train)
            self.saved_models[loss] = [other_agent, other_agent.get_model_weights()]
            if len(self.saved_models) == self.n_sampled:
                top_m = list(dict(sorted(self.saved_models.items())).values())[:self.top_m]
                weights = self.get_model_weights()
                for peer, peer_weights in top_m:
                    weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, peer_weights)
                    if peer not in self.selected_peers:
                        self.selected_peers[peer] = 0
                    self.selected_peers[peer] += 1
                # weights = weights_average([self.get_model_weights()] + [peer.get_model_weights() for peer in top_m])

                if self.received_msg:
                    self.send_to_peers()
                self.set_model_weights(weights)
                self.saved_models.clear()
                self.received_msg = True
                self.train_rounds = 1
            return super(P2PAgent, self).receive_message(other_agent)
        else:
            return super(PensAgent, self).receive_message(other_agent)


"""
class PENSAgent(SyncAgent):
    def __init__(self, pens_pars={"rounds": 100, "n_sampled": 6, "top_m": 3, "n_peers": 3, "oracle": False}, **kwargs):
        super(PENSAgent, self).__init__(**kwargs)
        self.rounds = pens_pars.get("rounds", 100)
        self.n_sampled = pens_pars.get("n_sampled", 6)
        self.top_m = pens_pars.get("top_m", 3)
        self.n_peers = pens_pars.get("n_peers", 3)
        self.oracle = pens_pars.get("oracle", False)
        self.iteration = 0
        # self.saved_models = {}
        self.new_weights = None
        self.selected_peers = {}

    def train_fn(self):
        if self.new_weights is not None:
            self.set_model_weights(self.new_weights)
            self.new_weights = None
        self.fit(epochs=1)
        # self.pull_models()
        self.iteration += 1
        return self.train_len

    def get_peers(self):
        if self.oracle:
            peers = [peer for peer in self.graph.nodes if peer.dataset_name == self.dataset_name]
            return np.random.choice(peers, size=self.n_peers, replace=False), False
        if self.iteration < self.rounds:
            return np.random.choice(self.graph.nodes, self.n_sampled, replace=False), True
        else:
            expected_samples = (self.top_m / self.graph.nodes_num) * self.rounds
            peers = [k for k, v in self.selected_peers.items() if v > expected_samples]
            return np.random.choice(peers, size=self.n_peers, replace=False), False

    def pull_models(self):
        peers, is_pens_round = self.get_peers()
        self.hist["useful_msg"][-1] += len(peers)

        if is_pens_round:
            saved_models = {}
            for peer in peers:
                loss = self.eval_model_loss(peer.model, self.train)
                saved_models[loss] = peer

            top_m = list(dict(sorted(saved_models.items())).values())[:self.top_m]
            weights = self.get_model_weights()
            for peer in top_m:
                weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, peer.get_model_weights())
                if peer not in self.selected_peers:
                    self.selected_peers[peer] = 0
                self.selected_peers[peer] += 1

            # weights = weights_average([self.get_model_weights()] + [peer.get_model_weights() for peer in top_m])
            # self.set_model_weights(weights)
            self.new_weights = weights
            saved_models.clear()

        else:
            weights = self.get_model_weights()
            for peer in peers:
                weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, peer.get_model_weights())
            # self.set_model_weights(weights)
            # weights = weights_average([self.get_model_weights()] + [peer.get_model_weights() for peer in peers])
            self.new_weights = weights

    def sync_parameters(self):
        self.pull_models()
"""
