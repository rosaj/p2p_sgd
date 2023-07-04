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

                    # We want to receive more messages from this peer so mark us as selected peer
                    if self not in peer.selected_peers:
                        peer.selected_peers[self] = 0
                    peer.selected_peers[self] += 1
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

