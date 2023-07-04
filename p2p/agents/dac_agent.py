from p2p.agents.p2p_agent import *
import numpy as np


def softmax_scale(x, tau):
    x_new = np.exp(x * tau) / sum(np.exp(x * tau))
    return x_new


def tau_function(x, a, b):
    tau = 2 * a / (1 + np.exp(-b * x)) - a + 1
    return tau


class DacAgent(P2PAgent):
    def __init__(self, tau=1, n_sampled=3, **kwargs):
        super(DacAgent, self).__init__(**kwargs)
        self.tau = tau
        self.n_sampled = n_sampled
        self.saved_models = {}
        self.selected_peers = None
        self.priors = None

    def start(self):
        self.selected_peers = np.zeros(self.graph.nodes_num)
        self.priors = np.zeros(self.graph.nodes_num)
        return super(DacAgent, self).start()

    @property
    def priors_norm(self):
        if sum(self.priors) == 0:
            priors = np.ones(self.graph.nodes_num) / self.graph.nodes_num
            priors[self.id] = 0.0
            return priors / np.sum(priors)

        not_i_idx = np.arange(len(self.priors)) != self.id
        return softmax_scale(self.priors[not_i_idx], self.tau)

    def send_to_peers(self):
        pn = self.priors_norm[list(set(range(self.graph.nodes_num)) - {self.id})]
        peers = np.random.choice(list(set(self.graph.nodes) - {self}), self.n_sampled, replace=False, p=pn)
        for peer in peers:
            peer.receive_message(self)

    def receive_message(self, other_agent):
        loss = self.eval_model_loss(other_agent.model, self.train)
        self.saved_models[loss] = [other_agent, other_agent.get_model_weights()]
        if len(self.saved_models) == self.n_sampled:
            weights = self.get_model_weights()
            for peer_loss, (peer, peer_weights) in self.saved_models.items():
                weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, weights, peer_weights)

                # We want to receive more messages from this peer so mark us as selected peer
                peer.selected_peers[self.id] += 1
                self.priors[peer.id] = 1 / peer_loss

            # Two-hop neighbor priors estimation
            neighbour_list = np.arange(self.graph.nodes_num)
            new_neighbours = []
            for peer_n, _ in self.saved_models.values():
                new_neighbours += list(set(neighbour_list[peer_n.priors > 0])
                                       - set(neighbour_list[self.selected_peers > 0])
                                       - {self.id})
            new_neighbours = np.unique(new_neighbours)
            for j in new_neighbours:
                prior_j = 0
                for k, (peer_k, _) in enumerate(self.saved_models.values()):
                    score_kj = self.graph.nodes[peer_k.id].priors[j]
                    prior_j = max(prior_j, score_kj)
                self.priors[j] = prior_j

            # weights = weights_average([self.get_model_weights()] + [peer.get_model_weights() for peer in top_m])

            if self.received_msg:
                self.send_to_peers()
            self.set_model_weights(weights)
            self.saved_models.clear()
            self.received_msg = True
            self.train_rounds = 1
        return super(P2PAgent, self).receive_message(other_agent)
