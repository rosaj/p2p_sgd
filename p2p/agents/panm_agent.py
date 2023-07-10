from p2p.agents.sync_agent import *
from models.abstract_model import weights_average
import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel(GaussianMixture):
    def __init__(self, initial_resp=None, **kwargs):
        super(GaussianMixtureModel, self).__init__(**kwargs)
        self.initial_resp = initial_resp

    def _initialize_parameters(self, X, random_state):
        if self.initial_resp is None:
            super(GaussianMixtureModel, self)._initialize_parameters(X, random_state)
        else:
            self._initialize(X, np.array(self.initial_resp).astype(np.float64))


def calc_vectorized_updates(weights, initial_weights):
    w_delta = tf.nest.map_structure(lambda wi, wo: wi-wo, weights, initial_weights)
    w = tf.concat([tf.reshape(w, (-1,)) for w in w_delta], axis=0)
    return tf.reshape(w, (1, -1))


class PanmAgent(SyncAgent):
    def __init__(self, method='loss', rounds=100, top_m=3, n_sampled=6, theta=10, alpha=0.5, **kwargs):
        super(PanmAgent, self).__init__(**kwargs)
        assert method in ['loss', 'grad']
        self.method = method
        self.alpha = alpha
        self.top_m = top_m
        self.n_sampled = n_sampled
        self.rounds = rounds
        self.theta = theta
        self.initial_weights = None
        self.new_weights = None
        self.similar_peers = None
        self.iteration = 0
        self.previous_peers = []
        self.neighbor_bag = []

    def start(self):
        self.initial_weights = self.get_model_weights()
        self.new_weights = self.initial_weights
        self.similar_peers = np.zeros(self.graph.nodes_num)
        return super(PanmAgent, self).start()

    def train_fn(self):
        self.set_model_weights(self.new_weights)
        self.iteration += 1
        return super(PanmAgent, self).train_fn()

    def calc_similarity(self, peer):
        if self.method == 'loss':
            loss = self.eval_model_loss(peer.model, self.train)
            return 1 / loss
        elif self.method == 'grad':

            gi = calc_vectorized_updates(self.get_model_weights(), self.new_weights)
            gj = calc_vectorized_updates(peer.get_model_weights(), peer.new_weights)
            cos_1 = tf.keras.losses.cosine_similarity(gi, gj, axis=1)

            hi = calc_vectorized_updates(self.get_model_weights(), self.initial_weights)
            hj = calc_vectorized_updates(peer.get_model_weights(), self.initial_weights)
            cos_2 = tf.keras.losses.cosine_similarity(hi, hj, axis=1)

            return float((cos_1*self.alpha + (1-self.alpha)*cos_2).numpy())

        raise ValueError(f"Invalid method {self.method}")

    def nsmc_selection(self):
        p = np.arange(self.graph.nodes_num)
        p = p[list(set(p).difference([self.id]+self.previous_peers))]
        indx = np.random.choice(p, self.n_sampled-len(self.previous_peers), replace=False)
        peers = [p for p in self.graph.nodes if p.id in indx or p.id in self.previous_peers]

        for peer in peers:
            super(PanmAgent, self).receive_message(peer)

        saved_models = {self.calc_similarity(p): p for p in peers}
        for k, v in saved_models.items():
            self.similar_peers[v.id] = k
        peers = list(dict(sorted(saved_models.items())).values())[-self.top_m:]
        self.previous_peers = [p.id for p in peers]
        return peers

    def naem_selection(self):
        if len(self.neighbor_bag) == 0:
            self.neighbor_bag = self.previous_peers

        if self.iteration % self.theta == 0:
            p = np.arange(self.graph.nodes_num)
            p = p[list(set(p).difference([self.id]+self.neighbor_bag))]
            indx = np.random.choice(p, self.n_sampled, replace=False)

            if len(self.neighbor_bag) < self.n_sampled:
                selected_peers = self.neighbor_bag
            else:
                selected_peers = np.random.choice(self.neighbor_bag, self.n_sampled, replace=False)

            combined_ind = list(indx) + list(selected_peers)
            peers = [p for p in self.graph.nodes if p.id in combined_ind]
            for peer in peers:
                super(PanmAgent, self).receive_message(peer)

            sims = [[self.calc_similarity(self.graph.nodes[p_id])] for p_id in combined_ind]
            gm = GaussianMixtureModel(n_components=2, initial_resp=[[1, 0] for _ in indx] + [[0, 1] for _ in selected_peers])
            labels = gm.fit_predict(X=sims)
            sims = np.array(sims)
            h_label = np.argsort(np.array([np.squeeze(sims[np.argwhere(labels == 0)]).sum(),
                                           np.squeeze(sims[np.argwhere(labels == 1)]).sum()]))[-1]
            h_peers = list(np.array(combined_ind)[np.squeeze(np.argwhere(labels == h_label))])
            self.neighbor_bag = list((set(self.neighbor_bag) - set(selected_peers)).union(set(h_peers)))

        n_i = np.random.choice(self.neighbor_bag, min(self.top_m, len(self.neighbor_bag)), replace=False)
        peers = [p for p in self.graph.nodes if p.id in n_i]
        return peers

    def pull_from_peers(self):
        if self.iteration < self.rounds:
            peers = self.nsmc_selection()
        else:
            peers = self.naem_selection()

        alphas = [self.train_len] + [peer.train_len for peer in peers]
        ws = [self.get_model_weights()] + [peer.get_model_weights() for peer in peers]
        self.new_weights = weights_average(ws, alphas)

    def sync_parameters(self):
        self.pull_from_peers()

    def update_parameters(self):
        pass

