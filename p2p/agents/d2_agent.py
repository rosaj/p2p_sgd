from p2p.agents.abstract_agent import *

# D2: Decentralized Training over Decentralized Data
# Authors: Tang, Hanlin
#          Lian, Xiangru
#          Yan, Ming
#          Zhang, Ce
#          Liu, Ji

# Desired graph topology: Ring, Sum(Wij) = 1


class D2Agent(Agent):
    def __init__(self, **kwargs):
        super(D2Agent, self).__init__(**kwargs)

        self.t_1_weights = self.get_model_weights()

        self.make_train_iter()
        self.msg_q = []

    def start(self):
        x, y = self.next_train_batch()
        return self._train_on_batch(x, y)

    def _compute_current_gradient(self):
        x, y = self.next_train_batch()
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        z_grads = tape.gradient(loss, self.model.trainable_variables)

        self.trained_examples += len(y)
        return len(y), z_grads

    def train_fn(self):
        trained_examples, z_grads = self._compute_current_gradient()

        self.set_model_weights(tf.nest.map_structure(lambda w: w * 2, self.get_model_weights()))
        self.model.optimizer.apply_gradients(zip(z_grads, self.model.trainable_variables))

        self.set_model_weights(tf.nest.map_structure(lambda x, xt: x - xt, self.get_model_weights(), self.t_1_weights))

        self.send_to_peers()
        return trained_examples

    def send_to_peers(self):
        # Should be undirected graph, so wij and wji are identical
        peers = self.graph.get_peers(self.id)
        for peer in peers:
            peer.receive_message(self)

    def receive_message(self, other_agent):
        super(D2Agent, self).receive_message(other_agent)
        wji = self.graph.get_edge_weight(self.id, other_agent.id)
        w_xj_t2 = tf.nest.map_structure(lambda xj_t2: xj_t2 * wji, other_agent.get_model_weights())
        self.msg_q.append(w_xj_t2)

    def update_local_parameters(self):
        wii = self.graph.get_edge_weight(self.id, self.id)
        xi_tw = tf.nest.map_structure(lambda xi: xi * wii, self.get_model_weights())
        self.msg_q.append(xi_tw)

        self.t_1_weights = np.sum(np.array(self.msg_q, dtype=object), axis=0)
        self.set_model_weights(self.t_1_weights)

        self.msg_q.clear()

