from p2p.agents.p2p_agent import *


class Msg:
    def __init__(self, cid, x, w, k):
        self.cid = cid
        self.x = x
        self.w = w
        self.k = k


class OverlapSGDAgent(Agent):
    def __init__(self, T=2, **kwargs):
        super(OverlapSGDAgent, self).__init__(**kwargs)

        self.T = T
        self.w = 1
        self.k = 0
        self.count_since_last = 0
        self.x_grads = self.model.trainable_variables

        self.msgs = []
        self.make_train_iter()

    def train_batch(self):
        x, y = self.next_train_batch()
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
            z_grads = tape.gradient(loss, self.model.trainable_variables)

        self.x_grads = tf.nest.map_structure(lambda x_par, z_g: x_par - self.model.optimizer.learning_rate * z_g,
                                             self.x_grads, z_grads)
        if self.k % self.T == 0:
            self.send_to_peers()
            pii = self.graph.get_self_node_weight(self.id)
            self.x_grads = tf.nest.map_structure(lambda o: o * pii, self.x_grads)
            self.w = pii * self.w

        self.clear_received_buffer(len(self.graph.get_peers()))
        self.k += 1
        self.count_since_last += 1

        return len(y)

    def can_send(self):
        return self.k % self.T == 0

    def send(self, neighbors):
        self.send_local_gradient_to_neighbor(neighbors)
        pii = 1 / (len(neighbors) + 1)
        self.x_grads = tf.nest.map_structure(lambda o: o * pii, self.x_grads)
        self.w = pii * self.w
        return len(neighbors)

    def can_train(self):
        return self.count_since_last < T

    def received_msgs(self):
        return len(self.msgs)

    def send_to_peers(self):
        peers, weights = self.graph.get_weighted_peers(self.id)
        for peer, pji in zip(peers, weights):
            c_w = tf.nest.map_structure(lambda o: o * pji, self.x_grads)
            peer.msg_q.append(Msg(self.id, c_w, self.w * pji, self.k))
            peer.receive_message(self)

    def clear_received_buffer(self, num_neighbors):
        if self.received_msgs() >= num_neighbors:
            self.update_local_parameters()

    def blocked(self, num_neighbors):
        return not self.can_train() and self.received_msgs() < num_neighbors

    def update_local_parameters(self):
        for msg in self.msgs:
            self.x_grads = tf.nest.map_structure(lambda xi, xj: xi + xj, self.x_grads, msg.x)
            self.w = msg.w

        for stv, xtv in zip(self.model.trainable_variables,
                            tf.nest.map_structure(lambda xi: xi / self.w, self.x_grads)):
            stv.assign(xtv)

        self.msgs = []
        self.count_since_last = 0
