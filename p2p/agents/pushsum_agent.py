from p2p.agents.abstract_agent import *


class Msg:
    def __init__(self, cid, x, w):
        self.cid = cid
        self.x = x
        self.w = w


class PushSumAgent(Agent):
    def __init__(self, **kwargs):
        super(PushSumAgent, self).__init__(**kwargs)

        self.w = 1
        self.x_grads = self.model.trainable_variables

        self.msg_q = []
        self.make_train_iter()

    def train_fn(self):
        x, y = self.next_train_batch()
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
            z_grads = tape.gradient(loss, self.model.trainable_variables)

        self.x_grads = tf.nest.map_structure(lambda x_par, z_g: x_par - self.model.optimizer.learning_rate * z_g,
                                             self.x_grads, z_grads)
        self.trained_examples += len(y)

        self.send_to_peers()

    def send_to_peers(self):
        peers, weights = self.graph.get_weighted_peers(self.id)
        for peer, pji in zip(peers, weights):
            c_x = tf.nest.map_structure(lambda o: o * pji, self.x_grads)
            peer.msg_q.append(Msg(self.id, c_x, self.w * pji))
            peer.receive_message(self)

    def update_local_parameters(self):
        # update x
        pij = 1 / (len(self.msg_q) + 1)
        x = tf.nest.map_structure(lambda o: o * pij, self.x_grads)

        for msg in self.msg_q:
            x = tf.nest.map_structure(lambda pwj, xj: pwj + xj, msg.x, x)
        self.x_grads = x

        # update w
        self.w *= pij
        for msg in self.msg_q:
            self.w += msg.w
        # print(self.id, self.w)

        for stv, xtv in zip(self.model.trainable_variables,
                            tf.nest.map_structure(lambda xi: xi / self.w, self.x_grads)):
            stv.assign(xtv)

        self.msg_q.clear()
