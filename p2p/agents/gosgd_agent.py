from p2p.agents.abstract_agent import *
from common.util import choose, draw

# Distributed optimization for deep learning with gossip exchange
# Authors: Blot, Michael
#          Picard, David
#          Thome, Nicolas
#          Cord, Matthieu

# Graph topology: No graph topologies, agent communicates directly with one random agent based ond the probability p

# Code: https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/gosgd_worker.py


class GoSGDAgent(Agent):
    def __init__(self, p=0.02, **kwargs):
        super(GoSGDAgent, self).__init__(**kwargs)
        self.w = 1.
        self.p = p
        self.make_train_iter()

    def start(self):
        self.w = 1. / self.graph.nodes_num

    def train_fn(self):
        x, y = self.next_train_batch()
        self._train_on_batch(x, y)
        if draw(self.p):
            self.send_to_peers()
        return len(y)

    def send_to_peers(self):
        self.w /= 2
        # Randomly choose a node
        a_j = choose(self.id, self.graph.nodes_num)
        other_agent = self.graph.get_node(a_j)
        other_agent.receive_message(self)

    def receive_message(self, other_agent):
        super(GoSGDAgent, self).receive_message(other_agent)
        weights = tf.nest.map_structure(lambda a, b: (a * self.w + b * other_agent.w) / (self.w + other_agent.w),
                                        self.get_model_weights(), other_agent.get_model_weights())
        self.set_model_weights(weights)
        self.w += other_agent.w
