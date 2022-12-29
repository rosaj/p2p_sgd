from p2p.agents.p2p_agent import *


class BertAgent(P2PAgent):
    def __init__(self, multi_task_optimization=True, **kwargs):
        if 'early_stopping' not in kwargs:
            kwargs['early_stopping'] = False
        super(BertAgent, self).__init__(**kwargs)
        self.bert_layer = self.model.layers[3]
        self.multi_task_optimization = multi_task_optimization

    def receive_message(self, other_agent):
        super(P2PAgent, self).receive_message(other_agent)
        if self.model.layers[-1].units != other_agent.model.layers[-1].units:
            self.bert_layer.set_weights(tf.nest.map_structure(lambda a, b: (a + b) / 2.0, self.bert_layer.get_weights(), other_agent.bert_layer.get_weights()))
            if self.multi_task_optimization:
                self.bert_layer.trainable = False
        else:
            weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, self.get_model_weights(), other_agent.get_model_weights())
            self.set_model_weights(weights)

        self.received_msg = True
        self.train_rounds = 1

        return True

    def fit(self, epochs=0):
        super(BertAgent, self).fit(epochs)
        if self.multi_task_optimization:
            self.bert_layer.trainable = True
