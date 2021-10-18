from common.model import *
import environ


class Agent:
    # noinspection PyDefaultArgument
    def __init__(self,
                 train,
                 val,
                 test,
                 shared_model,
                 private_model=None,
                 batch_size=5,
                 ensemble_metrics=[MaskedSparseCategoricalAccuracy()]):

        self.batch_size = batch_size
        self.train = self._create_dataset(train[0], train[1])
        self.val = self._create_dataset(val[0], val[1])
        self.test = self._create_dataset(test[0], test[1])
        self.train_len = len(train[1])

        self.shared_model = shared_model
        self.private_model = private_model

        self.ensemble_metrics = ensemble_metrics or []
        self.kl_loss = KLDivergence()

        self.trained_examples = 0

        self.train_rounds = 1
        self.can_msg = False

        self.hist = {"train_shared":   [0],
                     "train_private":  [0],
                     "train_ensemble": [0],
                     "val_shared":     [0],
                     "val_private":    [0],
                     "val_ensemble":   [0],
                     "test_shared":    [0],
                     "test_private":   [0],
                     "test_ensemble":  [0],
                     "examples":       [0],
                     "train_len":      self.train_len,
                     "useful_msg":     [0],
                     "useless_msg":    [0],
                     }

        self.id = environ.next_agent_id()
        self.train_fn = None
        self.device = None

    @property
    def train_private_acc(self):
        return self.hist["train_private"][-1]

    @property
    def train_shared_acc(self):
        return self.hist["train_shared"][-1]

    @property
    def train_ensemble_acc(self):
        return self.hist["train_ensemble"][-1]

    @property
    def val_private_acc(self):
        return self.hist["val_private"][-1]

    @property
    def val_shared_acc(self):
        return self.hist["val_shared"][-1]

    @property
    def val_ensemble_acc(self):
        return self.hist["val_ensemble"][-1]

    @property
    def test_private_acc(self):
        return self.hist["test_private"][-1]

    @property
    def test_shared_acc(self):
        return self.hist["test_shared"][-1]

    @property
    def test_ensemble_acc(self):
        return self.hist["test_ensemble"][-1]

    @property
    def has_private(self):
        return self.private_model is not None

    @property
    def trainable(self):
        return self.train_rounds > 0

    @property
    def memory_footprint(self):
        if self.has_private:
            return calculate_memory_model_size(self.shared_model) + calculate_memory_model_size(self.shared_model)
        return calculate_memory_model_size(self.shared_model)

    @staticmethod
    def _reset_compiled_metrics(model):
        model.compiled_metrics.reset_state()

    def deserialize(self):
        self.shared_model = load('p2p_models/agent_{}_shared'.format(self.id))
        if self.has_private:
            self.private_model = load('p2p_models/agent_{}_private'.format(self.id))

    def serialize(self, save_only=False):
        save(self.shared_model, 'p2p_models/agent_{}_shared'.format(self.id))
        if not save_only:
            self.shared_model = None
            self.train_fn = None
        if self.has_private:
            save(self.private_model, 'p2p_models/agent_{}_private'.format(self.id))
            if not save_only:
                self.private_model = True

    def _create_dataset(self, x, y):
        return tf.data.Dataset.from_tensor_slices((x, y)) \
            .shuffle(self.batch_size) \
            .batch(self.batch_size) \
            .prefetch(1)

    def receive_model(self, other_agent, mode='average'):
        only_improvement = 'improve' in mode
        if only_improvement:
            mode = mode.replace('improve', '').replace('_', '')

        new_weights = other_agent.get_shared_weights()
        if only_improvement:
            if Agent._model_acc(self.shared_model, self.val)[0] >= Agent._model_acc(other_agent.shared_model, self.val)[0]:
                self.hist["useless_msg"][-1] += 1
                return False

        if mode == 'average':
            weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, self.get_shared_weights(), new_weights)
            self.set_shared_weights(weights)
        elif mode == 'layer-average':
            for li, al1 in enumerate(self.shared_model.layers):
                al2 = other_agent.shared_model.layers[li]
                aw = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, al1.get_weights(), al2.get_weights())
                al1.set_weights(aw)
        else:
            raise NotImplementedError("No method {} found".format(mode))

        self.hist["useful_msg"][-1] += 1
        self.can_msg = True
        self.train_rounds = 1

        return True

    def set_shared_weights(self, weights):
        self.shared_model.set_weights(weights)

    def get_shared_weights(self):
        return self.shared_model.get_weights()

    def _train_on_batch(self, x, y):
        train_step_fn = self.make_train_function()
        train_step_fn(self.shared_model, self.private_model, x, y, self.kl_loss)

    def make_train_function(self):
        if self.train_fn is not None:
            return self.train_fn

        def train_step(shared_model, private_model, x, y, kl_loss):
            if private_model is None:
                Agent._model_train_batch(shared_model, x, y)
            else:
                Agent._model_train_dml(shared_model, private_model, x, y, kl_loss)

        train_fn = train_step
        # tf.function consumes a lot of RAM but is gradually faster on CPU
        # Training on GPU is around twice the time slower than without it
        # train_fn = tf.function(train_fn, experimental_relax_shapes=True)

        self.train_fn = train_fn
        return train_fn

    def train_epoch(self):
        if self.train_rounds < 1:
            return False

        Agent._reset_compiled_metrics(self.shared_model)
        if self.has_private:
            Agent._reset_compiled_metrics(self.private_model)

        for (x, y) in self.train:
            self._train_on_batch(x, y)

        self.train_rounds = max(self.train_rounds - 1, 0)
        self.trained_examples += self.train_len

        return True

    def fit(self):
        if self.train_rounds < 1:
            return

        for _ in range(self.train_rounds):
            acc_before = self.shared_val_acc()[0]
            self.train_epoch()
            acc_after = self.shared_val_acc()[0]
            for al1 in self.shared_model.layers:
                if 'batch_normalization' in al1.name:
                    continue
                al1.trainable = acc_before < acc_after

    @staticmethod
    def _model_train_batch(model, x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = model.loss(y, logits)
        grad = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss

    @staticmethod
    def _model_train_dml(shared_model, private_model, x, y, kl_loss):
        with tf.GradientTape(persistent=True) as tape:
            b_logits = shared_model(x, training=True)
            c_logits = private_model(x, training=True)

            bd_loss = kl_loss(b_logits, c_logits)
            cd_loss = kl_loss(c_logits, b_logits)
            b_loss = shared_model.loss(y, b_logits)
            c_loss = private_model.loss(y, c_logits)

            shared_loss = tf.add(b_loss, bd_loss)
            private_loss = tf.add(c_loss, cd_loss)

        b_grads = tape.gradient(shared_loss, shared_model.trainable_variables)
        shared_model.optimizer.apply_gradients(zip(b_grads, shared_model.trainable_variables))
        shared_model.compiled_metrics.update_state(y, b_logits)

        c_grads = tape.gradient(private_loss, private_model.trainable_variables)
        private_model.optimizer.apply_gradients(zip(c_grads, private_model.trainable_variables))
        private_model.compiled_metrics.update_state(y, c_logits)
        return tf.divide(tf.add(bd_loss, cd_loss), 2)

    def calc_new_acc(self):
        self.hist["examples"].append(self.trained_examples)
        self.hist["useful_msg"].append(0)
        self.hist["useless_msg"].append(0)

        self.hist["train_shared"].append(self.shared_train_acc()[0])
        self.hist["val_shared"].append(self.shared_val_acc()[0])
        self.hist["test_shared"].append(self.shared_test_acc()[0])
        if self.has_private:
            self.hist["train_private"].append(self.private_train_acc()[0])
            self.hist["train_ensemble"].append(self.ensemble_train_acc()[0])
            self.hist["val_private"].append(self.private_val_acc()[0])
            self.hist["val_ensemble"].append(self.ensemble_val_acc()[0])
            self.hist["test_private"].append(self.private_test_acc()[0])
            self.hist["test_ensemble"].append(self.ensemble_test_acc()[0])

    @staticmethod
    def _model_acc(m, dataset):
        if len(m.compiled_metrics.metrics) == 0:
            m.compiled_metrics.build(0, 0)
        metrics = m.compiled_metrics.metrics
        for metric in metrics:
            if hasattr(metric, "reset_state"):
                metric.reset_state()
            else:
                metric.reset_states()
        for (dx, dy) in dataset:
            preds = m(dx, training=False)
            for metric in metrics:
                metric.update_state(dy, preds)
        return [metric.result().numpy() for metric in metrics]

    @staticmethod
    def _ensemble_acc(m1, m2, dataset, metrics):
        alpha = 0.5
        for metric in metrics:
            if hasattr(metric, "reset_state"):
                metric.reset_state()
            else:
                metric.reset_states()

        for (dx, dy) in dataset:
            m1_pred = m1(dx, training=False)
            m2_pred = m2(dx, training=False)
            for metric in metrics:
                metric.update_state(dy, (m1_pred * (1 - alpha) + m2_pred * alpha))
        return [metric.result().numpy() for metric in metrics]

    def _train_acc(self, m):
        return self._model_acc(m, self.train)

    def _val_acc(self, m):
        return self._model_acc(m, self.val)

    def _test_acc(self, m):
        return self._model_acc(m, self.test)

    def shared_train_acc(self):
        return self._train_acc(self.shared_model)

    def private_train_acc(self):
        return self._train_acc(self.private_model)

    def ensemble_train_acc(self):
        return Agent._ensemble_acc(self.shared_model, self.private_model, self.train, self.ensemble_metrics)

    def shared_val_acc(self):
        return self._val_acc(self.shared_model)

    def private_val_acc(self):
        return self._val_acc(self.private_model)

    def ensemble_val_acc(self):
        return Agent._ensemble_acc(self.shared_model, self.private_model, self.val, self.ensemble_metrics)

    def shared_test_acc(self):
        return self._test_acc(self.shared_model)

    def private_test_acc(self):
        return self._test_acc(self.private_model)

    def ensemble_test_acc(self):
        return Agent._ensemble_acc(self.shared_model, self.private_model, self.test, self.ensemble_metrics)
