from common.model import *
import environ


class Agent:
    # noinspection PyDefaultArgument
    def __init__(self,
                 train,
                 val,
                 test,
                 base_model,
                 complex_model=None,
                 batch_size=5,
                 ensemble_metrics=[MaskedSparseCategoricalAccuracy()]):

        self.train_x, self.train_y = tf.convert_to_tensor(train[0]), tf.convert_to_tensor(train[1])
        self.val_x, self.val_y = tf.convert_to_tensor(val[0]), tf.convert_to_tensor(val[1])
        self.test_x, self.test_y = tf.convert_to_tensor(test[0]), tf.convert_to_tensor(test[1])

        self.batch_size = batch_size

        self.base_model = base_model
        self.complex_model = complex_model

        self.ensemble_metrics = ensemble_metrics or []
        self.kl_loss = KLDivergence()

        self.tr_iter = self._create_train_iter()
        self.val_metric = 0
        self._train_rounds = 1

        self.hist_acc = {"train_base": [0],
                         "train_complex": [0],
                         "train_ensemble": [0],
                         "val_base": [0],
                         "val_complex": [0],
                         "val_ensemble": [0],
                         "test_base": [0],
                         "test_complex": [0],
                         "test_ensemble": [0]}

        self.id = environ.next_agent_id()
        self.train_fn = None
        self.temp = []

    @property
    def train_complex_acc(self):
        return self.hist_acc["train_complex"][-1]

    @property
    def train_base_acc(self):
        return self.hist_acc["train_base"][-1]

    @property
    def train_ensemble_acc(self):
        return self.hist_acc["train_ensemble"][-1]

    @property
    def val_complex_acc(self):
        return self.hist_acc["val_complex"][-1]

    @property
    def val_base_acc(self):
        return self.hist_acc["val_base"][-1]

    @property
    def val_ensemble_acc(self):
        return self.hist_acc["val_ensemble"][-1]

    @property
    def test_complex_acc(self):
        return self.hist_acc["test_complex"][-1]

    @property
    def test_base_acc(self):
        return self.hist_acc["test_base"][-1]

    @property
    def test_ensemble_acc(self):
        return self.hist_acc["test_ensemble"][-1]

    @property
    def has_complex(self):
        return self.complex_model is not None

    @property
    def train_len(self):
        return len(self.train_y)

    @property
    def val_len(self):
        return len(self.val_y)

    @property
    def trainable(self):
        return self._train_rounds > 0

    @property
    def memory_footprint(self):
        if self.has_complex:
            return calculate_memory_model_size(self.base_model) + calculate_memory_model_size(self.base_model)
        return calculate_memory_model_size(self.base_model)

    @staticmethod
    def _update_metrics(metrics, preds, y_true):
        results = []
        for metric in metrics:
            if hasattr(metric, "reset_state"):
                metric.reset_state()
            else:
                metric.reset_states()
            metric.update_state(y_true, preds)
            results.append(metric.result().numpy())
        return results

    @staticmethod
    def _update_compiled_metrics(model, preds, y_true):
        if len(model.compiled_metrics.metrics) == 0:
            model.compiled_metrics.build(0, 0)
        return Agent._update_metrics(model.compiled_metrics.metrics, preds, y_true)

    @staticmethod
    def _reset_compiled_metrics(model):
        model.compiled_metrics.reset_state()
        # for metric in model.compiled_metrics.metrics:
        # metric.reset_states()

    def deserialize(self):
        self.base_model = load('p2p_models/agent_{}_base'.format(self.id))
        if self.has_complex:
            self.complex_model = load('p2p_models/agent_{}_complex'.format(self.id))

    def serialize(self):
        """
        signatures = self.make_train_function().get_concrete_function(
            x=tf.TensorSpec(shape=(None, 10), dtype=tf.int32, name='x'),
            y=tf.TensorSpec(shape=(None,), dtype=tf.int32, name='y')
        )
        """
        # signatures = self.make_train_function()
        # signatures = self.make_train_function().get_concrete_function()
        # signatures = {cfn.name.decode(): cfn}
        # signatures = self.make_train_function()
        self.train_fn = None
        signatures = None
        save(self.base_model, 'p2p_models/agent_{}_base'.format(self.id), signatures)
        self.base_model = None
        if self.has_complex:
            save(self.complex_model, 'p2p_models/agent_{}_complex'.format(self.id), signatures)
            self.complex_model = True

    def _create_dataset(self):
        return tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)) \
            .shuffle(self.batch_size) \
            .batch(self.batch_size)

    def _create_train_iter(self):
        self.tr_iter = self._create_dataset().as_numpy_iterator()
        return self.tr_iter

    def receive_update(self, new_weights):
        weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                              self.get_share_weights(),
                                              new_weights)
        self.temp.append(weights_delta)
        if len(self.temp) == 5:
            weights = average_weights(self.temp)
            # wd_mean = [-1.0 * x for x in wd_mean]
            self.base_model.optimizer.apply_gradients(zip(weights, self.base_model.trainable_variables))
            self._train_rounds = 1
            self.temp.clear()
        """ 
        weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                              trainable_variables,
                                              self.base_model.trainable_variables)
        self.base_model.optimizer.apply_gradients(zip(weights_delta, self.base_model.trainable_variables))
        self._train_rounds = 1
        
        self.temp.append(weights_delta)
        if len(self.temp) == 5:
            wd_mean = average_trainable_variables(self.temp)
            # wd_mean = [-1.0 * x for x in wd_mean]
            self.base_model.optimizer.apply_gradients(zip(wd_mean, self.base_model.trainable_variables))
            self.temp.clear()
            self._train_rounds = 1
        """
        return True

    def weighted_update(self, new_weights, num_examples):
        w1, w2 = self.get_share_weights(), new_weights
        n1, n2 = self.train_len, num_examples
        total_count = n1 + n2
        wn1, wn2 = n1 / total_count, n2 / total_count
        weights = tf.nest.map_structure(lambda a, b: a * wn1 + b * wn2, w1, w2)
        self.set_base_weights(weights)
        self._train_rounds = 1
        return True

    def receive_model(self, other_agent, mode='replace', only_improvement=True):
        new_weights = other_agent.get_share_weights()
        if only_improvement:
            print("only_improvement not implemented yet")
            return False

        if mode == 'replace':
            self.set_base_weights(new_weights)
            self._train_rounds = max(self._train_rounds, 1)
            return True
        elif mode == 'average':
            weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, self.get_share_weights(), new_weights)
            # self.base_model.set_weights(weights)
            self.set_base_weights(weights)
            self._train_rounds = 1
            return True
        elif mode == 'weighted_update':
            return self.weighted_update(new_weights, other_agent.train_len)
        elif mode == 'peer_sgd':
            Agent._peer_sgd_update_weights(self.base_model, other_agent.base_model, self.next_train_batch())
            self._train_rounds = 1
            return True
        elif mode == 'test':
            # weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, self.get_share_weights(), new_weights)
            weights = tf.nest.map_structure(lambda a, b: (b / 2.0), self.get_share_weights(), new_weights)
            self.base_model.optimizer.apply_gradients(zip(weights, self.base_model.trainable_variables))
            # self.base_model.set_weights(weights)
            self._train_rounds = 1
            return True
        elif mode == 'cache_average':
            if len(self.temp) == 15:
                self.temp.append(self.base_model.get_weights())
                # avg_weights = average_weights([self.base_model.get_weights(), average_weights(self.temp)])
                avg_weights = average_weights(self.temp)
                # avg_weights = average_weights([self.base_model.get_weights(), avg_weights])
                self.set_base_weights(avg_weights)
                # tf.nest.map_structure(lambda v, t: v.assign(t), self.base_model.trainable_variables, avg_weights)
                self.temp.clear()
                self._train_rounds = 1
            else:
                self.temp.append(new_weights)
            return True

        if mode is not None:
            return False
        m = self.base_model
        new_model = clone_model(m)
        new_model.set_weights(new_weights)
        compile_model(new_model)
        new_val_metric = self._train_acc(new_model)[0]

        """
        m_logits = m(self.train_x, training=False)
        n_logits = new_model(self.train_x, training=False)
        loss_m = m.loss(self.train_y, m_logits)
        loss_n = new_model.loss(self.train_y, n_logits)
        print(loss_m, loss_n)
        """
        # d_loss = self.kl_loss(m_logits, n_logits)
        # print(d_loss)
        # if d_loss.numpy() >= 1:
        #     return False

        if not only_improvement or new_val_metric > self.val_metric:
            if mode == 'average':
                m.set_weights(average_weights([m.get_weights(), new_model.get_weights()]))
                new_val_metric = self._train_acc(m)[0]
            elif mode == 'replace':
                m.set_weights(new_weights)
            elif mode == 'cache_average':
                if len(self.temp) == 14:
                    self.temp.append(m.get_weights())
                    m.set_weights(average_weights(self.temp))
                    new_val_metric = self._train_acc(m)[0]
                    self.temp.clear()
                else:
                    self.temp.append(new_weights)
                    return True
            elif mode == 'peer_sgd':
                Agent._peer_sgd_update_weights(m, new_model, (self.train_x, self.train_y))
                new_val_metric = self._train_acc(m)[0]
            self.val_metric = new_val_metric
            self._train_rounds = max(self._train_rounds, 1)
            return True
        return False

    @staticmethod
    def _peer_sgd_update_weights(mi, mj, data):
        (x, y) = data
        with tf.GradientTape() as tape:
            logits_i = mi(x)
            logits_j = mj(x)

            loss_i = mi.loss(y, logits_i)
            loss_j = mj.loss(y, logits_j)
            # print(loss_i.numpy(), loss_j.numpy())
            kl_loss = kl_loss_compute(logits_i, logits_j)

            i = 1 - loss_i / (loss_i + loss_j)
            j = 1 - loss_j / (loss_i + loss_j)

            mi.set_weights(add_weights((multiply_weights_with_num(mi.get_weights(), i.numpy()),
                                        multiply_weights_with_num(mj.get_weights(), j.numpy()))))

            loss = loss_i + kl_loss

        grads = tape.gradient(loss, mi.trainable_variables)
        mi.optimizer.apply_gradients(zip(grads, mi.trainable_variables))

    def get_share_weights(self):
        return self.base_model.get_weights()

    def next_train_batch(self):
        try:
            return self.tr_iter.next()
        except StopIteration:
            self._create_train_iter()
            return self.tr_iter.next()

    def _train_on_batch(self, x=None, y=None):
        if x is None and y is None:
            x, y = self.next_train_batch()

        train_step_fn = self.make_train_function()
        train_step_fn(self.base_model, self.complex_model, x, y, self.kl_loss)
        # self.base_model.train_on_batch(x, y)
        """
        if not self.has_complex:
            # self.base_model.train_on_batch(x, y)
            self._train_batch(x, y)
        else:
            self._train_dml(x, y)
        """

    def make_train_function(self):
        if self.train_fn is not None:
            return self.train_fn
        """
        if not self.has_complex:
            train_fn = self._train_batch
        else:
            train_fn = self._train_dml
        """
        def train_step(base_model, complex_model, x, y, kl_loss):
            if complex_model is None:
                Agent._model_train_batch(base_model, x, y)
            else:
                Agent._model_train_dml(base_model, complex_model, x, y, kl_loss)

        train_fn = train_step
        train_fn = tf.function(train_fn, experimental_relax_shapes=True)

        """
        train_fn = tf.function(train_fn, input_signature=[
            tf.TensorSpec(shape=(None, 10), dtype=tf.int32, name='x'),
            tf.TensorSpec(shape=(None,), dtype=tf.int32, name='y')
        ])
        """
        self.train_fn = train_fn
        return train_fn

    def train_epoch(self):
        if self._train_rounds < 1:
            return False

        Agent._reset_compiled_metrics(self.base_model)
        if self.has_complex:
            Agent._reset_compiled_metrics(self.complex_model)

        for (x, y) in self._create_dataset():
            self._train_on_batch(x, y)

        self._calc_new_accs()
        self._train_rounds = max(self._train_rounds - 1, 0)
        self.val_metric = self.train_base_acc

        return True

    def fit(self):
        for _ in range(self._train_rounds):
            self.train_epoch()

    def _train_batch(self, x, y):
        return Agent._model_train_batch(self.base_model, x, y)

    def _train_dml(self, x, y):
        return Agent._model_train_dml(self.base_model, self.complex_model, x, y, self.kl_loss)

    @staticmethod
    def _model_train_batch(model, x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = model.loss(y, logits)
        grad = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss

    @staticmethod
    def _model_train_dml(base_model, complex_model, x, y, kl_loss):
        alpha = 0.5
        with tf.GradientTape(persistent=True) as tape:
            b_logits = base_model(x, training=True)
            c_logits = complex_model(x, training=True)

            d_loss = kl_loss(c_logits, b_logits)
            b_loss = base_model.loss(y, b_logits)
            c_loss = complex_model.loss(y, c_logits)

            # c, b = c_loss, b_loss
            # alpha = 1 - c / (c + b)
            # tf.print(alpha, c, b)
            # alpha = 0.5
            # Bigger Alpha => complex model advantage
            base_loss = tf.add(tf.math.scalar_mul(1 - alpha, b_loss), tf.math.scalar_mul(alpha, d_loss))
            complex_loss = tf.add(tf.math.scalar_mul(alpha, c_loss), tf.math.scalar_mul(1 - alpha, d_loss))

        b_grads = tape.gradient(base_loss, base_model.trainable_variables)
        base_model.optimizer.apply_gradients(zip(b_grads, base_model.trainable_variables))
        base_model.compiled_metrics.update_state(y, b_logits)

        c_grads = tape.gradient(complex_loss, complex_model.trainable_variables)
        complex_model.optimizer.apply_gradients(zip(c_grads, complex_model.trainable_variables))
        complex_model.compiled_metrics.update_state(y, c_logits)
        return d_loss

    @staticmethod
    def _assign_weights(model, weights):
        tf.nest.map_structure(lambda v, t: v.assign(t), model.trainable_variables, weights)

    def set_base_weights(self, weights):
        Agent._assign_weights(self.base_model, weights)

    def set_complex_weights(self, weights):
        Agent._assign_weights(self.complex_model, weights)

    def _calc_new_accs(self):
        self.hist_acc["train_base"].append(self.base_train_acc()[0])
        self.hist_acc["val_base"].append(self.base_val_acc()[0])
        self.hist_acc["test_base"].append(self.base_test_acc()[0])
        if self.has_complex:
            self.hist_acc["train_complex"].append(self.complex_train_acc()[0])
            self.hist_acc["train_ensemble"].append(self.ensemble_train_acc()[0])
            self.hist_acc["val_complex"].append(self.complex_val_acc()[0])
            self.hist_acc["val_ensemble"].append(self.ensemble_val_acc()[0])
            self.hist_acc["test_complex"].append(self.complex_test_acc()[0])
            self.hist_acc["test_ensemble"].append(self.ensemble_test_acc()[0])

    @staticmethod
    def _model_acc(m, x, y):
        preds = m(x, training=False)
        return Agent._update_compiled_metrics(m, preds, y)

    @staticmethod
    def _ensemble_acc(m1, m2, x, y, metrics):
        alpha = 0.5
        m1_pred = m1(x, training=False)
        m2_pred = m2(x, training=False)
        results = Agent._update_metrics(metrics, (m1_pred * (1 - alpha) + m2_pred * alpha), y)
        return results

    def _train_acc(self, m):
        return Agent._model_acc(m, self.train_x, self.train_y)

    def _val_acc(self, m):
        return Agent._model_acc(m, self.val_x, self.val_y)

    def _test_acc(self, m):
        return Agent._model_acc(m, self.test_x, self.test_y)

    def base_train_acc(self):
        return self._train_acc(self.base_model)

    def complex_train_acc(self):
        return self._train_acc(self.complex_model)

    def ensemble_train_acc(self):
        return Agent._ensemble_acc(self.base_model, self.complex_model, self.train_x, self.train_y, self.ensemble_metrics)

    def base_val_acc(self):
        return self._val_acc(self.base_model)

    def complex_val_acc(self):
        return self._val_acc(self.complex_model)

    def ensemble_val_acc(self):
        return Agent._ensemble_acc(self.base_model, self.complex_model, self.val_x, self.val_y, self.ensemble_metrics)

    def base_test_acc(self):
        return self._test_acc(self.base_model)

    def complex_test_acc(self):
        return self._test_acc(self.complex_model)

    def ensemble_test_acc(self):
        return Agent._ensemble_acc(self.base_model, self.complex_model, self.test_x, self.test_y, self.ensemble_metrics)
