import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Precision, Recall
from tensorflow_addons.metrics import F1Score


class MaskedSparseCategoricalCrossentropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __call__(self, y_true, y_pred, **kwargs):
        label_ids, label_mask = y_true[0], y_true[1]
        label_ids_masked = tf.boolean_mask(label_ids, label_mask)
        logits_masked = tf.boolean_mask(y_pred, label_mask)
        return super().__call__(label_ids_masked, logits_masked, **kwargs)


class MaskedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """An accuracy metric that masks some tokens."""

    def __init__(self, name='masked_sparse_categorical_accuracy', **kwargs):
        super().__init__(name, **kwargs)

    def update_state(self, y_true, y_pred, **kwargs):
        label_ids, label_mask = y_true[0], y_true[1]
        label_ids_masked = tf.boolean_mask(label_ids, label_mask)
        logits_masked = tf.boolean_mask(y_pred, label_mask)
        super().update_state(label_ids_masked, logits_masked, **kwargs)


class MaskedF1Score(F1Score):

    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'masked_f1_score'
        if 'num_classes' not in kwargs:
            kwargs['num_classes'] = 30522
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, **kwargs):
        label_ids, label_mask = y_true[0], y_true[1]
        label_ids_masked = tf.boolean_mask(label_ids, label_mask)
        logits_masked = tf.boolean_mask(y_pred, label_mask)
        super().update_state(label_ids_masked, logits_masked, **kwargs)


class MaskedPrecision(Precision):
    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'masked_precision'
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, **kwargs):
        label_ids, label_mask = y_true[0], y_true[1]
        label_ids_masked = tf.boolean_mask(label_ids, label_mask)
        logits_masked = tf.boolean_mask(y_pred, label_mask)
        super().update_state(label_ids_masked, logits_masked, **kwargs)


class MaskedRecall(Recall):
    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'masked_recall'
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, **kwargs):
        label_ids, label_mask = y_true[0], y_true[1]
        label_ids_masked = tf.boolean_mask(label_ids, label_mask)
        logits_masked = tf.boolean_mask(y_pred, label_mask)
        super().update_state(label_ids_masked, logits_masked, **kwargs)
