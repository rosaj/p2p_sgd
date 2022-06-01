import tensorflow as tf


class MaskedSparseCategoricalCrossentropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __call__(self, y_true, y_pred, **kwargs):
        label_ids, label_mask = y_true[0], y_true[1]
        label_ids_masked = tf.boolean_mask(label_ids, label_mask)
        logits_masked = tf.boolean_mask(y_pred, label_mask)
        return super().__call__(label_ids_masked, logits_masked, **kwargs)
