import tensorflow as tf


def restore_model_ckpt(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(model=model)
    latest_chkpt = tf.train.latest_checkpoint(checkpoint_path)
    checkpoint.restore(latest_chkpt).run_restore_ops()
    return model
