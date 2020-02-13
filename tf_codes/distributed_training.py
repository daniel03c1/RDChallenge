# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/densenet/distributed_train.py
import tensorflow as tf


class Trainer:
    def __init__(self, strategy, batch_size):
        self.strategy = strategy
        self.batch_size = batch_size

    def compute_loss(self):
        loss = tf.reduce_sum(self.loss(label, predictions)) / self.batch_size
        loss += sum(self.model.losses) / self.strategy.num_replicas_in_sync
        return loss

    def train_step(self, inputs):
        image, label = inputs
        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)
            loss = self.compute_loss(label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))
        self.train_acc_metric(label, predictions)
        return loss

    def test_step(self):
        image, label = inputs
        predictions = self.model(image, training=False)
        unscaled_test_loss = self.loss_object(label, predictions) \
                             + sum(self.model.losses)

        self.test_acc_metrics(label, predictions)
        self.test_loss_metric(unscaled_test_loss)

    def fit(self):
        pass


@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        features, labels = inputs

        with tf.GradientTape() as tape:
            logits = model(features)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1. / global_batch_size)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        return cross_entropy

    per_example_losses = strategy.experimental_run_v2(
        step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
    return mean_loss


if __name__ == '__main__':
    devices = ['/device:GPU:{}'.format(i) for i in (0, 1, 2, 3)]
    strategy = tf.distribute.MirroredStrategy(devices)

    ''' Defining a Model '''
    with strategy.scope():
        model = None
        optimizer = tf.keras.optimizers.SGD()

    with strategy.scope():
        dataset = None


