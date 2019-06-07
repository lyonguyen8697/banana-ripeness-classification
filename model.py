import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from utils import NeuralLayers, Optimizer


class BananaRipenessClassifier:
    def __init__(self,
                 hparams,
                 trainable):

        self.trainable = trainable
        self.hparams = hparams
        self.image_shape = [224, 224, 3]
        self.is_train = tf.placeholder_with_default(False, shape=[], name='is_train')

        self.layers = NeuralLayers(trainable=self.trainable,
                                   is_train=self.is_train,
                                   hparams=self.hparams)
        self.optimizer_builder = Optimizer(hparams=hparams)
        self.saver = None
        self.build_model()
        if trainable:
            self.build_optimizer()
            self.build_metrics()
            self.build_summary()

    def build_model(self):
        hparams = self.hparams

        images = tf.placeholder(dtype=tf.float32,
                                shape=[None] + self.image_shape)

        conv1 = self.layers.conv2d(images,
                                   filters=16,
                                   kernel_size=(3, 3),
                                   activation=None)
        conv1 = self.layers.batch_norm(conv1)
        conv1 = tf.nn.relu(conv1)

        conv2 = self.layers.conv2d(conv1,
                                   filters=32,
                                   kernel_size=(3, 3),
                                   activation=None)
        conv2 = self.layers.batch_norm(conv2)
        conv2 = tf.nn.relu(conv2)

        conv3 = self.layers.conv2d(conv2,
                                   filters=64,
                                   kernel_size=(3, 3),
                                   activation=None)
        conv3 = self.layers.batch_norm(conv3)
        conv3 = tf.nn.relu(conv3)

        global_avg_pool = self.layers.global_avg_pool2d(conv3,
                                                        keepdims=False)
        global_avg_pool = self.layers.dropout(global_avg_pool)

        logits = self.layers.dense(global_avg_pool,
                                   units=7,
                                   activation=None)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(probabilities, axis=1)

        self.images = images
        self.logits = logits
        self.probabilities = probabilities
        self.predictions = predictions

    def build_optimizer(self):
        hparams = self.hparams

        global_step = tf.train.get_or_create_global_step()

        labels = tf.placeholder(dtype=tf.int64, shape=[None])

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                    logits=self.logits)

        regularization_loss = tf.losses.get_regularization_loss()

        total_loss = cross_entropy_loss + regularization_loss

        learning_rate = self.optimizer_builder.compute_learning_rate(global_step)

        optimizer = self.optimizer_builder.build(name=hparams.optimizer,
                                                 learning_rate=learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(total_loss, ))
        gradients, _ = tf.clip_by_global_norm(gradients, hparams.clip_gradients)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.apply_gradients(zip(gradients, variables),
                                             global_step=global_step)
        train_op = tf.group([train_op, update_ops])

        self.global_step = global_step
        self.labels = labels
        self.cross_entropy_loss = cross_entropy_loss
        self.regularization_loss = regularization_loss
        self.total_loss = total_loss
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_op = train_op

    def build_metrics(self):

        avg_cross_entropy_loss, avg_cross_entropy_loss_op = tf.metrics.mean_tensor(self.cross_entropy_loss)
        avg_reg_loss, avg_reg_loss_op = tf.metrics.mean_tensor(self.regularization_loss)
        avg_total_loss, avg_total_loss_op = tf.metrics.mean_tensor(self.total_loss)

        accuracy, accuracy_op = tf.metrics.accuracy(labels=self.labels,
                                                    predictions=self.predictions)

        self.metrics = {'cross_entropy_loss': avg_cross_entropy_loss,
                        'regularization_loss': avg_reg_loss,
                        'total_loss': avg_total_loss,
                        'accuracy': accuracy}
        self.metric_ops = {'cross_entropy_loss': avg_cross_entropy_loss_op,
                           'regularization_loss': avg_reg_loss_op,
                           'total_loss': avg_total_loss_op,
                           'accuracy': accuracy_op}

        self.metric_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
        self.reset_metric_op = tf.variables_initializer(self.metric_vars)

    def build_summary(self):

        with tf.name_scope('metric'):
            for metric_name, metric_tensor in self.metrics.items():
                tf.summary.scalar(metric_name, metric_tensor)

        with tf.name_scope('hyperparam'):
            tf.summary.scalar('learning_rate', self.learning_rate)

        self.summary = tf.summary.merge_all()

    def cache_metric_values(self, sess):
        metric_values = sess.run(self.metric_vars)
        self.metric_values = metric_values

    def restore_metric_values(self, sess):
        for var, value in zip(self.metric_vars, self.metric_values):
            sess.run(var.assign(value))

    def train(self, sess, train_dataset, val_dataset, test_dataset=None, load_checkpoint=False, checkpoint=None):
        hparams = self.hparams

        if not os.path.exists(hparams.summary_dir):
            os.mkdir(hparams.summary_dir)
        train_writer = tf.summary.FileWriter(hparams.summary_dir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(hparams.summary_dir + '/val')
        if test_dataset is not None:
            test_writer = tf.summary.FileWriter(hparams.summary_dir + '/test')

        train_fetches = {'train_op': self.train_op,
                         'global_step': self.global_step}
        train_fetches.update(self.metric_ops)
        val_fetches = self.metric_ops

        if test_dataset is not None:
            test_fetches = self.metric_ops

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if load_checkpoint:
            self.load(sess, checkpoint)

        # Training
        for _ in tqdm(range(self.hparams.num_epochs), desc='epoch'):
            for _ in tqdm(range(train_dataset.num_batches), desc='batch', leave=False):
                images, labels = train_dataset.next_batch()

                feed_dict = {self.images: images,
                             self.labels: labels,
                             self.is_train: True}

                train_record = sess.run(train_fetches, feed_dict=feed_dict)

                tqdm.write("Train step {}: total loss: {:>10.5f}   accuracy: {:8.2f}"
                           .format(train_record['global_step'],
                                   train_record['total_loss'],
                                   train_record['accuracy'] * 100))
                if train_record['global_step'] % hparams.summary_period == 0:
                    summary = sess.run(self.summary)
                    train_writer.add_summary(summary, train_record['global_step'])

                # Validation
                if (train_record['global_step'] + 1) % hparams.eval_period == 0:
                    self.cache_metric_values(sess)
                    sess.run(self.reset_metric_op)
                    for _ in tqdm(range(val_dataset.num_batches), desc='val', leave=False):
                        images, labels = val_dataset.next_batch()

                        feed_dict = {self.images: images,
                                     self.labels: labels}

                        val_record = sess.run(val_fetches, feed_dict=feed_dict)

                    tqdm.write(
                        "Validation step {}: total loss: {:>10.5f}   accuracy: {:8.2f}"
                        .format(train_record['global_step'],
                                val_record['total_loss'],
                                val_record['accuracy'] * 100))
                    summary = sess.run(self.summary)
                    val_writer.add_summary(summary, train_record['global_step'])
                    val_writer.flush()
                    val_dataset.reset()

                    self.restore_metric_values(sess)

            sess.run(self.reset_metric_op)

            self.save(sess, global_step=train_record['global_step'])

            train_dataset.reset()

        train_writer.close()
        val_writer.close()

        # Testing
        if test_dataset is not None:
            sess.run(self.reset_metric_op)
            for _ in tqdm(range(test_dataset.num_batches), desc='testing', leave=False):
                images, labels = val_dataset.next_batch()

                feed_dict = {self.images: images,
                             self.labels: labels}

                test_record = sess.run(test_fetches, feed_dict=feed_dict)

            tqdm.write("Testing: total loss: {:>10.5f}   accuracy: {:8.2f}"
                       .format(test_record['total_loss'],
                               test_record['accuracy'] * 100))
            summary = sess.run(self.summary)
            test_writer.add_summary(summary, train_record['global_step'])
            test_writer.flush()
            test_writer.close()

    def eval(self, sess, test_dataset, checkpoint=None):
        hparams = self.hparams

        result = {'image': [],
                  'ground truth': [],
                  'prediction': []}

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self.load(sess, checkpoint)

        # Testing
        for _ in tqdm(range(test_dataset.num_batches), desc='batch', leave=False):
            images, labels = test_dataset.next_batch()

            predictions, _ = sess.run([self.predictions, self.metric_ops], feed_dict={self.images: images,
                                                                                      self.labels: labels})

            for image, file, label, prediction in zip(images, test_dataset.current_image_files, labels, predictions):
                result['image'].append(file)
                result['ground truth'].append(label)
                result['prediction'].append(prediction)

                plt.imshow(image)
                plt.title(prediction)
                plt.savefig('{}/{}'.format(hparams.test_result_dir, file))
                plt.close()

        result = pd.DataFrame.from_dict(result)
        result.to_csv('result.txt')

        eval_result = sess.run(self.metrics)
        with open('eval.txt', 'w') as f:
            for name, value in eval_result.items():
                print('{}: {}'.format(name, value))
                print('{}: {}'.format(name, value), file=f, end='\n')

    def test(self, sess, test_dataset, checkpoint=None):
        hparams = self.hparams

        result = {'image': [],
                  'prediction': []}

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self.load(sess, checkpoint)

        # Testing
        for _ in tqdm(range(test_dataset.num_batches), desc='batch', leave=False):
            images = test_dataset.next_batch()

            predictions = sess.run(self.predictions, feed_dict={self.images: images})

            for image, file, prediction in zip(images, test_dataset.current_image_files, predictions):
                result['image'].append(file)
                result['prediction'].append(prediction)

                plt.imshow(image)
                plt.title(prediction)
                plt.savefig('{}/{}'.format(hparams.test_result_dir, file))
                plt.close()

        result = pd.DataFrame.from_dict(result)
        result.to_csv('result.txt')

    def save(self, sess, save_dir=None, global_step=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        save_dir = save_dir or self.hparams.save_dir
        global_step = global_step or self.global_step.eval(session=sess)

        self.saver.save(sess, save_dir + '/banana-ripeness-classifier-model.ckpt', global_step=global_step)

    def load(self, sess, checkpoint=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(self.hparams.save_dir)
            if checkpoint is None:
                return
        self.saver.restore(sess, checkpoint)