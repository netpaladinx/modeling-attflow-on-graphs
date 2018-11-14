from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging

import numpy as np
import tensorflow as tf

from utils import write_rows

TABLE = 'xxr_experiment_attflow_beta'
FIELDS = ['model', 'dataset', 'gridworld_seed', 'splitting_seed', 'shuffling_seeed', 'model_seed', 'rank',
          'h1r_valid', 'h1r_test', 'h1r_epoch', 'h5r_valid', 'h5r_test', 'h5r_epoch', 'h10r_valid', 'h10r_test', 'h10r_epoch',
          'mr_r_valid', 'mr_r_test', 'mr_r_epoch', 'mrr_r_valid', 'mrr_r_test', 'mrr_r_epoch',
          'h1f_valid', 'h1f_test', 'h1f_epoch', 'h5f_valid', 'h5f_test', 'h5f_epoch', 'h10f_valid', 'h10f_test', 'h10f_epoch',
          'mr_f_valid', 'mr_f_test', 'mr_f_epoch', 'mrr_f_valid', 'mrr_f_test', 'mrr_f_epoch']

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.train_tracker = []
        self.valid_evaluator = Evaluator(model, source='valid')
        self.test_evaluator = Evaluator(model, source='test')

        self.logger = logging.getLogger('Trainer_%s' % self.name)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('%s.log' % self.name)
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] ## %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info('\n=========================')
        self.logger.info('model: %s, dataset: %s, gridword_seed: %d, splitting_seed: %d, shuffling_seed: %d, model_seed: %d'
                         % (self.model_name, self.dataset_name, self.gridworld_seed, self.splitting_seed, self.shuffling_seed, self.model_seed))

    def __getattr__(self, name):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        else:
            raise ValueError('`%s` is not defined.' % name)

    def __call__(self):
        with tf.Session(graph=self.tf_graph, config=self.tf_config) as sess:
            if self.checkpoint is not None:
                self.saver.restore(sess, self.checkpoint)
            else:
                sess.run(self.init_op)

            n_epochs = 0
            n_itrs = 0
            while n_epochs < self.max_epochs:
                n_epochs += 1

                for bs, batch in self.get_train_batch(self.batch_size):
                    n_itrs += 1
                    _, loss, accuracy = \
                        sess.run([self.train_op, self.loss, self.accuracy],
                                 feed_dict={self.inputs: batch, self.learning_rate: self.learning_rates[int((n_epochs-1)/10)]})
                    self.train_tracker.append((n_epochs, n_itrs, loss, accuracy))

                metric_valid = self.valid_evaluator(sess, n_epochs)
                metric_test = self.test_evaluator(sess, n_epochs)
                self._write_log(n_epochs, metric_valid, metric_test)


    def _write_log(self, n_epochs, metric_valid, metric_test):
        self.logger.info('[EVAL] epoch: %d, h1_r: %.4f (%.4f), h5_r: %.4f (%.4f), h10_r: %.4f (%.4f), mr_r: %.4f (%.4f), mrr_r: %.4f (%.4f), '
                         'h1_f: %.4f (%.4f), h5_f: %.4f (%.4f), h10_f: %.4f (%.4f), mr_f: %.4f (%.4f), mrr_f: %.4f (%.4f)' %
                         (n_epochs, metric_valid[1], metric_test[1], metric_valid[2], metric_test[2],
                          metric_valid[3], metric_test[1], metric_valid[2], metric_test[2],
                          metric_valid[5], metric_test[1], metric_valid[2], metric_test[2],
                          metric_valid[7], metric_test[1], metric_valid[2], metric_test[2],
                          metric_valid[9], metric_test[1], metric_valid[2], metric_test[2],))

    def _summarize_and_write_sql(self):
        mtr_va = self.valid_evaluator.metrics
        mtr_te = self.test_evaluator.metrics
        mtr = []
        for c in range(1, 11):
            epoch_mv_mt = [(mtr_va[r][0], mtr_va[r][c], mtr_te[r][c]) for r in range(len(mtr_va))]
            if c == 4 or c == 9:
                epoch_mv_mt = sorted(epoch_mv_mt, cmp=lambda (e1, v1, t1), (e2, v2, t2): 0 if (e1, v1) == (e2, v2) else 1 if (v1, e1) > (v2, e2) else -1)
            else:
                epoch_mv_mt = sorted(epoch_mv_mt, cmp=lambda (e1, v1, t1), (e2, v2, t2): 0 if (e1, v1) == (e2, v2) else 1 if (-v1, e1) > (-v2, e2) else -1)
            mtr.append(epoch_mv_mt)

        rows = []
        for r in range(len(mtr_va)):
            rank = r + 1
            summ = '[SUMM] %3d | h1r: %.4f (%.5f, %d) | h5r: %.4f (%.5f, %d) | h10r: %.4f (%.5f, %d) | mr_r: %.4f (%.5f, %d) | mrr_r: %.4f (%.5f, %d) | ' \
                   'h1f: %.4f (%.5f, %d) | h5f: %.4f (%.5f, %d) | h10f: %.4f (%.5f, %d) | mr_f: %.4f (%.5f, %d) | mrr_f: %.4f (%.5f, %d)' % \
                   (rank, mtr[0][r][1], mtr[0][r][2], mtr[0][r][0], mtr[1][r][1], mtr[1][r][2], mtr[1][r][0], mtr[2][r][1], mtr[2][r][2], mtr[2][r][0],
                    mtr[3][r][1], mtr[3][r][2], mtr[3][r][0], mtr[4][r][1], mtr[4][r][2], mtr[4][r][0], mtr[5][r][1], mtr[5][r][2], mtr[5][r][0],
                    mtr[6][r][1], mtr[6][r][2], mtr[6][r][0], mtr[7][r][1], mtr[7][r][2], mtr[7][r][0], mtr[8][r][1], mtr[8][r][2], mtr[8][r][0],
                    mtr[9][r][1], mtr[9][r][2], mtr[9][r][0])
            row = (self.model_name, self.dataset_name, self.gridword_seed, self.splitting_seed, self.shuffling_seed, self.model_seed, rank,
                   mtr[0][r][1], mtr[0][r][2], mtr[0][r][0], mtr[1][r][1], mtr[1][r][2], mtr[1][r][0], mtr[2][r][1], mtr[2][r][2], mtr[2][r][0],
                   mtr[3][r][1], mtr[3][r][2], mtr[3][r][0], mtr[4][r][1], mtr[4][r][2], mtr[4][r][0], mtr[5][r][1], mtr[5][r][2], mtr[5][r][0],
                   mtr[6][r][1], mtr[6][r][2], mtr[6][r][0], mtr[7][r][1], mtr[7][r][2], mtr[7][r][0], mtr[8][r][1], mtr[8][r][2], mtr[8][r][0],
                   mtr[9][r][1], mtr[9][r][2], mtr[9][r][0])
            self.logger.info(summ)
            rows.append(row)
        write_rows(rows, TABLE, FIELDS)


class Evaluator(object):
    def __init__(self, model, source='test'):
        self.model = model
        self.source = source
        self.metrics = []

    def __getattr__(self, name):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        else:
            raise ValueError('`%s` is not defined.' % name)

    def __call__(self, sess, n_epochs):
        observed = []
        predicted = []

        for bs, batch in self.get_eval_batch(self.batch_size, source=self.source):
            prediction = self.sess.run(self.prediction, feed_dict={self.inputs: batch})
            pred_idx = np.argsort(-prediction)
            predicted.append(pred_idx)
            observed.append(batch)

        observed = np.concatenate(observed)
        predicted = np.concatenate(predicted)

        h1_r, h5_r, h10_r, mr_r, mrr_r, h1_f, h5_f, h10_f, mr_f, mrr_f = \
            self._calc_metrics(predicted, observed, self.observed_pool)
        metric = (n_epochs, h1_r, h5_r, h10_r, mr_r, mrr_r, h1_f, h5_f, h10_f, mr_f, mrr_f)
        self.metrics.append(metric)
        return metric

    def _calc_metrics(self, predicted, observed, filter_pool):
        hit_1_raw, hit_5_raw, hit_10_raw = 0., 0., 0.
        hit_1_flt, hit_5_flt, hit_10_flt = 0., 0., 0.
        mr_raw, mrr_raw = 0., 0.
        mr_flt, mrr_flt = 0., 0.

        n_pred = 0
        for pred, obsv in zip(predicted, observed):
            src, dst = obsv[-2], obsv[-1]
            rank_raw = 0
            rank_flt = 0
            for e in pred:
                if e == dst:
                    break
                else:
                    if (src, e) not in filter_pool:
                        rank_flt += 1
                    rank_raw += 1

            if rank_raw == 0:
                hit_1_raw += 1
            if rank_raw < 5:
                hit_5_raw += 1
            if rank_raw < 10:
                hit_10_raw += 1
            mr_raw += (rank_raw + 1)
            mrr_raw += 1. / (rank_raw + 1)

            if rank_flt == 0:
                hit_1_flt += 1
            if rank_flt < 5:
                hit_5_flt += 1
            if rank_flt < 10:
                hit_10_flt += 1
            mr_flt += (rank_flt + 1)
            mrr_flt += 1. / (rank_flt + 1)

            n_pred += 1

        hit_1_raw /= n_pred
        hit_5_raw /= n_pred
        hit_10_raw /= n_pred
        mr_raw /= n_pred
        mrr_raw /= n_pred
        hit_1_flt /= n_pred
        hit_5_flt /= n_pred
        hit_10_flt /= n_pred
        mr_flt /= n_pred
        mrr_flt /= n_pred

        return hit_1_raw, hit_5_raw, hit_10_raw, mr_raw, mrr_raw, \
               hit_1_flt, hit_5_flt, hit_10_flt, mr_flt, mrr_flt
