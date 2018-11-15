from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
import tensorflow as tf

import tf_graph_ops as gops


class ModelBase(object):
    def __init__(self, dataset, hparams, gridworld, seed=None):
        self.dataset = dataset
        self.hparams = hparams
        self.gridworld = gridworld

        self.model_seed = seed
        self.gridworld_seed = self.gridworld.seed
        self.splitting_seed = self.gridworld.splitting_seed
        self.shuffling_seed = self.dataset.shuffling_seed

        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True

        with self.tf_graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.model_seed)
            self._build_model()

    def __getattr__(self, name):
        if hasattr(self.hparams, name):
            return getattr(self.hparams, name)
        elif hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        elif hasattr(self.gridworld, name):
            return getattr(self.gridworld, name)
        else:
            raise ValueError('`%s` is not defined.' % name)

    def _build_model(self):
        # The initialization phase
        self._initialize()

        # The propagation phase
        for step in range(self.n_steps):
            self.propagate(step)

        # The output phase
        self.prediction = self.output_prediction()
        self.loss, self.accuracy = self.compute_loss_and_accuracy(self.prediction)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=0)
        self.init_op = tf.global_variables_initializer()

    def _initialize(self):
        self.inputs = tf.placeholder(tf.int32, [None, 3], name='inputs')  # bs x 3 (trajectory_id, src_id, dst_id)
        _, src, dst = tf.split(self.inputs, 3, axis=1)  # src: bs x 1, dst: bs x 1
        self.src = tf.squeeze(src, axis=1)  # bs
        self.dst = tf.squeeze(dst, axis=1)  # bs
        self.bs = tf.shape(self.src)[0]

        self.node_embs = tf.get_variable('node_emb', [self.n_nodes, self.n_dims],
                                         initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))  # n_nodes x n_dims
        self.initialize()

    def initialize(self):
        raise NotImplementedError("`initialize()` has not been implemented.")

    def propagate(self, step):
        raise NotImplementedError("`propagate()` has not been implemented.")

    def compute_message(self, hidden, g, step):
        raise NotImplementedError("`compute_message` has not been implemented.")

    def aggregate_message(self, message):
        message_aggr = gops.edge_aggregate(message, self.v2_ids, n_nodes=self.n_nodes, sorted=False, op='sum',
                                           name='message_aggr')  # bs x n_nodes x n_dims
        return message_aggr

    def update_node(self, hidden, message_aggr, g, step):
        raise NotImplementedError("`update_node` has not been implemented.")

    def update_global(self, g, hidden_total, message_total, step):
        raise NotImplementedError("`update_node` has not been implemented.")

    def output_prediction(self):
        raise NotImplementedError("`output_prediction` has not been implemented.")

    def compute_loss_and_accuracy(self, prediction):
        raise NotImplementedError("`compute_loss_and_accuracy` has not been implemented.")


class GNModelBase(ModelBase):
    def __init__(self, *args, **kwargs):
        self.att_hiddens = []  # [ bs x n_nodes x n_att_dims ]
        self.info_hiddens = []  # [ bs x n_nodes x n_info_dims ]
        self.globals = []  # [ bs x n_dims ]
        super(GNModelBase, self).__init__(*args, **kwargs)

    def initialize(self):
        self.normalized_ones = np.ones([self.n_att_dims], dtype=np.float32) / np.sqrt(self.n_att_dims)  # n_att_dims

        att_hidden = tf.expand_dims(tf.one_hot(self.src, self.n_nodes), axis=2) * self.normalized_ones  # bs x n_nodes x n_att_dims
        self.att_hiddens.append(att_hidden)

        info_hidden = gops.mlp(self.node_embs, [self.n_info_dims], [tf.tanh], name='info_hidden_init')  # n_nodes x n_info_dims
        info_hidden = tf.tile(tf.expand_dims(info_hidden, axis=0), [self.bs, 1, 1])  # bs x n_nodes x n_info_dims
        self.info_hiddens.append(info_hidden)

        if self.use_global:
            g = tf.zeros([self.bs, self.n_dims])  # bs x n_dims
            self.globals.append(g)

    def propagate(self, step):
        hidden = tf.concat([self.att_hiddens[-1], self.info_hiddens[-1]], axis=2)  # bs x n_nodes x n_dims
        g = self.globals[-1] if self.use_global else None  # bs x n_dims

        message = self.compute_message(hidden, g, step)  # bs x n_edges x n_dims
        message_aggr = self.aggregate_message(message)  # bs x n_nodes x n_dims
        new_hidden = self.update_node(hidden, message_aggr, g, step)  # bs x n_nodes x n_dims

        # att_hidden: bs x n_nodes x n_att_dims, info_hidden: bs x n_nodes x n_info_dims
        att_hidden, info_hidden = tf.split(new_hidden, [self.n_att_dims, self.n_info_dims], axis=2)
        self.att_hiddens.append(att_hidden)
        self.info_hiddens.append(info_hidden)

        if self.use_global:
            hidden_total = tf.reduce_mean(new_hidden, axis=1)  # bs x n_dims
            message_total = tf.reduce_mean(message_aggr, axis=1)  # bs x n_dims
            new_g = self.update_global(g, hidden_total, message_total, step)
            self.globals.append(new_g)

    def output_prediction(self):
        final_att_hidden = self.att_hiddens[-1]  # bs x n_nodes x n_att_dims
        final_att_logits = tf.tensordot(final_att_hidden, self.normalized_ones, [[2], [0]])  # bs x n_nodes
        return final_att_logits

    def compute_loss_and_accuracy(self, prediction):
        pred_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=self.dst))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = pred_loss + reg_loss
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1, output_type=tf.int32), self.dst), tf.float32))
        return loss, accuracy


class AFModelBase(ModelBase):
    def __init__(self, *args, **kwargs):
        self.node_attentions = []  # [ bs x n_nodes ]
        self.hiddens = []  # [ bs x n_nodes x n_dims ]
        self.globals = []  # [ bs x n_dims ]
        super(AFModelBase, self).__init__(*args, **kwargs)

    def initialize(self):
        node_attention = tf.one_hot(self.src, self.n_nodes)  # bs x n_nodes
        self.node_attentions.append(node_attention)

        hidden = gops.mlp(self.node_embs, [self.n_dims], [tf.tanh], name='hidden_init')  # n_nodes x n_dims
        hidden = tf.tile(tf.expand_dims(hidden, axis=0), [self.bs, 1, 1])  # bs x n_nodes x n_dims
        self.hiddens.append(hidden)

        if self.use_global:
            g = tf.zeros([self.bs, self.n_dims])  # bs x n_dims
            self.globals.append(g)

    def propagate(self, step):
        hidden = self.hiddens[-1]  # bs x n_nodes x n_dims
        g = self.globals[-1] if self.use_global else None  # bs x n_dims
        node_attention = self.node_attentions[-1]  # bs x n_nodes

        transition = self.compute_transition(hidden, step)  #  bs x n_edges
        flowing_attention, new_node_attention = self.compute_attentions(node_attention, transition)  # bs x n_edges, bs x n_nodes
        self.node_attentions.append(new_node_attention)

        message = self.compute_message(hidden, g, step)  # bs x n_edges x n_dims
        message = self.attend_message(message, flowing_attention, step)  # bs x n_edges x n_dims
        message_aggr = self.aggregate_message(message)  # bs x n_nodes x n_dims
        new_hidden = self.update_node(hidden, message_aggr, g, step)  # bs x n_nodes x n_dims
        self.hiddens.append(new_hidden)

        if self.use_global:
            hidden_total = tf.reduce_mean(tf.expand_dims(new_node_attention, axis=2) * new_hidden, axis=1)  # bs x n_dims
            message_total = tf.reduce_mean(message_aggr, axis=1)  # bs x n_dims
            new_g = self.update_global(g, hidden_total, message_total, step)
            self.globals.append(new_g)

    def compute_transition(self, hidden, step):
        if step == 0:
            self.transition_logit_fn = gops.EdgeFunction(self.dataset, nn_module='MLP', nn_layers=1, nn_out_units=1,
                                                         nn_out_acti=None, ignore_receiver=False, ignore_edgetype=False,
                                                         use_interacted_feature=True, name='transition_logit_fn')
        transition = self.transition_logit_fn(hidden)  # bs x n_edges x 1
        transition = tf.squeeze(gops.edge_normalize(transition, self.v1_ids), axis=2)  # bs x n_edges
        return transition

    def compute_attentions(self, node_attention, transition):
        node_attention = tf.gather(tf.transpose(node_attention, perm=[1, 0]), self.v1_ids)  # n_edges x bs
        flowing_attention = node_attention * tf.transpose(transition, perm=[1, 0])  # n_edges x bs
        node_attention = tf.transpose(tf.unsorted_segment_sum(flowing_attention, self.v2_ids, self.n_nodes), perm=[1, 0])  # bs x n_nodes
        flowing_attention = tf.transpose(flowing_attention, perm=[1, 0])  # bs x n_edges
        return flowing_attention, node_attention

    def attend_message(self, message, flowing_attention, step):
        raise NotImplementedError("`attend_message` has not been implemented.")

    def output_prediction(self):
        final_node_attention = self.node_attentions[-1]  # bs x n_nodes
        return final_node_attention

    def compute_loss_and_accuracy(self, prediction):
        dst_idx = tf.stack([tf.range(0, self.bs), self.dst], axis=1)
        prediction_prob = tf.gather_nd(prediction, dst_idx)
        pred_loss = tf.reduce_mean(-tf.log(prediction_prob + 1e-12))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = pred_loss + reg_loss
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1, output_type=tf.int32), self.dst), tf.float32))
        return loss, accuracy


class RWModelBase(ModelBase):
    def __init__(self, *args, **kwargs):
        super(RWModelBase, self).__init__(*args, **kwargs)

    def compute_transition(self, hidden, step):
        if step == 0:
            self.transition_logit_fn = gops.EdgeFunction(self.dataset, nn_module='MLP', nn_layers=1, nn_out_units=1,
                                                         nn_out_acti=None, ignore_receiver=False, ignore_edgetype=False,
                                                         use_interacted_feature=True, name='transition_logit_fn')
        transition = self.transition_logit_fn(hidden)  # bs x n_edges x 1
        transition = tf.squeeze(gops.edge_normalize(transition, self.v1_ids), axis=2)  # bs x n_edges
        return transition

    def compute_attentions(self, node_attention, transition):
        node_attention = tf.gather(tf.transpose(node_attention, perm=[1, 0]), self.v1_ids)  # n_edges x bs
        flowing_attention = node_attention * tf.transpose(transition, perm=[1, 0])  # n_edges x bs
        node_attention = tf.transpose(tf.unsorted_segment_sum(flowing_attention, self.v2_ids, self.n_nodes), perm=[1, 0])  # bs x n_nodes
        flowing_attention = tf.transpose(flowing_attention, perm=[1, 0])  # bs x n_edges
        return flowing_attention, node_attention

    def output_prediction(self):
        final_node_attention = self.node_attentions[-1]  # bs x n_nodes
        return final_node_attention

    def compute_loss_and_accuracy(self, prediction):
        dst_idx = tf.stack([tf.range(0, self.bs), self.dst], axis=1)
        prediction_prob = tf.gather_nd(prediction, dst_idx)
        pred_loss = tf.reduce_mean(-tf.log(prediction_prob + 1e-12))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = pred_loss + reg_loss
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1, output_type=tf.int32), self.dst), tf.float32))
        return loss, accuracy

