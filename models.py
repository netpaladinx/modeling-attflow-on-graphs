from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from model_base import GNModelBase, AFModelBase, RWModelBase
import tf_graph_ops as gops


class FullGNMixin(object):
    def compute_message(self, hidden, g, step):
        if step == 0:
            self.message_fn = gops.EdgeFunction(self.dataset, nn_module='MLP', nn_layers=1, nn_out_units=self.n_dims,
                                                nn_out_acti=tf.tanh, ignore_receiver=False, ignore_edgetype=False,
                                                use_interacted_feature=False, name='message_fn')
        message = self.message_fn(hidden, global_state=g)  # bs x n_edges x n_dims
        return message

    def update_node(self, hidden, message_aggr, g, step):
        if step == 0:
            self.node_fn = gops.NodeFunction(self.dataset, nn_module='MLP', nn_layers=1, nn_out_units=self.n_dims,
                                             nn_out_acti=tf.tanh, ignore_nodetype=True, name='node_update_fn')
        new_hidden = self.node_fn(hidden, node_inps=message_aggr, node_embs=self.node_embs, global_state=g)  # bs x n_nodes x n_dims
        return new_hidden

    def update_global(self, g, hidden_total, message_total, step):
        inp = tf.concat([g, hidden_total, message_total], axis=1)  # bs x (3*n_dims)
        new_g = gops.mlp(inp, [self.n_dims], [tf.tanh], reuse=None if step == 0 else True, name='global_update_fn')  # bs x n_dims
        return new_g


class GGNNMixin(object):
    def compute_message(self, hidden, g, step):
        if step == 0:
            self.message_fn = gops.EdgeFunction(self.dataset, nn_module='MLP', nn_layers=1, nn_out_units=self.n_dims,
                                                nn_out_acti=None, ignore_receiver=True, ignore_edgetype=False,
                                                use_interacted_feature=False, name='message_fn')
        message = self.message_fn(hidden)  # bs x n_edges x n_dims
        return message

    def update_node(self, hidden, message_aggr, g, step):
        if step == 0:
            self.node_fn = gops.NodeFunction(self.dataset, nn_module='GRU', ignore_nodetype=True, name='node_update_fn')
        new_hidden = self.node_fn(hidden, node_inps=message_aggr, node_embs=self.node_embs)  # bs x n_nodes x n_dims
        return new_hidden


class GATMixin(object):
    def compute_message(self, hidden, g, step):
        with tf.variable_scope('message', reuse=None if step == 0 else True):
            hidden_v1 = tf.transpose(tf.gather(hidden, self.v1_ids, axis=1), perm=[1, 0, 2])  # n_edges x bs x n_dims
            hidden_v2 = tf.transpose(tf.gather(hidden, self.v2_ids, axis=1), perm=[1, 0, 2])  # n_edges x bs x n_dims

            message_li = []
            for i in range(self.n_heads):
                weight = tf.get_variable('weight_%d' % i, shape=[self.n_etypes, self.n_dims, self.n_selfatt_dims],
                                         initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))
                weight_ga = tf.gather(weight, self.etype_ids)  # n_edges x n_dims x n_selfatt_dims
                hidden_v1v2 = tf.concat([tf.matmul(hidden_v1, weight_ga), tf.matmul(hidden_v2, weight_ga)], axis=2)  # n_edges x bs x (2*n_selfatt_dims)

                vec_a = tf.get_variable('vec_a_%d' % i, shape=[self.n_etypes, 2 * self.n_selfatt_dims, 1],
                                        initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))
                vec_a_ga = tf.gather(vec_a, self.etype_ids)  # n_edges x (2*n_selfatt_dims) x 1
                att_logits = tf.nn.leaky_relu(tf.matmul(hidden_v1v2, vec_a_ga))  # n_edges x bs x 1

                att_logits = tf.transpose(att_logits, perm=[1, 0, 2])  # bs x n_edges x 1
                att = gops.edge_normalize(att_logits, self.v1_ids)  # bs x n_edges x 1
                att = tf.transpose(att, perm=[1, 0, 2])  # n_edges x bs x 1

                message = tf.transpose(tf.matmul(hidden_v1, weight_ga) * att, perm=[1, 0, 2])  # bs x n_edges x n_selfatt_dims
                message_li.append(message)

            return message_li

    def aggregate_message(self, message_li):
        message_aggr_li = []
        for i in range(self.n_heads):
            message_aggr = gops.edge_aggregate(message_li[i], self.v2_ids, n_nodes=self.n_nodes, sorted=False, op='sum',
                                               name='message_aggr')  # bs x n_nodes x n_selfatt_dims
            message_aggr_li.append(message_aggr)
        return message_aggr_li

    def update_node(self, hidden, message_aggr_li, g, step):
        if step == 0:
            self.node_fn = gops.NodeFunction(self.dataset, nn_module='GRU', ignore_nodetype=True, name='node_update_fn')
        message_aggr = tf.concat(message_aggr_li, axis=2)  # bs x n_nodes x n_dims
        new_hidden = self.node_fn(hidden, node_inps=message_aggr, node_embs=self.node_embs)  # bs x n_nodes x n_dims
        return new_hidden


class FullGN(FullGNMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = True
        super(FullGN, self).__init__(*args, **kwargs)


class FullGN_NoAct(FullGNMixin, AFModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = True
        super(FullGN_NoAct, self).__init__(*args, **kwargs)

    def attend_message(self, message, flowing_attention, step):
        return message


class FullGN_Mul(FullGNMixin, AFModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = True
        super(FullGN_Mul, self).__init__(*args, **kwargs)

    def attend_message(self, message, flowing_attention, step):
        message = message * tf.expand_dims(flowing_attention, axis=2)  # bs x n_edges x n_dims
        return message


class FullGN_MulMlp(FullGNMixin, AFModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = True
        super(FullGN_MulMlp, self).__init__(*args, **kwargs)

    def attend_message(self, message, flowing_attention, step):
        message = message * tf.expand_dims(flowing_attention, axis=2)  # bs x n_edges x n_dims
        message = gops.mlp(message, [self.n_dims], [tf.tanh], reuse=None if step == 0 else True,
                           name='flowing_message')  # bs x n_edges x n_dims
        return message


class GGNN(GGNNMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GGNN, self).__init__(*args, **kwargs)


class GGNN_NoAct(GGNNMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GGNN_NoAct, self).__init__(*args, **kwargs)

    def attend_message(self, message, flowing_attention, step):
        return message


class GGNN_Mul(GGNNMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GGNN_Mul, self).__init__(*args, **kwargs)

    def attend_message(self, message, flowing_attention, step):
        message = message * tf.expand_dims(flowing_attention, axis=2)  # bs x n_edges x n_dims
        return message


class GGNN_MulMlp(GGNNMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GGNN_MulMlp, self).__init__(*args, **kwargs)

    def attend_message(self, message, flowing_attention, step):
        message = message * tf.expand_dims(flowing_attention, axis=2)  # bs x n_edges x n_dims
        message = gops.mlp(message, [self.n_dims], [tf.tanh], reuse=None if step == 0 else True,
                           name='flowing_message')  # bs x n_edges x n_dims
        return message


class GAT(GATMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GAT, self).__init__(*args, **kwargs)


class GAT_NoAct(GATMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GAT_NoAct, self).__init__(*args, **kwargs)

    def attend_message(self, message_li, flowing_attention, step):
        return message_li


class GAT_Mul(GATMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GAT_Mul, self).__init__(*args, **kwargs)

    def attend_message(self, message_li, flowing_attention, step):
        att_message_li = []
        for message in message_li:
            att_message = message * tf.expand_dims(flowing_attention, axis=2)  # bs x n_edges x n_selfatt_dims
            att_message_li.append(att_message)
        return att_message_li

class GAT_MulMlp(GATMixin, GNModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.use_global = False
        super(GAT_MulMlp, self).__init__(*args, **kwargs)

    def attend_message(self, message_li, flowing_attention, step):
        att_message_li = []
        for message in message_li:
            att_message = message * tf.expand_dims(flowing_attention, axis=2)  # bs x n_edges x n_selfatt_dims
            att_message = gops.mlp(att_message, [self.n_selfatt_dims], [tf.tanh], reuse=None if step == 0 else True,
                                   name='flowing_message')  # bs x n_edges x n_selfatt_dims
            att_message_li.append(att_message)
        return att_message_li


class RW_Stationary(RWModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.node_attentions = []  # [ bs x n_nodes ]
        super(RW_Stationary, self).__init__(*args, **kwargs)

    def initialize(self):
        node_attention = tf.one_hot(self.src, self.n_nodes)  # bs x n_nodes
        self.node_attentions.append(node_attention)
        self.hidden = tf.expand_dims(self.node_embs, axis=0)  # 1 x n_nodes x n_dims

    def propagate(self, step):
        node_attention = self.node_attentions[-1]  # bs x n_nodes

        transition = self.compute_transition(self.hidden, step)  # 1 x n_edges
        flowing_attention, new_node_attention = self.compute_attentions(node_attention, transition)  # bs x n_edges, bs x n_nodes
        self.node_attentions.append(new_node_attention)


class RW_Dynamic(RWModelBase):
    def __init__(self, *args, **kwargs):
        self.model_name = self.__class__.__name__
        self.node_attentions = []  # [ bs x n_nodes ]
        self.hiddens = []  # [ bs x n_nodes x n_dims ]
        self.globals = []  # [ bs x n_dims ]
        super(RW_Dynamic, self).__init__(*args, **kwargs)

    def initialize(self):
        node_attention = tf.one_hot(self.src, self.n_nodes)  # bs x n_nodes
        self.node_attentions.append(node_attention)

        hidden = gops.mlp(self.node_embs, [self.n_dims], [tf.tanh], name='hidden_init')  # n_nodes x n_dims
        hidden = tf.tile(tf.expand_dims(hidden, axis=0), [self.bs, 1, 1])  # bs x n_nodes x n_dims
        self.hiddens.append(hidden)

        g = tf.zeros([self.bs, self.n_dims])  # bs x n_dims
        self.globals.append(g)

    def propagate(self, step):
        hidden = self.hiddens[-1]  # bs x n_nodes x n_dims
        g = self.globals[-1]  # bs x n_dims
        node_attention = self.node_attentions[-1]  # bs x n_nodes

        transition = self.compute_transition(hidden, step)  # bs x edges
        flowing_attention, new_node_attention = self.compute_attentions(node_attention, transition)  # bs x n_edges, bs x n_nodes
        self.node_attentions.append(new_node_attention)

        new_hidden = self.update_node_2(hidden, g, step)  # bs x n_nodes x n_dims
        self.hiddens.append(new_hidden)

        hidden_total = tf.reduce_mean(tf.expand_dims(new_node_attention, axis=2) * new_hidden, axis=1)  # bs x n_dims
        new_g = self.update_global_2(g, hidden_total, step)
        self.globals.append(new_g)

    def update_node_2(self, hidden, g, step):
        if step == 0:
            self.node_fn = gops.NodeFunction(self.dataset, nn_module='MLP', nn_layers=1, nn_out_units=self.n_dims,
                                             nn_out_acti=tf.tanh, ignore_nodetype=True, name='node_update_fn')
        new_hidden = self.node_fn(hidden, node_embs=self.node_embs, global_state=g)  # bs x n_nodes x n_dims
        return new_hidden

    def update_global_2(self, g, hidden_total, step):
        inp = tf.concat([g, hidden_total], axis=1)  # bs x (2*n_dims)
        new_g = gops.mlp(inp, [self.n_dims], [tf.tanh], reuse=None if step == 0 else True, name='global_update_fn')  # bs x n_dims
        return new_g
