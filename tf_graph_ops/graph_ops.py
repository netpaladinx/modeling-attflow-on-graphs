from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tf_graph_ops.basic_ops import mlp, mmlp, gru, mgru


class EdgeFunction(object):
    def __init__(self, graph,
                 nn_module='MLP',
                 nn_layers=1,
                 nn_mid_units=128,
                 nn_mid_acti=tf.tanh,
                 nn_out_units=1,
                 nn_out_acti=None,
                 ignore_receiver=False,
                 ignore_edgetype=False,
                 use_interacted_feature=False,
                 name='edge_fn'):
        """ Compute output for each edge

        Args:
            graph: stores information of nodes and edges
            nn_module: type of the neural network building block used for the edge-level function
        """
        self.graph = graph
        self.nn_module = nn_module
        self.nn_layers = nn_layers
        self.nn_mid_units = nn_mid_units
        self.nn_mid_acti = nn_mid_acti
        self.nn_out_units = nn_out_units
        self.nn_out_acti = nn_out_acti
        self.ignore_receiver = ignore_receiver
        self.ignore_edgetype = ignore_edgetype
        self.use_interacted_feature = use_interacted_feature
        self.name = name

        self.reuse = None

    def __call__(self, node_states, global_state=None, edge_states=None):
        """
        Args:
            node_states: batch_size x n_nodes x n_node_dims
            global_state: batch_size x n_global_dims
            edge_states: batch_size x n_edges x n_edge_dims (`n_edge_dims` should be equal to `nn_out_units`)
        Returns:
            edge_states: batch_size x n_edges x n_edge_dims
        """
        inputs = []

        if edge_states is not None:
            if self.nn_module == 'MLP':
                inputs.append(edge_states)
        else:
            if self.nn_module == 'GRU':
                raise ValueError("`edge_states` should not be None for GRU")

        node_v1 = tf.gather(node_states, self.graph.v1_ids, axis=1)  # batch_size x n_edges x n_node_dims
        inputs.append(node_v1)

        if not self.ignore_receiver:
            node_v2 = tf.gather(node_states, self.graph.v2_ids, axis=1)  # batch_size x n_edges x n_node_dims
            inputs.append(node_v2)

            if self.use_interacted_feature:
                node_v12 = node_v1 * node_v2  # batch_size x n_edges x n_node_dims
                inputs.append(node_v12)

        if global_state is not None:
            global_s = tf.tile(tf.expand_dims(global_state, axis=1), [1, self.graph.n_edges, 1])  # batch_size x n_edges x n_node_dims
            inputs.append(global_s)

        inputs = tf.concat(inputs, axis=2)  # batch_size x n_edges x n_comb_dims
        inputs = tf.transpose(inputs, perm=[1, 0, 2])  # n_edges x batch_size x n_comb_dims

        if self.nn_module == 'MLP':
            n_units_li = [self.nn_mid_units] * (self.nn_layers - 1) + [self.nn_out_units]
            acti_li = [self.nn_mid_acti] * (self.nn_layers - 1) + [self.nn_out_acti]

            if not self.ignore_edgetype:
                edge_states = mmlp(inputs, n_units_li, acti_li, self.graph.etype_ids, self.graph.n_etypes,
                                  reuse=self.reuse, name='%s_mmlp' % self.name)  # n_edges x batch_size x n_edge_dims
            else:
                edge_states = mlp(inputs, n_units_li, acti_li,
                                 reuse=self.reuse, name='%s_mlp' % self.name)  # n_edges x batch_size x n_edge_dims
            edge_states = tf.transpose(edge_states, perm=[1, 0, 2])  # batch_size x n_edges x n_edge_dims

        elif self.nn_module == 'GRU':
            if not self.ignore_edgetype:
                edge_states = mgru(edge_states, inputs, self.graph.etype_ids, self.graph.n_etypes,
                                   reuse=self.reuse, name='%s_mgru' % self.name)  # n_edges x batch_size x n_edge_dims
            else:
                edge_states = gru(edge_states, inputs, reuse=self.reuse, name='%s_gru' % self.name)  # n_edges x batch_size x n_edge_dims
        else:
            raise ValueError('We do not support %s yet' % self.nn_module)
        edge_states = tf.transpose(edge_states, perm=[1, 0, 2])  # batch_size x n_edges x n_edge_dims

        self.reuse = True
        return edge_states

def edge_aggregate(edge_states, receiver_node_ids, n_nodes=None, sorted=True, op='sum', name='edge_aggr'):
    """
    Args:
        edge_states: batch_size x n_edges x n_edge_dims
        receiver_node_ids: n_edges
        sorted: the list receiver_node_ids is sorted or not
    Returns:
        aggr_states: batch_size x n_nodes x n_edge_dims
    """
    if sorted:
        aggr_op = tf.segment_sum if op == 'sum' else \
            tf.segment_max if op == 'max' else \
            tf.segment_min if op == 'min' else \
            tf.segment_mean if op == 'mean' else tf.segment_prod
        aggr_states = aggr_op(tf.transpose(edge_states, perm=[1, 0, 2]), receiver_node_ids, name=name)  # n_nodes x batch_size x n_edge_dims
    else:
        if n_nodes is None:
            raise ValueError('`n_nodes` should not be None')
        aggr_op = tf.unsorted_segment_sum if op == 'sum' else \
            tf.unsorted_segment_max if op == 'max' else \
            tf.unsorted_segment_min if op == 'min' else \
            tf.unsorted_segment_mean if op == 'mean' else tf.unsorted_segment_prod
        aggr_states = aggr_op(tf.transpose(edge_states, perm=[1, 0, 2]), receiver_node_ids, n_nodes, name=name)  # n_nodes x batch_size x n_edge_dims

    aggr_states = tf.transpose(aggr_states, perm=[1, 0, 2])  # batch_size x n_nodes x n_edge_dims
    return aggr_states


def edge_normalize(edge_states, sender_node_ids, n_nodes=None, sorted=True):
    """
    Args:
        edge_states: batch_size x n_edges x n_edge_dims
        sender_node_ids: n_edges
        sorted: the list sender_node_ids is sorted or not
    Returns:
        edge_states_norm: batch_size x n_nodes x n_edge_dims
    """
    if sorted:
        edge_states_max = tf.segment_max(tf.transpose(edge_states, perm=[1, 0, 2]), sender_node_ids)  # n_nodes x batch_size x n_edge_dims
        edge_states_max = tf.gather(edge_states_max, sender_node_ids)  # n_edges x batch_size x n_edge_dims
        edge_states_exp = tf.exp(edge_states - edge_states_max)  # n_edges x batch_size x n_edge_dims
        edge_states_sumexp = tf.segment_sum(edge_states_exp, sender_node_ids)  # n_nodes x batch_size x n_edge_dims
        edge_states_sumexp = tf.gather(edge_states_sumexp, sender_node_ids)  # n_edges x batch_size x n_edge_dims
        edge_states_norm = edge_states_exp / edge_states_sumexp  # n_edges x batch_size x n_edge_dims
    else:
        if n_nodes is None:
            raise ValueError('`n_nodes` should not be None')
        edge_states_max = tf.unsorted_segment_max(tf.transpose(edge_states, perm=[1, 0, 2]), sender_node_ids, n_nodes)  # n_nodes x batch_size x n_edge_dims
        edge_states_max = tf.gather(edge_states_max, sender_node_ids)  # n_edges x batch_size x n_edge_dims
        edge_states_exp = tf.exp(edge_states - edge_states_max)  # n_edges x batch_size x n_edge_dims
        edge_states_sumexp = tf.unsorted_segment_sum(edge_states_exp, sender_node_ids, n_nodes)  # n_nodes x batch_size x n_edge_dims
        edge_states_sumexp = tf.gather(edge_states_sumexp, sender_node_ids)  # n_edges x batch_size x n_edge_dims
        edge_states_norm = edge_states_exp / edge_states_sumexp  # n_edges x batch_size x n_edge_dims
    return edge_states_norm


class NodeFunction(object):
    def __init__(self, graph,
                 nn_module='MLP',
                 nn_layers=1,
                 nn_mid_units=128,
                 nn_mid_acti=tf.tanh,
                 nn_out_units=1,
                 nn_out_acti=None,
                 ignore_nodetype=True,
                 name='node_fn'):
        """ Compute output for each node

        Args:
            graph: stores information of nodes and edges
            nn_module: type of the neural network building block used for the node-level function
        """
        self.graph = graph
        self.nn_module = nn_module
        self.nn_layers = nn_layers
        self.nn_mid_units = nn_mid_units
        self.nn_mid_acti = nn_mid_acti
        self.nn_out_units = nn_out_units
        self.nn_out_acti = nn_out_acti
        self.ignore_nodetype = ignore_nodetype
        self.name = name

        self.reuse = None

    def __call__(self, node_states, node_inps, node_embs=None, global_state=None):
        """
        Args:
            node_states: batch_size x n_nodes x n_node_dims
            node_inps: batch_size x n_nodes x n_in_dims
            node_embs: n_nodes x n_node_dims
            global_state: batch_size x n_global_dims
        Returns:
            node_states: batch_size x n_nodes x n_node_dims
        """
        inputs = []
        bs = tf.shape(node_states)[0]

        if self.nn_module == 'MLP':
            inputs.append(node_states)

        inputs.append(node_inps)

        if node_embs is not None:
            node_e = tf.tile(tf.expand_dims(node_embs, axis=0), [bs, 1, 1])  # batch_size x n_nodes x n_node_dims
            inputs.append(node_e)

        if global_state is not None:
            global_s = tf.tile(tf.expand_dims(global_state, axis=1), [1, self.graph.n_nodes, 1])  # batch_size x n_nodes x n_node_dims
            inputs.append(global_s)

        inputs = tf.concat(inputs, axis=2)  # batch_size x n_nodes x n_comb_dims
        inputs = tf.transpose(inputs, perm=[1, 0, 2])  # n_nodes x batch_size x n_comb_dims

        if self.nn_module == 'MLP':
            n_units_li = [self.nn_mid_units] * (self.nn_layers - 1) + [self.nn_out_units]
            acti_li = [self.nn_mid_acti] * (self.nn_layers - 1) + [self.nn_out_acti]

            if not self.ignore_nodetype:
                node_states = mmlp(inputs, n_units_li, acti_li, self.graph.ntype_ids, self.graph.n_ntypes,
                                   reuse=self.reuse, name='%s_mmlp' % self.name)  # n_nodes x batch_size x n_node_dims
            else:
                node_states = mlp(inputs, n_units_li, acti_li,
                                  reuse=self.reuse, name='%s_mlp' % self.name)  # n_nodes x batch_size x n_node_dims
            node_states = tf.transpose(node_states, perm=[1, 0, 2])  # batch_size x n_nodes x n_node_dims

        elif self.nn_module == 'GRU':
            if not self.ignore_nodetype:
                node_states = mgru(node_states, inputs, self.graph.ntype_ids, self.graph.n_ntypes,
                                   reuse=self.reuse, name='%s_mgru' % self.name)  # n_nodes x batch_size x n_node_dims
            else:
                node_states = gru(node_states, inputs, reuse=self.reuse, name='%s_gru' % self.name)  # n_nodes x batch_size x n_node_dims
        else:
            raise ValueError('We do not support %s yet' % self.nn_module)
        node_states = tf.transpose(node_states, perm=[1, 0, 2])  # batch_size x n_nodes x n_node_dims

        self.reuse = True
        return node_states
