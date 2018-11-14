from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def mlp(inp, n_units_li, acti_li, reuse=None, name='mlp'):
    """ Implement a single MLP

    Args:
        inp: d0 x ... x dk x n_in_dims
        n_units_li: a list of numbers of units
        acti_li: a list of activation functions or None
    Returns:
        out: d0 x ... x dk x n_out_dims
    """
    with tf.variable_scope(name, reuse=reuse):
        out = inp
        for i, n_units in enumerate(n_units_li):
            acti = acti_li[i]
            kernel = tf.get_variable('kernel_%d' % i, shape=[out.get_shape()[-1], n_units],
                                     initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))  # n_dims x n_next_dims
            bias = tf.get_variable('bias_%d' % i, shape=[n_units], initializer=tf.zeros_initializer())  # n_next_dims

            out = tf.tensordot(inp, kernel, [[-1], [0]]) + bias  # d0 x ... x dk x n_next_dims
            if acti is not None:
                out = acti(out)
        return out  # d0 x ... x dk x n_out_dims

def mmlp(inp, n_units_li, acti_li, mlp_ids, n_mlps, reuse=None, name='mmlp'):
    """ Implement multiple MLPs corresponding to `mlp_ids`

    Args:
        inp: mlp_ids_len x d1 x ... x dk x n_in_dims
        n_units_li: a list of number of units
        acti_li: a list of activation functions or None
        mlp_ids: a list of mlp ids, where len(mlp_ids) == shape(inp)[0]
        n_mlps: number of MLPs
    Returns:
        out: mlp_ids_len x d1 x ... x dk x n_out_dims
    """
    with tf.variable_scope(name, reuse=reuse):
        mlp_ids_len = tf.shape(inp)[0]
        out = tf.reshape(inp, [mlp_ids_len, -1, n_mlps])  # mlp_ids_len x (d1*...*dk) x n_in_dims
        for i, n_units in enumerate(n_units_li):
            acti = acti_li[i]
            kernel = tf.get_variable('kernel_%d' % i, shape=[n_mlps, out.get_shape()[-1], n_units],
                                     initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))  # n_mlps x n_dims x n_next_dims
            bias = tf.get_variable('bias_%d' % i, shape=[n_mlps, n_units], initializer=tf.zeros_initializer())  # n_mlps x n_next_dims

            kernel_ga = tf.gather(kernel, mlp_ids)  # mlp_ids_len x n_dims x n_next_dims
            bias_ga = tf.gather(bias, mlp_ids)  # mlp_ids_len x n_next_dims

            out = tf.matmul(out, kernel_ga) + tf.expand_dims(bias_ga, axis=1)  # mlp_ids_len x (d1*...*dk) x n_next_dims
            if acti is not None:
                out = acti(out)
        out = tf.reshape(out, tf.concat([tf.shape(inp)[:-1], [n_units_li[-1]]], 0))  # mlp_ids_len x d1 x ... x dk x n_out_dims
        return out  # mlp_ids_len x d1 x ... x dk x n_out_dims

def gru(state, inp, acti=tf.tanh, reuse=None, name='gru'):
    """ Implement a single GRU

    Args:
        state: d0 x ... x dk x n_dims
        inp: d0 x ... x dk x n_in_dims
        acti: activation function or None
    Returns:
        state: d0 x ... x dk x n_dims
    """
    with tf.variable_scope(name, reuse=reuse):
        n_dims = state.get_shape()[-1]
        n_in_dims = inp.get_shape()[-1]

        gate_kernel = tf.get_variable('gate_kernel', shape=[n_in_dims + n_dims, 2 * n_dims],
                                      initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))  # (n_in_dims+n_dims) x (2*n_dims)
        gate_bias = tf.get_variable('gate_bias', shape=[2 * n_dims], initializer=tf.ones_initializer())  # (2*n_dims)
        candidate_kernel = tf.get_variable('candidate_kernel', shape=[n_in_dims + n_dims, n_dims],
                                           initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))  # (n_in_dims+n_dims) x n_dims
        candidate_bias = tf.get_variable('candidate_bias', shape=[n_dims], initializer=tf.zeros_initializer())  # n_dims

        inp_state = tf.concat([inp, state], -1)  # d0 x ... x dk x (n_in_dims+n_dims)
        gate = tf.sigmoid(tf.tensordot(inp_state, gate_kernel, [[-1], [0]]) + gate_bias)  # d0 x ... x dk x (2*n_dims)
        r_gate, u_gate = tf.split(gate, 2, axis=-1)  # r_gate: d0 x ... x dk x n_dims, u_gate: d0 x ... x dk x n_dims
        r_state = r_gate * state  # d0 x ... x dk x n_dims

        inp_state = tf.concat([inp, r_state], -1)  # d0 x ... x dk x n_dims
        candidate = acti(tf.tensordot(inp_state, candidate_kernel, [[-1], [0]]) + candidate_bias)  # d0 x ... x dk x n_dims

        new_state = u_gate * state + (1 - u_gate) * candidate  # d0 x ... x dk x n_dims
        return new_state

def mgru(state, inp, gru_ids, n_grus, acti=tf.tanh, reuse=None, name='mgru'):
    """ Implement multiple GRUs corresponding to gru_ids

    Args:
        state: gru_ids_len x d1 x ... x dk x n_dims
        inp: gru_ids_len x d1 x ... x dk x n_in_dims
        gru_ids: a list of gru ids, when len(gru_ids) == shape(inp)[0]
        n_grus: number of GRUs
        acti: activation function or None
    Returns:
        state: gru_ids_len x d1 x ... x dk x n_dims
    """
    with tf.variable_scope(name, reuse=reuse):
        n_dims = state.get_shape()[-1]
        n_in_dims = inp.get_shape()[-1]
        gru_ids_len = inp.get_shape()[0]

        inp_re = tf.reshape(inp, [gru_ids_len, -1, n_in_dims])  # gru_ids_len x (d1*...*dk) x n_in_dims
        state_re = tf.reshape(state, [gru_ids_len, -1, n_dims])  # gru_ids_len x (d1*...*dk) x n_dims

        gate_kernel = tf.get_variable('gate_kernel', shape=[n_grus, n_in_dims + n_dims, 2 * n_dims],
                                      initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))  # n_grus x (n_in_dims+n_dims) x (2*n_dims)
        gate_bias = tf.get_variable('gate_bias', shape=[n_grus, 2 * n_dims], initializer=tf.ones_initializer())  # n_grus x (2*n_dims)
        candidate_kernel = tf.get_variable('candidate_kernel', shape=[n_grus, n_in_dims + n_dims, n_dims],
                                         initializer=tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform'))  # n_grus x (n_in_dims+n_dims) x n_dims
        candidate_bias = tf.get_variable('candidate_bias', shape=[n_grus, n_dims], initializer=tf.zeros_initializer())  # n_grus x n_dims

        gate_kernel_ga = tf.gather(gate_kernel, gru_ids)  # gru_ids_len x (n_in_dims+n_dims) x (2*n_dims)
        gate_bias_ga = tf.gather(gate_bias, gru_ids)  # gru_ids_len x (2*n_dims)
        candidate_kernel_ga = tf.gather(candidate_kernel, gru_ids)  # gru_ids_len x (n_in_dims+n_dims) x n_dims
        candidate_bias_ga = tf.gather(candidate_bias, gru_ids)  # gru_ids_len x n_dims

        inp_state = tf.concat([inp_re, state_re], -1)  # gru_ids_len x (d1*...*dk) x (n_in_dims+n_dims)
        gate = tf.sigmoid(tf.matmul(inp_state, gate_kernel_ga) + tf.expand_dims(gate_bias_ga, axis=1))  # gru_ids_len x (d1*...*dk) x (2*n_dims)
        r_gate, u_gate = tf.split(gate, 2, axis=-1)  # r_gate: gru_ids_len x (d1*...*dk) x n_dims, u_gate: gru_ids_len x (d1*...*dk) x n_dims
        r_state = r_gate * state_re  # gru_ids_len x (d1*...*dk) x n_dims

        inp_state = tf.concat([inp_re, r_state], -1)  # gru_ids_len x (d1*...*dk) x (n_in_dims+n_dims)
        candidate = acti(tf.matmul(inp_state, candidate_kernel_ga) + tf.expand_dims(candidate_bias_ga, axis=1))  # gru_ids_len x (d1*...*dk) x n_dims

        new_state = u_gate * state_re + (1 - u_gate) * candidate  # gru_ids_len x (d1*...*dk) x n_dims
        new_state = tf.reshape(new_state, tf.shape(state))  # gru_ids_len x d1 x ... x dk x n_dims
        return new_state



