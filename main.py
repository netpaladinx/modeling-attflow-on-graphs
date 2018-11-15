from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import logging

import tensorflow as tf

from gridworld import GridWorld
from dataset import Dataset
from train import Trainer
from models import FullGN, GGNN, GAT, FullGN_NoAct, FullGN_Mul, FullGN_MulMlp, GGNN_NoAct, GGNN_Mul, GGNN_MulMlp, \
    GAT_NoAct, GAT_Mul, GAT_MulMlp, RW_Stationary, RW_Dynamic

# ===== Data Generation =====

# Configuration for generating data
class Config(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._attrs = {}

    def add(self, attr, name, config):
        self._kwargs.update(config.get_kwargs())
        self._attrs[attr] = name

    def get_kwargs(self):
        return self._kwargs

    def get_name(self):
        return '%s_%s_%s_%s' % (self._attrs['direction'], self._attrs['size'], self._attrs['drop'], self._attrs['std'])

BASE_DIR = 'dataset'
CONFIG_BASE = Config(n_rollouts=10, base_dir=BASE_DIR)
CONFIG_STDS = {'STD0.2': Config(sigma=0.2),
               'STD0.5': Config(sigma=0.5)
              }
CONFIG_DROPS = {'NDRP': Config(node_drop=0.1),
                'EDRP': Config(edge_drop=0.2)
               }
CONFIG_DIRECTIONS = {'LINE': Config(omega=1., alpha=1., lambda_1=0., lambda_2=0., varphi=0., a_1=0., b_1=1., a_2=0., b_2=0.4),
                     'SINE': Config(omega=0.4, alpha=1., lambda_1=0., lambda_2=0., varphi=1.6, a_1=0., b_1=1., a_2=1., b_2=0.),
                     'LOCATION': Config(omega=0., alpha=1., lambda_1=0.2, lambda_2=0.2, varphi=0., a_1=1., b_1=0., a_2=1., b_2=0.),
                     'HISTORY': Config(omega=0., alpha=1., lambda_1=0.2, lambda_2=0.2, varphi=0., a_1=1., b_1=0, a_2=1., b_2=0., depend_on_history=True, history_op='max')
                    }
GRIDWORLD_SEEDS = [1111, 2222, 3333, 4444, 5555]
SPLITTING_SEED = 12345

# Generate 16 dataset groups with a specified `size` and `max_steps`
def generate_data(FLAGS):
    name_size = 'SZ%d-STP%d' % (FLAGS.sz, FLAGS.stp)
    config_size = Config(size=FLAGS.sz, max_steps=FLAGS.stp)

    for name_std, config_std in CONFIG_STDS.iteritems():
        for name_drop, config_drop in CONFIG_DROPS.iteritems():
            for name_direction, config_direction in CONFIG_DIRECTIONS.iteritems():
                config = Config()
                config.add('base', 'base', CONFIG_BASE)
                config.add('size', name_size, config_size)
                config.add('direction', name_direction, config_direction)
                config.add('drop', name_drop, config_drop)
                config.add('std', name_std, config_std)
                gridworld = GridWorld(name=config.get_name(), **config.get_kwargs())

                for seed in GRIDWORLD_SEEDS:
                    data_dir = '%s-SEED%d' % (config.get_name(), seed)
                    gridworld.generate(data_dir=data_dir, seed=seed, splitting_seed=SPLITTING_SEED)


# ===== Model Running =====

COMMON_HPARAMS = tf.contrib.training.HParams(
    batch_size=None,
    n_steps=None,
    n_info_dims=None,
    n_att_dims=None,
    n_dims=None,
    weight_decay=0.00001,
    learning_rates=[0.0005, 0.0004, 0.0003, 0.0002, 0.0001],
    max_epochs=None,
    checkpoint=None,
    n_heads=None,
    n_selfatt_dims=None
)

SHUFFLING_SEEDS = [1234, 5678]
MODEL_SEED = 98765

def _run(FLAGS, model_cls):
    logger = logging.getLogger('Trainer_%s' % model_cls.__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('%s.log' % model_cls.__name__)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] ## %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    hparams = tf.contrib.training.HParams(**COMMON_HPARAMS.values())
    hparams.set_hparam('batch_size', FLAGS.bs)
    hparams.set_hparam('n_steps', FLAGS.stp)
    hparams.set_hparam('n_dims', FLAGS.dims)
    hparams.set_hparam('n_info_dims', FLAGS.info_dims)
    hparams.set_hparam('n_att_dims', FLAGS.att_dims)
    hparams.set_hparam('max_epochs', FLAGS.epochs)
    hparams.set_hparam('checkpoint', FLAGS.ckpt)
    hparams.set_hparam('n_heads', FLAGS.heads)
    hparams.set_hparam('n_selfatt_dims', FLAGS.selfatt_dims)

    assert hparams.n_dims == hparams.n_info_dims + hparams.n_att_dims, "`n_dims` should be equal to the sum of `n_info_dims` and `n_att_dims`"
    assert hparams.n_dims == hparams.n_heads * hparams.n_selfatt_dims, "`n_dims` should be equal to the product of `n_heads` and `n_selfatt_dims`"

    name_size = 'SZ%d-STP%d' % (FLAGS.sz, FLAGS.stp)
    config_size = Config(size=FLAGS.sz, max_steps=FLAGS.stp)

    for name_std, config_std in CONFIG_STDS.iteritems():
        for name_drop, config_drop in CONFIG_DROPS.iteritems():
            for name_direction, config_direction in CONFIG_DIRECTIONS.iteritems():
                config = Config()
                config.add('base', 'base', CONFIG_BASE)
                config.add('size', name_size, config_size)
                config.add('direction', name_direction, config_direction)
                config.add('drop', name_drop, config_drop)
                config.add('std', name_std, config_std)
                gridworld = GridWorld(name=config.get_name(), **config.get_kwargs())

                for seed in GRIDWORLD_SEEDS:
                    data_dir = '%s-SEED%d' % (config.get_name(), seed)
                    gridworld.load(data_dir, seed=seed, splitting_seed=SPLITTING_SEED)

                    dataset_name = config.get_name()
                    for shuffling_seed in SHUFFLING_SEEDS:
                        dataset = Dataset(dataset_name, os.path.join(BASE_DIR, data_dir), shuffling_seed=shuffling_seed)
                        model = model_cls(dataset, hparams, gridworld, seed=MODEL_SEED)
                        Trainer(model, logger)()

def run_FullGN(FLAGS):
    _run(FLAGS, FullGN)

def run_FullGN_NoAct(FLAGS):
    _run(FLAGS, FullGN_NoAct)

def run_FullGN_Mul(FLAGS):
    _run(FLAGS, FullGN_Mul)

def run_FullGN_MulMlp(FLAGS):
    _run(FLAGS, FullGN_MulMlp)

def run_GGNN(FLAGS):
    _run(FLAGS, GGNN)

def run_GGNN_NoAct(FLAGS):
    _run(FLAGS, GGNN_NoAct)

def run_GGNN_Mul(FLAGS):
    _run(FLAGS, GGNN_Mul)

def run_GGNN_MulMlp(FLAGS):
    _run(FLAGS, GGNN_MulMlp)

def run_GAT(FLAGS):
    _run(FLAGS, GAT)

def run_GAT_NoAct(FLAGS):
    _run(FLAGS, GAT_NoAct)

def run_GAT_Mul(FLAGS):
    _run(FLAGS, GAT_Mul)

def run_GAT_MulMlp(FLAGS):
    _run(FLAGS, GAT_MulMlp)

def run_RW_Stationary(FLAGS):
    _run(FLAGS, RW_Stationary)

def run_RW_Dynamic(FLAGS):
    _run(FLAGS, RW_Dynamic)

if __name__ == '__main__':
    tf.flags.DEFINE_string('cmd', 'run_GGNN', "choosee `generate_data` or `run_{model}`")
    tf.flags.DEFINE_integer('sz', 32, "the size of a grid map")
    tf.flags.DEFINE_integer('stp', 16, "the maximal steps of a trajectory")
    tf.flags.DEFINE_string('ckpt', None, "the checkpoint")
    tf.flags.DEFINE_integer('bs', 16, "the batch size")
    tf.flags.DEFINE_integer('dims', 40, "the number of dimensions")
    tf.flags.DEFINE_integer('info_dims', 32, "the number of information dimensions used in regular graph networks")
    tf.flags.DEFINE_integer('att_dims', 8, "the number of attention dimensions used in regular graph networks")
    tf.flags.DEFINE_integer('epochs', 50, "the maximal epochs for training")
    tf.flags.DEFINE_integer('heads', 5, "the number of heads used in graph attention networks")
    tf.flags.DEFINE_integer('selfatt_dims', 8, "the number of dimensions per head in graph attention networks")

    FLAGS = tf.flags.FLAGS

    """
        if FLAGS.cmd == 'generate_data':
            generate_data(FLAGS)
        elif FLAGS.cmd == 'run_FullGN':
            run_FullGN(FLAGS)
        elif FLAGS.cmd == 'run_FullGN_NoAct':
            run_FullGN_NoAct(FLAGS)
        elif FLAGS.cmd == 'run_FullGN_Mul':
            run_FullGN_Mul(FLAGS)
        elif FLAGS.cmd == 'run_FullGN_MulMlp':
            run_FullGN_MulMlp(FLAGS)
        elif FLAGS.cmd == 'run_GGNN':
            run_GGNN(FLAGS)
        elif FLAGS.cmd == 'run_GGNN_NoAct':
            run_GGNN_NoAct(FLAGS)
        elif FLAGS.cmd == 'run_GGNN_Mul':
            run_GGNN_Mul(FLAGS)
        elif FLAGS.cmd == 'run_GGNN_MulMlp':
            run_GGNN_MulMlp(FLAGS)
        elif FLAGS.cmd == 'run_GAT':
            run_GAT(FLAGS)
        elif FLAGS.cmd == 'run_GAT_NoAct':
            run_GAT_NoAct(FLAGS)
        elif FLAGS.cmd == 'run_GAT_Mul':
            run_GAT_Mul(FLAGS)
        elif FLAGS.cmd == 'run_GAT_MulMlp':
            run_GAT_MulMlp(FLAGS)
        elif FLAGS.cmd == 'run_RW_Stationary':
            run_RW_Stationary(FLAGS)
        elif FLAGS.cmd == 'run_RW_Dynamic':
            run_RW_Dynamic(FLAGS)
    """

    generate_data(FLAGS)
    run_FullGN(FLAGS)
    run_FullGN_NoAct(FLAGS)
    run_FullGN_Mul(FLAGS)
    run_FullGN_MulMlp(FLAGS)
    run_GGNN(FLAGS)
    run_GGNN_NoAct(FLAGS)
    run_GGNN_Mul(FLAGS)
    run_GGNN_MulMlp(FLAGS)
    run_GAT(FLAGS)
    run_GAT_NoAct(FLAGS)
    run_GAT_Mul(FLAGS)
    run_GAT_MulMlp(FLAGS)
    run_RW_Stationary(FLAGS)
    run_RW_Dynamic(FLAGS)
