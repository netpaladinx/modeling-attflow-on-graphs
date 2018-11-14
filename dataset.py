from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import defaultdict

import numpy as np

from tf_graph_ops import Graph


# Input pipeline to load graph-structured data
class Dataset(Graph):
    def __init__(self, name, data_dir, shuffling_seed=None):
        self.dataset_name = name
        self.data_dir = data_dir
        self.shuffling_seed = shuffling_seed

        if not os.path.exists(self.data_dir):
            raise ValueError('`data_dir` does not exist.')

        if (not os.path.exists(os.path.join(self.data_dir, 'entity2id.txt'))) \
            or (not os.path.exists(os.path.join(self.data_dir, 'relation2id.txt'))):
            self._prepare_val2id_files()

        self.entity2id, self.id2entity = self._load_val2id_file('entity2id.txt')
        self.relation2id, self.id2relation = self._load_val2id_file('relation2id.txt')

        self.train_observation = self._load_observation_file('train.txt')
        self.valid_observation = self._load_observation_file('valid.txt')
        self.test_observation = self._load_observation_file('test.txt')

        pool = set(self.train_observation) | set(self.valid_observation) | set(self.test_observation)
        self.observation_pool = set((t[1], t[2]) for t in pool)

        self.graph = self._load_graph_file('graph.txt')

        self._add_selfloop()

    def _prepare_val2id_files(self):
        entities, relations = set(), set()
        with open(os.path.join(self.data_dir, 'graph.txt')) as f:
            for line in f.readlines():
                e1, rel, e2 = line.strip().split('\t')
                entities.add(e1)
                relations.add(rel)
                entities.add(e2)
        entities = sorted(entities)
        relations = sorted(relations)
        with open(os.path.join(self.data_dir, 'entity2id.txt'), 'w') as f:
            for id, entity in enumerate(entities):
                f.write('%s\t%d\n' % (entity, id))
        with open(os.path.join(self.data_dir, 'relation2id.txt'), 'w') as f:
            for id, relation in enumerate(relations):
                f.write('%s\t%d\n' % (relation, id))

    def _load_val2id_file(self, filename):
        val2id, id2val = dict(), dict()
        with open(os.path.join(self.data_dir, filename)) as f:
            for line in f.readlines():
                val, id = line.strip().split('\t')
                val2id[val] = int(id)
                id2val[int(id)] = val
        return val2id, id2val

    def _load_observation_file(self, filename):
        observation = []
        with open(os.path.join(self.data_dir, filename)) as f:
            for line in f.readlines():
                sp = line.strip().split('\t')
                traj_id = int(sp[0])
                observed_seq = map(lambda e: self.entity2id[e], sp[1:])
                observation.append(tuple([traj_id] + observed_seq))
        return observation

    def _load_graph_file(self, filename):
        graph = defaultdict(dict)
        with open(os.path.join(self.data_dir, filename)) as f:
            for line in f.readlines():
                sp = line.strip().split('\t')
                e1 = self.entity2id[sp[0]]
                rel = self.relation2id[sp[1]]
                e2 = self.entity2id[sp[2]]
                graph[e1][rel] = [e2]
        return graph

    @property
    def n_entities(self):
        return len(self.entity2id)

    @property
    def n_relations(self):
        return len(self.relation2id)

    @property
    def n_train(self):
        return len(self.train_observation)

    @property
    def n_valid(self):
        return len(self.valid_observation)

    @property
    def n_test(self):
        return len(self.test_observation)

    def _add_selfloop(self):
        selfloop_id = self.n_relations
        self.relation2id['selfloop'] = selfloop_id
        self.id2relation[selfloop_id] = 'selfloop'

        e1_rel_e2_list = []
        for e1, rel_e2 in self.graph.iteritems():
            for rel, e2 in rel_e2.iteritems():
                e1_rel_e2_list.append((e1, rel, e2))
            e1_rel_e2_list.append((e1, selfloop_id, e1))
        self.e1_rel_e2_list = np.array(sorted(e1_rel_e2_list))

    def get_train_batch(self, batch_size):
        if self.shuffling_seed:
            np.random.seed(self.shuffling_seed)
        rand_idx = np.random.permutation(self.n_train)
        start = 0
        while start < self.n_train:
            end = min(start + batch_size, self.n_train)
            batch = [self.train_observation[i] for i in rand_idx[start:end]]
            yield end - start, np.array(batch)
            start = end

    def get_eval_batch(self, batch_size, source='test'):
        start = 0
        num, observation = (self.n_test, self.test_observation) if source == 'test' else (self.n_valid, self.valid_observation)
        while start < num:
            end = min(start + batch_size, num)
            batch = observation[start:end]
            yield end - start, np.array(batch)
            start = end

    # implement methods in superclass Graph

    @property
    def n_nodes(self):
        return self.n_entities

    @property
    def n_edges(self):
        return self.e1_rel_e2_list.shape[0]

    @property
    def n_etypes(self):
        return self.n_relations

    @property
    def v1_ids(self):
        return self.e1_rel_e2_list[:, 0]

    @property
    def v2_ids(self):
        return self.e1_rel_e2_list[:, 2]

    @property
    def etype_ids(self):
        return self.e1_rel_e2_list[:, 1]
