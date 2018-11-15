from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import shutil
from itertools import product
from collections import defaultdict
import random

import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


PLT_OPTIONS = {
    'map_linewidth': 0.4, 'map_color': '#888888', 'map_zorder': 0,
    'traj_linewidth': 0.8, 'traj_color': '#0000FF', 'traj_alpha': 0.8, 'traj_zorder': 1,
    'src_c': 'r', 'src_s': 10, 'src_alpha': 0.5, 'src_zorder': 4,
    'dst_c': 'g', 'dst_s': 10, 'dst_alpha': 0.5, 'dst_zorder': 4,
    'ldir_fc': 'darkred', 'ldir_ec': 'darkred', 'ldir_linewidth': 0.6, 'ldir_alpha': 0.8, 'ldir_zorder': 1,
    'node_size': 6, 'node_alpha': 0.8, 'node_zorder': 3,
    'attended_color': [1., 0.6, 0.], 'unattended_color': [0.2, 0.2, 0.2],
    'title_size': 8, 'small_title_size': 7
}

class GridWorld(object):
    edge_types = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    edge_directions = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    edge_type2direction = dict(zip(edge_types, edge_directions))
    edge_direction2type = dict(zip(edge_directions, edge_types))

    def __init__(self, size=16, node_drop=0., edge_drop=0., max_steps=16, omega=1., alpha=1., lambda_1=0., lambda_2=0.,
                 varphi=0., a_1=1., b_1=0., a_2=1., b_2=0., sigma=0.1, depend_on_history=False, history_op='sum',
                 n_rollouts=10, observed_indices=(0, -1), draw_mode='grid', base_dir='data', name='gridworld'):
        """
        Latent direction function d_{t,x,y} = ( a_1 cos(theta_{t,x,y}) + b_1, a_2 sin(theta_{t,x,y}) + b_2 )
        where theta_{t,x,y} = omega * t^alpha + lambda_1 * x + lambda_2 * y + varphi.

        For the history-dependent direction,
        theta_{t,x,y} = omega * t^alpha + lambda_1 * history_op(x_0, ..., x_t) + lambda_2 * history_op(y_0, ..., y_t) + varphi.

        Args:
            size: grid size
            node_drop: the fraction of nodes to drop
            edge_drop: the fraction of edges to drop
            max_steps: the maximal length of a trajectory
            omega, alpha, lambda_1, lambda_2, varphi: parameters to compute theta_{t,x,y}
            a_1, b_1, a_2, b_2: parameters to compute latent directions
            sigma: the standard deviation to sample an edge direction based on current latent direction
            depend_on_history: whether to use history-dependent latent directions
            history_op: 'max' or 'sum', the aggregation operation to compute history-dependent latent directions
            n_rollouts: how many times to try to draw a trajectory given a source node
            observed_indices: by default, only the source node and the destination node are observed
            draw_mode: 'grid' or 'frame'
            base_dir: the root directory to store all gridworld data
            name: name of the specified gridworld data
        """
        self.size = size
        self.node_drop = node_drop
        self.edge_drop = edge_drop
        self.max_steps = max_steps
        self.omega = omega
        self.alpha = alpha
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.varphi = varphi
        self.a_1 = a_1
        self.b_1 = b_1
        self.a_2 = a_2
        self.b_2 = b_2
        self.sigma = sigma
        self.depend_on_history = depend_on_history
        self.history_op = history_op
        self.n_rollouts = n_rollouts
        self.observed_indices = observed_indices
        self.draw_mode = draw_mode
        self.base_dir = base_dir
        self.gridworld_name = name
        self.seed = None
        self.splitting_seed = None

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

    def __str__(self):
        return self.gridworld_name

    # generate data of a gridworld

    def generate(self, data_dir=None, seed=None, splitting_seed=None):
        """
        Generate the map and trajectories data for the given configuration,
        and save them into `data_dir` or a default location
        """
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self._generate_map()
        self._generate_trajectories()

        self.save(data_dir=data_dir, splitting_seed=splitting_seed)

    def _generate_map(self):
        nodes = [(x, y) for x, y in product(range(self.size), range(self.size))]  # range of x, y: [0, size)
        undirected_edges = [(node, (node[0] + e_dir[0], node[1] + e_dir[1]))
                            for node, e_dir in product(nodes, GridWorld.edge_directions[:4])
                            if (0 <= node[0] + e_dir[0] < self.size) and (0 <= node[1] + e_dir[1] < self.size)]
        n_nodes = int(len(nodes) * (1. - self.node_drop))
        n_undirected_edges = int(len(undirected_edges) * (1. - self.edge_drop))
        nodes = random.sample(nodes, n_nodes)
        undirected_edges = random.sample(undirected_edges, n_undirected_edges)

        nodes = set(nodes)
        directed_edges = []
        for node1, node2 in undirected_edges:
            if (node1 in nodes) and (node2 in nodes):
                e_dir = (node2[0] - node1[0], node2[1] - node1[1])
                directed_edges.append((node1, e_dir, node2))
                e_dir = (-e_dir[0], -e_dir[1])
                directed_edges.append((node2, e_dir, node1))
        nodes = set([node1 for node1, _, _ in directed_edges])

        self.nodes = sorted(nodes)
        self.edges = sorted(directed_edges)
        self.edges_dict = defaultdict(dict)
        for node1, e_dir, node2 in self.edges:
            self.edges_dict[node1][e_dir] = node2

    def _generate_trajectories(self):
        self.trajectories = []
        self.action_chains = []
        self.latent_direction_chains = []

        existing_pairs = set()
        for node in self.nodes:
            for _ in range(self.n_rollouts):
                trajectory, action_chain, latent_direction_chain = self._go(node)
                src, dst = trajectory[0], trajectory[-1]
                if (src, dst) in existing_pairs:
                    continue
                self.trajectories.append(trajectory)
                self.action_chains.append(action_chain)
                self.latent_direction_chains.append(latent_direction_chain)
                existing_pairs.add((src, dst))

    def _go(self, src):
        trajectory = [src]
        action_chain = []
        latent_direction_chain = []

        for t in range(self.max_steps):
            pos = trajectory[-1]
            next_pos, chosen_edge_type, latent_direction = self._next(pos, t, trajectory=trajectory)

            if chosen_edge_type == 'END':
                break

            trajectory.append(next_pos)
            action_chain.append(chosen_edge_type)
            latent_direction_chain.append(latent_direction)

        return trajectory, action_chain, latent_direction_chain

    def _next(self, pos, t, trajectory=None):
        latent_direction = self._get_latent_direction(pos, t, trajectory=trajectory)

        edge_directions = np.array(GridWorld.edge_directions)
        edge_directions = edge_directions / np.sqrt(np.sum(np.square(edge_directions), axis=1, keepdims=True))

        logits = np.matmul(edge_directions, latent_direction) * 2 / (self.sigma ** 2)
        logits = logits - np.max(logits)

        probs = np.exp(logits)
        probs = probs / np.sum(probs)

        candidate = np.random.choice(len(probs), p=probs)
        chosen_edge_direction = GridWorld.edge_directions[candidate]
        if chosen_edge_direction in self.edges_dict[pos]:
            next_pos = (pos[0] + chosen_edge_direction[0], pos[1] + chosen_edge_direction[1])
            chosen_edge_type = self.edge_types[candidate]
            return next_pos, chosen_edge_type, latent_direction
        else:
            return pos, 'END', latent_direction

    def _get_latent_direction(self, pos, t, trajectory=None):
        if self.depend_on_history:
            if self.history_op == 'max':
                x, y = reduce(lambda t1, t2: (max(t1[0], t2[0]), max(t1[1], t2[1])), trajectory)
            elif self.history_op == 'sum':
                x, y = reduce(lambda t1, t2: (t1[0] + t2[0], t1[1] + t2[1]), trajectory)
        else:
            x, y = pos
        theta = self.omega * np.power(t, self.alpha) + self.lambda_1 * x + self.lambda_2 * y + self.varphi
        direction_x = self.a_1 * np.cos(theta) + self.b_1
        direction_y = self.a_2 * np.sin(theta) + self.b_2
        latent_direction = np.array((direction_x, direction_y))
        latent_direction = latent_direction / np.maximum(np.sqrt(np.sum(np.square(latent_direction))), 1e-20)
        return latent_direction

    def save(self, data_dir=None, splitting_seed=None):
        """
        Save the gridworld data into `data_dir` or a default location
        """
        self.splitting_seed = splitting_seed

        if data_dir is None:
            data_dir = os.path.join(self.base_dir, self.gridworld_name)
        else:
            data_dir = os.path.join(self.base_dir, data_dir)
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.mkdir(data_dir)

        with open(os.path.join(data_dir, 'graph.txt'), 'w') as f:
            for node1 in self.nodes:
                for e_dir, node2 in self.edges_dict[node1].iteritems():
                    f.write('%s\t%s\t%s\n' %
                            ('%d_%d' % node1, GridWorld.edge_direction2type[e_dir], '%d_%d' % node2))

        if self.splitting_seed is not None:
            random.seed(self.splitting_seed)

        # split train:valid:test by partition of nodes
        n_nodes = len(self.nodes)
        n_train = int(n_nodes * 0.8)
        n_valid = int((n_nodes - n_train) * 0.5)

        indices = range(n_nodes)
        random.shuffle(indices)
        train_nodes = set(self.nodes[i] for i in indices[:n_train])
        valid_nodes = set(self.nodes[i] for i in indices[n_train : (n_train + n_valid)])
        test_nodes = set(self.nodes[i] for i in indices[(n_train + n_valid):])

        with open(os.path.join(data_dir, 'train.txt'), 'w') as f_train, \
            open(os.path.join(data_dir, 'valid.txt'), 'w') as f_valid, \
            open(os.path.join(data_dir, 'test.txt'), 'w') as f_test, \
            open(os.path.join(data_dir, 'trajectories.txt'), 'w') as f_traj, \
            open(os.path.join(data_dir, 'action_chains.txt'), 'w') as f_ac, \
            open(os.path.join(data_dir, 'latent_direction_chains.txt'), 'w') as f_ldc:

            for i, trajectory in enumerate(self.trajectories):
                observation = [trajectory[ind] for ind in self.observed_indices]
                observation_str = '\t'.join(map(lambda pos: '%d_%d' % pos, observation))

                src = trajectory[0]
                if src in train_nodes:
                    f_train.write('%d\t%s\n' % (i, observation_str))
                elif src in valid_nodes:
                    f_valid.write('%d\t%s\n' % (i, observation_str))
                elif src in test_nodes:
                    f_test.write('%d\t%s\n' % (i, observation_str))

                trajectory_str = '\t'.join(map(lambda pos: '%d_%d' % pos, trajectory))
                f_traj.write('%d\t%s\n' % (i, trajectory_str))

                action_chain_str = '\t'.join(self.action_chains[i])
                f_ac.write('%d\t%s\n' % (i, action_chain_str))

                latent_direction_chain_str = '\t'.join(map(lambda d: '%.4f_%.4f' % (d[0], d[1]), self.latent_direction_chains[i]))
                f_ldc.write('%d\t%s\n' % (i, latent_direction_chain_str))

    # load data of a gridworld

    def load(self, data_dir=None, seed=None, splitting_seed=None):
        if data_dir is None:
            data_dir = os.path.join(self.base_dir, self.gridworld_name)
        else:
            data_dir = os.path.join(self.base_dir, data_dir)
        if not os.path.exists(data_dir):
            raise ValueError("`data_dir` %s does not exist." % data_dir)

        self.seed = seed
        self.splitting_seed = splitting_seed

        self._load_map(data_dir)
        self._load_trajectories(data_dir)

    def _load_map(self, data_dir):
        self.edges_dict = defaultdict(dict)
        edges = []
        with open(os.path.join(data_dir, 'graph.txt')) as f:
            for line in f.readlines():
                node1, e_dir, node2 = line.strip().split('\t')
                node1 = tuple(map(lambda x: int(x), node1.split('_')))
                e_dir = GridWorld.edge_type2direction[e_dir]
                node2 = tuple(map(lambda x: int(x), node2.split('_')))

                self.edges_dict[node1][e_dir] = node2
                edges.append((node1, e_dir, node2))
        self.nodes = sorted(self.edges_dict.keys())
        self.edges = sorted(edges)

    def _load_trajectories(self, data_dir):
        self.trajectories = []
        self.action_chains = []
        self.latent_direction_chains = []

        with open(os.path.join(data_dir, 'trajectories.txt')) as f:
            for line in f.readlines():
                traj = line.strip().split('\t')[1:]
                traj = map(lambda pos: (int(pos.split('_')[0]), int(pos.split('_')[1])), traj)
                self.trajectories.append(traj)

        with open(os.path.join(data_dir, 'action_chains.txt')) as f:
            for line in f.readlines():
                action_chain = line.strip().split('\t')[1:]
                self.action_chains.append(action_chain)

        with open(os.path.join(data_dir, 'latent_direction_chains.txt')) as f:
            for line in f.readlines():
                latent_d_chain = line.strip().split('\t')[1:]
                latent_d_chain = map(lambda d: (float(d.split('_')[0]), float(d.split('_')[1])), latent_d_chain)
                self.latent_direction_chains.append(latent_d_chain)

    # for visualization

    def visualize_examples(self, n_examples=-1, data_dir=None, seed=None):
        """ Visualize generated data examples
        """
        if data_dir is None:
            data_dir = os.path.join(self.base_dir, self.gridworld_name)

        visual_dir = os.path.join(data_dir, 'visualized_data')
        if os.path.exists(visual_dir):
            shutil.rmtree(visual_dir)
        os.mkdir(visual_dir)

        if n_examples == -1:
            example_indices = range(len(self.trajectories))
        else:
            if seed is not None:
                random.seed(seed)
            example_indices = random.sample(range(len(self.trajectories)), n_examples)

        for i in example_indices:
            trajectory = self.trajectories[i]
            latent_direction_chain = self.latent_direction_chains[i]
            self._draw_data_example(trajectory, latent_direction_chain, visual_dir)

    def visualize_attflow(self, examples, node_attentions, visualized_steps=None, data_dir=None):
        """ Visualize inferred attention flow examples
        """
        if data_dir is None:
            data_dir = os.path.join(self.base_dir, self.gridworld_name)

        visual_dir = os.path.join(data_dir, 'visualized_attflow')
        if os.path.exists(visual_dir):
            shutil.rmtree(visual_dir)
        os.mkdir(visual_dir)

        node_scores_per_step = np.array(node_attentions)  # n_steps x n_examples x n_nodes
        node_scores_max_per_step = np.max(node_scores_per_step, axis=2, keepdims=True)
        node_scores_per_step = node_scores_per_step / node_scores_max_per_step
        node_scores_across_steps = np.max(node_scores_per_step, axis=0)  # n_examples x n_nodes

        for i, example in enumerate(examples):
            self._draw_attention_belt_example(example, node_scores_across_steps[i], visual_dir)
            self._draw_attention_flow_example(example, node_scores_per_step[:, i, :], visualized_steps, visual_dir)

    def _draw_data_example(self, trajectory, latent_direction_chain, visual_dir):
        plt.subplot(121)
        self._draw_map()
        self._draw_trajectory(trajectory)
        self._set_plt()

        plt.subplot(122)
        self._draw_map()
        self._draw_latent_direction_chain(trajectory, latent_direction_chain)
        self._set_plt()

        src, dst = trajectory[0], trajectory[-1]
        filename = 'src_%d_%d_dst_%d_%d.svg' % (src[0], src[1], dst[0], dst[1])
        self._save_fig(os.path.join(visual_dir, filename))

    def _draw_attention_belt_example(self, example, node_scores_across_steps, visual_dir):
        traj_id = example[0]
        trajectory = self.trajectories[traj_id]
        latent_direction_chain = self.latent_direction_chains[traj_id]
        src, dst = trajectory[0], trajectory[-1]

        plt.subplot(131)
        self._draw_map()
        self._draw_trajectory(trajectory)
        self._set_plt(title='Trajectory: (%d, %d) to (%d, %d)' % (src[0], src[1], dst[0], dst[1]))

        plt.subplot(132)
        self._draw_map()
        self._draw_latent_direction_chain(trajectory, latent_direction_chain)
        self._set_plt(title='Latent directions')

        plt.subplot(133)
        self._draw_map()
        self._draw_node_attentions(node_scores_across_steps)
        self._set_plt(title='Attention flow')

        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.1)
        filename = 'belt_src_%d_%d_dst_%d_%d.svg' % (src[0], src[1], dst[0], dst[1])
        self._save_fig(os.path.join(visual_dir, filename))

    def _draw_attention_flow_example(self, example, node_scores_per_step, visualized_steps, visual_dir):
        traj_id = example[0]
        trajectory = self.trajectories[traj_id]
        src, dst = trajectory[0], trajectory[-1]

        if visualized_steps is None:
            visualized_steps = range(1, len(trajectory), 3)
        nc = 3
        nr = int(np.ceil(len(visualized_steps) * 1. / nc))
        for i, t in enumerate(visualized_steps):
            plt.subplot(nr, nc, i+1)
            self._draw_map()
            self._draw_trajectory(trajectory)
            self._draw_node_attentions(node_scores_per_step[t])
            self._set_plt(title='Step %d' % t, title_size=PLT_OPTIONS['small_title_size'])

        plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=-0.5)
        plt.suptitle('Src: (%d, %d), Dst: (%d, %d)' % (src[0], src[1], dst[0], dst[1]), size=PLT_OPTIONS['title_size'])
        filename = 'flow_src_%d_%d_dst_%d_%d.svg' % (src[0], src[1], dst[0], dst[1])
        self._save_fig(os.path.join(visual_dir, filename))

    def _draw_map(self):
        if self.draw_mode == 'grid':
            for node1 in self.nodes:
                for e_dir, node2 in self.edges_dict[node1].iteritems():
                    if node1 < node2:
                        plt.plot((node1[0], node2[0]), (node1[1], node2[1]), '_',
                                 linewidth=PLT_OPTIONS['map_linewidth'],
                                 color=PLT_OPTIONS['map_color'],
                                 zorder=PLT_OPTIONS['map_zorder'])
        elif self.draw_mode == 'frame':
            N = self.size
            plt.plot((0, N, N, 0), (0, 0, N, N), '_',
                     linewidth=PLT_OPTIONS['map_linewidth'],
                     color=PLT_OPTIONS['map_color'],
                     zorder=PLT_OPTIONS['map_zorder'])

    def _draw_trajectory(self, trajectory):
        for t in range(len(trajectory) - 1):
            node1 = trajectory[t]
            node2 = trajectory[t + 1]
            plt.plot((node1[0], node2[0]), (node1[1], node2[1]), '-',
                     linewidth=PLT_OPTIONS['traj_linewidth'],
                     color=PLT_OPTIONS['traj_color'],
                     alpha=PLT_OPTIONS['traj_alpha'],
                     zorder=PLT_OPTIONS['traj_zorder'])
        src, dst = trajectory[0], trajectory[-1]
        plt.scatter(src[0], src[1], c=PLT_OPTIONS['src_c'], s=PLT_OPTIONS['src_s'], marker='o',
                    alpha=PLT_OPTIONS['src_alpha'], zorder=PLT_OPTIONS['src_zorder'])
        plt.scatter(dst[0], dst[1], c=PLT_OPTIONS['dst_c'], s=PLT_OPTIONS['dst_s'], marker='o',
                    alpha=PLT_OPTIONS['dst_alpha'], zorder=PLT_OPTIONS['dst_zorder'])

    def _draw_latent_direction_chain(self, trajectory, latent_direction_chain):
        for t in range(len(trajectory) - 1):
            node1 = trajectory[t]
            latent_d = latent_direction_chain[t]
            plt.arrow(node1[0], node1[1], latent_d[0]*0.8, latent_d[1]*0.8, head_width=0.3, head_length=0.4,
                      fc=PLT_OPTIONS['ldir_fc'], ec=PLT_OPTIONS['ldir_ec'],
                      linewidth=PLT_OPTIONS['ldir_linewidth'],
                      alpha=PLT_OPTIONS['ldir_alpha'],
                      zorder=PLT_OPTIONS['ldir_zorder'])

    def _draw_node_attentions(self, node_scores):
        xs, ys = zip(*self.nodes)
        colors = np.expand_dims(node_scores, 1) * np.array(PLT_OPTIONS['attended_color']) + \
                 (1. - np.expand_dims(node_scores, 1)) * np.array(PLT_OPTIONS['unattended_color'])
        plt.scatter(xs, ys, c=colors, s=PLT_OPTIONS['node_size'],
                    alpha=PLT_OPTIONS['node_alpha'],
                    zorder=PLT_OPTIONS['node_zorder'])

    def _set_plt(self, title=None, title_size=None):
        plt.xlim(-1, self.size)
        plt.ylim(-1, self.size)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
        if title is not None:
            plt.gca().set_title(title, size=PLT_OPTIONS['title_size'] if title_size is None else title_size)

    def _save_fig(self, path, format='svg'):
        plt.savefig(path, format=format)
        plt.close()
