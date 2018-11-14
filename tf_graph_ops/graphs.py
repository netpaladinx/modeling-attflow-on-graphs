from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

class Graph(object):
    """
    In sorted_edges = np.array(
        [...,
        (v1, etype, v2),
        ... ]
    ), `v1_ids` means sorted_edges[:, 0],
    `v2_ids` means sorted_edges[:, 2],
    `etype_ids` means sorted_edges[:, 1]
    """

    @property
    def v1_ids(self):
        raise NotImplementedError('`v1_ids` should be implemented')

    @property
    def v2_ids(self):
        raise NotImplementedError('`v2_ids` should be implemented')

    @property
    def etype_ids(self):
        raise NotImplementedError('`etype_ids` should be implemented')

    @property
    def ntype_ids(self):
        raise NotImplementedError('`ntype_ids` should be implemented')

    @property
    def n_edges(self):
        raise NotImplementedError('`n_edges` should be implemented')

    @property
    def n_etypes(self):
        raise NotImplementedError('`n_etypes` should be implemented')

    @property
    def n_nodes(self):
        raise NotImplementedError('`n_nodes` should be implemented')
