import networkx as nx
import numpy as np


def sample_neighbors(client_num, num_clients, self_ind):
    c_list = list(range(client_num))
    c_list.remove(self_ind)
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]


def sparse_graph(n, neighbor_num, create_using):
    if create_using.is_directed():
        m = np.zeros(shape=(n, n))
        for i in range(n):
            nb = sample_neighbors(n, neighbor_num, i)
            m[i][nb] = 1
        g = nx.from_numpy_matrix(np.asmatrix(m), create_using=nx.Graph())
    else:
        # undirected d-regular graph (sum row == sum column)
        g = nx.random_regular_graph(neighbor_num, n)
    return g


_graph_type_dict = {
    'complete': lambda **kwargs: nx.complete_graph(kwargs['n'], create_using=kwargs['create_using']),
    'ring': lambda **kwargs: nx.cycle_graph(kwargs['n'], create_using=kwargs['create_using']),
    'sparse': lambda **kwargs: sparse_graph(kwargs['n'], kwargs['neighbor_num'], create_using=kwargs['create_using']),
    'erdos_renyi': lambda **kwargs: nx.erdos_renyi_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'binomial': lambda **kwargs: nx.binomial_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'grid': lambda **kwargs: nx.grid_2d_graph(kwargs['neighbor_num'], kwargs['n'], kwargs['periodic'],
                                              kwargs['create_using'])
}


class GraphManager:

    def __init__(self, graph_type, nodes, directed=False, time_varying=-1, neighbor_num=None):
        self.n = len(nodes)
        self.nodes = nodes
        self.directed = directed
        self.time_varying = time_varying
        self.neighbor_num = neighbor_num
        self._graph_type = graph_type
        self._nx_graph = self._resolve_graph_type()
        self._resolve_weights_mixing()

    @property
    def nodes_num(self):
        return self.n

    def check_time_varying(self, time_iter):
        if self.time_varying > 0 and time_iter % self.time_varying:
            self._nx_graph = self._resolve_graph_type()
            self._resolve_weights_mixing()

    def _resolve_graph_type(self):
        assert self._graph_type in _graph_type_dict
        graph_fn = _graph_type_dict[self._graph_type]
        kwargs = {'n': self.n,
                  'create_using': nx.DiGraph() if self.directed else nx.Graph(),
                  'directed': self.directed,
                  'neighbor_num': self.neighbor_num
                  }
        """
        kwargs = {}
        for arg_name in graph_fn.__code__.co_varnames[:graph_fn.__code__.co_argcount]:
            kwargs[arg_name] = args[arg_name]
        """
        graph = graph_fn(**kwargs)
        return graph

    def _resolve_weights_mixing(self):
        for node in self._nx_graph.nodes:
            self._nx_graph.add_edge(node, node)
            nbs = self._graph_neighbors(node)
            weight = 1 / len(nbs)  # uniform weights
            for nb in nbs:
                self._nx_graph[node][nb]['weight'] = weight

    def get_edge_weight(self, i, j):
        return self._nx_graph.get_edge_data(i, j)['weight']

    def get_self_node_weight(self, node_id):
        return self.get_edge_weight(node_id, node_id)

    def _graph_neighbors(self, node_id):
        return list(self._nx_graph.neighbors(node_id))

    def get_peers(self, node_id):
        nb = self._graph_neighbors(node_id)
        return [n for n in self.nodes if n.id in nb and n.id != node_id]

    def get_weighted_peers(self, node_id):
        nb = self.get_peers(node_id)
        wb = [self.get_edge_weight(node_id, n.id) for n in nb]
        return nb, wb

    def get_node(self, node_id):
        return self.nodes[node_id]

    def draw(self):
        nx.draw(self._nx_graph)

    def print_info(self):
        print("Graph: {} ({}), n: {}, time-vary: {}, neighbors: {}".format(self._graph_type,
                                                                           'directed' if self.directed else 'undirected',
                                                                           self.n, self.time_varying,
                                                                           self.neighbor_num))


if __name__ == "__main__":
    gm = GraphManager('ring', list(range(10)), directed=True, neighbor_num=3)
    gm.draw()

"""

def test():
    g = nx.cycle_graph(10).to_directed()
    g[0][1]['weight'] = 0.1
    g[1][0]['weight'] = 0.2
    g[0][0]['weight'] = 0.05

    nx.to_numpy_matrix(nx.complete_graph(10, create_using=nx.Graph())) == \
    nx.to_numpy_matrix(nx.complete_graph(10, create_using=nx.DiGraph()))

    g = nx.complete_graph(10).to_directed()
    # nx.draw(nx.cycle_graph(10, create_using=nx.DiGraph()))

    for node in g.nodes:
        g.add_edge(node, node)
        nbs = list(g.neighbors(node))
        weight = 1 / len(nbs)
        for nb in nbs:
            g[node][nb]['weight'] = weight
    list(g.neighbors(0))

    nx.to_numpy_matrix(g)

    nx.to_numpy_matrix(nx.gnm_random_graph(10, 10, directed=True))
    nx.gnp_random_graph(10, 0.1)
    nx.gnm_random_graph()
    # binomial_graph = gnp_random_graph
    # erdos_renyi_graph = gnp_random_graph

    nx.to_numpy_matrix(nx.gnm_random_graph(10, 2, seed=1, directed=False))
    nx.to_numpy_matrix(nx.gnm_random_graph(10, 10, seed=1, directed=True))

    print(nx.to_numpy_matrix(nx.gnm_random_graph(10, 2, seed=1, directed=False)) == \
          nx.to_numpy_matrix(nx.gnm_random_graph(10, 2, seed=1, directed=True)))
    nx.draw(nx.gnm_random_graph(10, 40, seed=1, directed=True))
    nx.draw(nx.random_regular_graph(2, 10))
    nx.draw(nx.fast_gnp_random_graph(10, 0.2))

    m = np.zeros(shape=(10, 10))
    for i in range(10):
        n = sample_neighbors(10, 2, i)
        m[i][n] = 1

    nx.draw(nx.from_numpy_matrix(np.asmatrix(m), create_using=nx.Graph()))
    nx.draw(nx.gnp_random_graph(10, 0.2))
    # i = 2
    for i in range(9):
        num_nb = sum(m[i])
        print(i, num_nb)
        if num_nb < 2:
            n = sample_neighbors(10, 2 - int(num_nb), i)
            m[i, n] = 1
            m[n, i] = 1
        else:
            while num_nb > 2:
                for ni, nv in enumerate(m[i]):
                    if nv > 0:
                        if sum(m[ni]) > 2:
                            m[ni, i] = 0
                            m[i, ni] = 0
                            break
                new_num_nb = sum(m[i])
                if new_num_nb == num_nb:
                    break
                num_nb = new_num_nb

    g = nx.from_numpy_matrix(np.asmatrix(m), create_using=nx.Graph())
    nx.draw(g)
    sum([sum(m[i]) for i in range(10)]) / 10
    sum([len(list(g.neighbors(i))) for i in range(10)]) / 10

    g = nx.watts_strogatz_graph(10, 2, 0.2)
    nx.draw(g)
    nx.k_nearest_neighbors()
    nx.draw(nx.random_k_out_graph(10, 2, 1, False))



m = np.zeros(shape=(10, 10))
m[[1, 2], 0] = 1
m[[0, 3], 1] = 1
m[[0, 4], 2] = 1
m[[1, 5], 3] = 1
m[[2, 6], 4] = 1
m[[3, 7], 5] = 1
m[[4, 8], 6] = 1
m[[5, 9], 7] = 1
m[[6, 9], 8] = 1
m[[7, 8], 9] = 1
m
nx.draw(nx.from_numpy_matrix(np.asmatrix(m), create_using=nx.Graph()))
"""
