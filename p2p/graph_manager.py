import networkx as nx
import numpy as np


class DummyNode:
    def __init__(self, node_id):
        self.id = node_id


def sample_neighbors(client_num, num_clients, self_ind):
    c_list = list(range(client_num))
    c_list.remove(self_ind)
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]


def create_ring(n, create_using):
    return nx.cycle_graph(n, create_using=create_using)


def sparse_graph(n, num_neighbors, create_using):
    if create_using.is_directed():
        in_n = [num_neighbors] * n
        out_n = [num_neighbors] * n
        g = nx.directed_havel_hakimi_graph(in_n, out_n, create_using)
        """
        m = np.zeros(shape=(n, n))
        for i in range(n):
            nb = sample_neighbors(n, num_neighbors, i)
            m[i][nb] = 1
        g = nx.from_numpy_matrix(np.asmatrix(m), create_using=create_using)
        """
    else:
        if num_neighbors == 2 and not create_using.is_directed():
            # create ring graph to prevent separations into groups
            return create_ring(n, create_using)
        # undirected d-regular graph (sum row == sum column)
        g = nx.random_regular_graph(num_neighbors, n)
    return g


def create_torus(**kwargs):
    assert kwargs['n'] % kwargs['num_neighbors'] == 0
    # kwargs = {'num_neighbors': 3, 'n': 15, 'create_using': nx.DiGraph()}
    if not kwargs['create_using'].is_directed():
        g = nx.grid_graph([kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors'])], [True, True])
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        return g
    else:
        """
        g = nx.grid_graph([kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors'])], [True, True])
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        mx = nx.to_numpy_matrix(g)
        for i in range(len(mx)):
            for j in range(i+1, len(mx)):
                mx[i, j] = 0

        nx.draw(g, pos=nx.spring_layout(g), with_labels=True)
        return g
        """
        raise NotImplementedError()


def create_grid(**kwargs):
    assert kwargs['n'] % kwargs['num_neighbors'] == 0
    if not kwargs['create_using'].is_directed():
        g = nx.grid_2d_graph(kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors']), [False, True], kwargs['create_using'])
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        return g
    else:
        """
        graph = nx.grid_graph([kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors'])], periodic=[False, True])
        edges = graph.edges()
        g = nx.from_edgelist(edges, nx.DiGraph())
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        return g
        """
        raise NotImplementedError()


def create_sparse_clusters(n, num_neighbors, create_using, clusters=2, cluster_conns=1, **kwargs):
    # if 'cluster_directed' in kwargs:
    #     create_using = nx.DiGraph() if kwargs['cluster_directed'] else nx.Graph()
    # else:
    #     create_using = kwargs['create_using']

    if isinstance(clusters, list) or isinstance(clusters, tuple):
        clusters_num = len(clusters)
        cluster_inds = []
        count = 0
        for c in clusters:
            cluster_inds.append(list(range(count, count + c)))
            count += c
    else:
        clusters_num = clusters
        nc = int(n / clusters_num)
        cluster_inds = [list(range(i*nc, nc*(i+1))) for i in range(clusters_num)]

    assert cluster_conns <= min([len(x) for x in cluster_inds])

    if 'cluster_directed' in kwargs:
        cluster_directed = kwargs['cluster_directed']
    else:
        cluster_directed = create_using.is_directed()

    if int(cluster_conns) != cluster_conns and not cluster_directed:
        raise ValueError("Undirected clustered connections not yet working properly with fractions")

    adj_mx = np.zeros((n, n))
    # cluster_inds = []
    for i, nc in enumerate(cluster_inds):
        nc_l = len(nc)
        g = sparse_graph(nc_l, num_neighbors, create_using)
        m = nx.to_numpy_matrix(g)
        adj_mx[i*nc_l:nc_l*(i+1), i*nc_l:nc_l*(i+1)] = m
        # cluster_inds.append(list(range(i*nc, nc*(i+1))))

    if cluster_conns > 0:
        for i in range(len(cluster_inds)):
            for j in range(0 if cluster_directed else i+1, len(cluster_inds)):
                if i == j:
                    continue
                ci, cj = cluster_inds[i], cluster_inds[j]
                conns = int(cluster_conns * len(ci))
                ci_sorted = ci.copy()
                ci_sorted.sort(key=lambda x: sum(adj_mx[x] > 0))
                rnd_ci = np.concatenate([np.tile(ci, int(cluster_conns)),
                                         ci_sorted[:int(len(ci) * abs(cluster_conns - int(cluster_conns)))]
                                         # ci[:int(len(ci) * (cluster_conns - int(cluster_conns)))
                                         # np.random.choice(ci, size=int(len(ci) * (cluster_conns - int(cluster_conns))), replace=False)
                                         ]).astype(np.int)
                # [sum(adj_mx[:, x]) for x in cj]
                in_peers = [1/(sum(adj_mx[:, x])**100) for x in cj]
                in_peers = [x/sum(in_peers) for x in in_peers]
                if all([round(min(in_peers), 2) == round(x, 2) for x in in_peers]):
                    in_peers = None
                # in_peers
                while True:
                    rnd_cj = np.random.choice(cj, size=conns, replace=conns > len(cj), p=in_peers)
                    if len(set(zip(rnd_ci, rnd_cj))) == conns:
                        break
                adj_mx[rnd_ci, rnd_cj] = 1
                if not cluster_directed:
                    adj_mx[rnd_cj, rnd_ci] = 1

    # for i in range(n):
    #     print("{}-{}".format(sum(adj_mx[i, :] > 0), sum(adj_mx[:, i] > 0)))

    if cluster_directed and not create_using.is_directed():
        create_using = nx.DiGraph()
    g = nx.from_numpy_matrix(np.asmatrix(adj_mx), create_using=create_using)
    # nx.draw(g, node_color=[["blue", "green", "red", "yellow"][i] for i, c in enumerate(cluster_inds) for _ in c], with_labels=True)

    return g


def create_acc_conns(n, create_using, **kwargs):
    return nx.empty_graph(n, create_using=create_using)


def start_acc_conns(gm):
    nodes = gm.nodes
    num_neighbors = gm.num_neighbors
    adj_mx = np.zeros((len(nodes), len(nodes)))
    prob = np.zeros(adj_mx.shape)

    for i, ni in enumerate(nodes):
        neigh_i = []
        for j, nj in enumerate(nodes):
            if i == j:
                neigh_i.append(0)
                continue
            acc_i = list(ni.eval_model(ni.model, nj.train).values())[0]
            neigh_i.append(acc_i)
            # acc_j = list(ni.eval_model(nj.model, nj.train).values())[0]
            # print(i, j, acc_i >= acc_j, acc_i, acc_j)
            # if acc_i >= acc_j:
            #     adj_mx[i, j] = 1
        neigh_indices = np.argsort(neigh_i)[-num_neighbors:]
        adj_mx[i, neigh_indices] = np.array(neigh_i)[neigh_indices]

        prob[i] = np.array(neigh_i)
    # print(adj_mx)
    from_prob = gm.kwargs.get('from_prob', True)
    if from_prob:
        print("Prob: Send-Receive")
        for i in range(len(prob)):
            print(i, "{}-{}".format(sum(prob[i, :] > 0), sum(prob[:, i] > 0)))
        gm._nx_graph = build_from_probabilities(prob, num_neighbors, create_using=nx.DiGraph() if gm._nx_graph.is_directed() else nx.Graph())
        return

    print("Send-Receive")
    for i in range(len(adj_mx)):
        print(i, "{}-{}".format(sum(adj_mx[i, :] > 0), sum(adj_mx[:, i] > 0)))
    """
    mx = [[0, 0, 0, 0, 0.05882353, 0.04365904, 0, 0.03630515, 0, 0],
             [0.0060698,  0.00478469, 0, 0, 0, 0, 0, 0, 0.01333333, 0],
             [0.02731411, 0.07272727, 0.03941909, 0, 0, 0, 0, 0, 0, 0],
             [0, 0.05454545, 0.03827751, 0, 0, 0.03118503, 0, 0, 0, 0],
             [0.01669196, 0.03636364, 0, 0.01452282, 0, 0, 0, 0, 0, 0],
             [0.01820941, 0, 0, 0, 0, 0.01039501, 0, 0.01286765, 0, 0],
             [0, 0, 0.02392345, 0.02074689, 0, 0, 0.01863354, 0, 0, 0],
             [0, 0, 0, 0, 0, 0.05882353, 0.03118503, 0.03170956, 0, 0],
             [0, 0.03636364, 0, 0, 0.03271028, 0, 0.03118503, 0, 0, 0],
             [0.0030349,  0, 0, 0, 0.0046729,  0, 0, 0, 0.00045956, 0]]
    adj_mx = np.zeros((len(mx), len(mx)))
    for i, m in enumerate(mx):
        adj_mx[i, np.array(m) > 0] = np.array(m)[np.array(m) > 0]
    # """

    for i in range(len(adj_mx)):
        if sum(adj_mx[:, i] > 0) > num_neighbors:
            indices = np.where(adj_mx[:, i] > 0)[0][np.argsort(adj_mx[np.where(adj_mx[:, i] > 0)[0], i])]
            adj_mx[indices[:-num_neighbors], i] = 0
            other_ind = list(indices[:-num_neighbors])
            print(i, other_ind)
            while len(other_ind) > 0:
                neigh_num_summary = np.sum(adj_mx > 0, axis=0)
                min_ind = np.argmin(neigh_num_summary)
                num_n = int(neigh_num_summary[min_ind])
                n_diff = num_neighbors - num_n
                print('- Selected', min_ind, num_n)
                if n_diff > 0:
                    adj_mx[other_ind[:n_diff], min_ind] = 1
                    other_ind = other_ind[n_diff:]

    print("Final Send-Receive")
    for i in range(len(adj_mx)):
        print(i, "{}-{}".format(sum(adj_mx[i, :] > 0), sum(adj_mx[:, i] > 0)))
    g = nx.from_numpy_matrix(np.asmatrix(adj_mx), create_using=nx.DiGraph() if gm._nx_graph.is_directed() else nx.Graph())
    gm._nx_graph = g
    # nx.draw(g, with_labels=True)


def prepare_for_clustering(nodes, use_data='data points'):
    from data.metrics import convert_to_global_vector
    if use_data == 'data points':
        train = []
        for a in nodes:
            a_t = []
            for dsi in list(a.train):
                a_t.extend(dsi[1])
            train.append(np.array(a_t))

        v_space = nodes[0].model.layers[-1].units
        if v_space == 10_002:
            train_ds = [t[t > 1] for t in train]  # 1-> OOV token, 0-> padding
            labels = np.asarray(convert_to_global_vector([a - 2 for a in train_ds], 10_000))
        else:
            labels = np.asarray(convert_to_global_vector(train, v_space))
    else:
        v_space = []
        for a in nodes:
            v_space.extend(a._data['metadata-subreddits'])
        v_space = np.unique(v_space)
        labels = np.asarray(convert_to_global_vector([list(map(lambda x: list(v_space).index(x), a._data['metadata-subreddits'])) for a in nodes], len(v_space)))

    for i in range(len(labels)):
        labels[i] /= labels[i].sum()
    return labels


def build_from_classes(pred, num_neighbors, create_using):

    prob = np.zeros((len(pred), len(pred)))
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            if i == j:
                continue
            diff = abs(pred[i]-pred[j]) + 1
            prob[i, j] = 1 / diff**2
    return build_from_probabilities(prob, num_neighbors, create_using)


def build_from_probabilities(prob, num_neighbors, create_using):
    searches = 0
    while True:
        adj_mx = np.zeros(prob.shape)
        for i in range(adj_mx.shape[0]):
            p = np.copy(prob[i])

            # remove all nodes that already have enough receiving neighbors
            for j in range(p.shape[0]):
                if p[j] > 0 and sum(adj_mx[:, j] > 0) >= num_neighbors:
                    p[j] = 0

            if sum(p > 0) < num_neighbors:
                for j in range(p.shape[0]):
                    if p[j] == 0 and sum(adj_mx[:, j] > 0) < num_neighbors:
                        p[j] = 1e-10

            p /= p.sum()  # Normalize
            choices = np.random.choice(np.arange(0, prob[i].shape[0]), size=min(num_neighbors, sum(p > 0)), p=p, replace=False)
            adj_mx[i, choices] = prob[i, choices]
            # print(i, [prob[i, ki] for ki, k in enumerate(adj_mx[i]) if k])
            searches += 1

        np.fill_diagonal(adj_mx, 0)
        if all([sum(adj_mx[a, :] > 0) == num_neighbors for a in range(adj_mx.shape[0])]) and all([sum(adj_mx[:, a] > 0) == num_neighbors for a in range(adj_mx.shape[0])])\
                or searches > 10_000:
            print(searches, "searches")
            if searches > 10_000:
                print("Suboptimal solution found")
                for i in range(len(adj_mx)):
                    print(i, "{}-{}".format(sum(adj_mx[i, :] > 0), sum(adj_mx[:, i] > 0)))
            break

    # for i in range(len(adj_mx)):
    #     print(i, "{}-{}".format(sum(adj_mx[i, :] > 0), sum(adj_mx[:, i] > 0)))
    graph = nx.from_numpy_matrix(adj_mx, create_using=create_using)
    # nx.draw(graph, node_color=[["blue", "green", "red", "yellow", "black"][c] for c in pred], with_labels=True)
    return graph


def build_from_clusters(clusters, num_neighbors, create_using):
    num_nodes = sum([len(c) for c in clusters])
    adj_mx = np.zeros((num_nodes, num_nodes))
    for cl in clusters:
        g = sparse_graph(n=len(cl), num_neighbors=num_neighbors, create_using=create_using)
        g_mx = nx.to_numpy_array(g)
        for i in range(len(cl)):
            adj_mx[cl[i]][np.array(cl)[g_mx[i].astype(np.int) > 0]] = 1

    graph = nx.from_numpy_matrix(adj_mx, create_using=create_using)
    """
    pred = np.zeros(num_nodes)
    for ci, c in enumerate(clusters):
        for cl in c:
            pred[cl] = ci
    pred = pred.astype(np.int)
    nx.draw(graph, node_color=[["blue", "green", "red", "yellow", "black"][c] for c in pred], with_labels=True)
    # """
    return graph


def k_means_clusters(nodes, create_using, num_neighbors, n_clusters=2, use_data='data points', form_clusters=False, **kwargs):
    from sklearn.cluster import KMeans
    labels = prepare_for_clustering(nodes, use_data)
    pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(labels)
    if not form_clusters:
        return build_from_classes(pred, num_neighbors, create_using)
    else:
        clusters = [[] for _ in range(len(np.unique(pred)))]
        for i, p in enumerate(pred):
            clusters[p].append(i)
        return build_from_clusters(clusters, num_neighbors, create_using)


def aucccr_clusters(nodes, create_using, num_neighbors, use_data='data points', form_clusters=False, **kwargs):
    labels = prepare_for_clustering(nodes, use_data)
    from data.metrics.aucccr import recommend_clusters, scf, thd
    clusters = recommend_clusters(labels, v=lambda x: np.sqrt(scf*thd) if x > thd else np.sqrt(scf*x))
    print("AUCCCR produced", len(clusters), "clusters with cluster lenghts:", [len(c) for c in clusters])
    if not form_clusters:
        pred = np.zeros(len(labels))
        for ci, c in enumerate(clusters):
            for cl in c:
                pred[cl] = ci
        return build_from_classes(pred, num_neighbors, create_using)
    else:
        return build_from_clusters(clusters, num_neighbors, create_using)


def jensen_shannon(num_neighbors, create_using, filepath='reddit_50_stackoverflow_50_clients', **kwargs):
    # from plot.visualize import read_json
    from plot.data import read_symmetric_matrix
    data = read_symmetric_matrix(filepath)
    # data = np.array(data)[0:10, 0:10]

    adj_mx = np.zeros(data.shape)
    for i in range(data.shape[0]):
        # Node i sending to peers in p
        p = np.copy(data[i])
        p_s = np.argsort(p)

        # remove all nodes that already have enough receiving neighbors
        for j in range(p_s.shape[0]):
            if sum(adj_mx[:, p_s[j]] > 0) >= num_neighbors or i == p_s[j]:
                p_s[j] = -1

        p_s = p_s[p_s != -1]
        adj_mx[i, p_s[:num_neighbors]] = p[p_s[:num_neighbors]] + 0.01

    # for am in range(len(adj_mx)):
    #     print(am, "{}-{}".format(sum(adj_mx[am, :] > 0), sum(adj_mx[:, am] > 0)))
    graph = nx.from_numpy_matrix(adj_mx, create_using=create_using)
    return graph


_graph_type_dict = {
    'complete': lambda **kwargs: nx.complete_graph(kwargs['n'], create_using=kwargs['create_using']),
    'ring': lambda **kwargs: create_ring(kwargs['n'], create_using=kwargs['create_using']),
    'sparse': lambda **kwargs: sparse_graph(kwargs['n'], kwargs['num_neighbors'], create_using=kwargs['create_using']),
    'erdos_renyi': lambda **kwargs: nx.erdos_renyi_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'binomial': lambda **kwargs: nx.binomial_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'torus': create_torus,
    'grid': create_grid,
    'sparse_clusters': lambda **kwargs: create_sparse_clusters(**kwargs),
    'acc_conns': lambda **kwargs: create_acc_conns(**kwargs),
    'kmeans': lambda **kwargs: k_means_clusters(**kwargs),
    'aucccr': lambda **kwargs: aucccr_clusters(**kwargs),
    'jensen_shannon': lambda **kwargs: jensen_shannon(**kwargs),
}


class GraphManager:

    def __init__(self, graph_type, nodes, directed=False, time_varying=-1, num_neighbors=1, **kwargs):
        self.n = len(nodes)
        self.nodes = nodes
        self.directed = directed
        self.time_varying = time_varying
        self.num_neighbors = num_neighbors
        self.graph_type = graph_type
        self.kwargs = kwargs
        self._nx_graph = self._resolve_graph_type(**kwargs)
        self._resolve_weights_mixing()

    @property
    def nodes_num(self):
        return self.n

    def check_time_varying(self, time_iter):
        if self.time_varying > 0 and time_iter % self.time_varying == 0:
            print('Changing graph communication matrix')
            self._nx_graph = self._resolve_graph_type()
            self._resolve_weights_mixing()

    def _resolve_graph_type(self,  **kwargs):
        assert self.graph_type in _graph_type_dict
        graph_fn = _graph_type_dict[self.graph_type]
        params = {'n': self.n,
                  'create_using': nx.DiGraph() if self.directed else nx.Graph(),
                  'directed': self.directed,
                  'num_neighbors': self.num_neighbors,
                  'p': self.num_neighbors / self.n,
                  'nodes': self.nodes
                  }
        kwargs.update(params)
        graph = graph_fn(**kwargs)
        return graph

    def _resolve_weights_mixing(self):
        for node in self._nx_graph.nodes:
            # For now, all algorithms require Wii > 0, so we add self as an edge
            self._nx_graph.add_edge(node, node)
            nbs = self._graph_neighbors(node)
            # For now, uniform weights
            weight = 1 / len(nbs)
            for nb in nbs:
                self._nx_graph[node][nb]['weight'] = weight

    def get_edge_weight(self, i, j):
        edge_data = self._nx_graph.get_edge_data(i, j)
        if edge_data is None:
            return 0
        return edge_data['weight']

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

    def start(self):
        if self.graph_type == 'acc_conns':
            start_acc_conns(self)

    def draw(self):
        nx.draw(self._nx_graph, pos=nx.spring_layout(self._nx_graph), with_labels=True)

    def graph_info(self):
        nb_num = self.num_neighbors
        if self.graph_type == 'ring':
            nb_num = 1 if self.directed else 2
        elif self.graph_type == 'complete':
            nb_num = self.n - 1
        info = "{} ({}), N: {}, NB: {}, TV: {}".format(self.graph_type,
                                                       'directed' if self.directed else 'undirected',
                                                       self.n, nb_num,
                                                       self.time_varying)
        return info

    def as_numpy_array(self):
        return nx.to_numpy_array(self._nx_graph)


def nx_graph_from_saved_lists(np_array, directed=False):
    return nx.from_numpy_array(np.asarray(np_array), create_using=nx.DiGraph() if directed else nx.Graph())


if __name__ == "__main__":
    gm = GraphManager('sparse_clusters', [DummyNode(_) for _ in range(80)], directed=True, num_neighbors=2,
                      **{'cluster_conns': 0.33, 'clusters': 4, 'cluster_directed': True})
                      # **{'cluster_directed': True, 'clusters': [10, 10, 10, 10]})
    # gm.draw()

    adj_mx = nx.to_numpy_array(gm._nx_graph)
    for no in gm.nodes:
        adj_mx[no.id, no.id] = 0
        print(no.id,#  len(gm.get_peers(no.id)),
              "{}-{}".format(sum(adj_mx[no.id, :] > 0), sum(adj_mx[:, no.id] > 0)),
              '\t', [p.id for p in gm.get_peers(no.id)])
