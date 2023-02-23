from plot.visualize import side_by_side, plt, resolve_timeline, parse_timeline, read_graph
from scipy.stats import ttest_ind
import numpy as np


def exp_1():
    colors = ['r', 'g', 'b', 'indigo', 'orange']
    viz = {
        'Reddit small (leagueoflegends)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-leagueoflegends->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_18-02-2023_21_55',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_09_04',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_09_54',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_10_45',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_11_37'],

                'Sparse': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_18-02-2023_22_50',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_16_24',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_18_16',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_20_15',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_22_14'],

                'Sparse-clustered': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_18-02-2023_22_53',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_08_22',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_10_18',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_12_30',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_14_31'],

                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-02-2023_23_11',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_15_59',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_18_00',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_20_13',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_22_23'],

                # 'KMeans': ['conns/exp1/small/P2PAgent_100A_100E_50B_kmeans(directed-3)_19-02-2023_00_32'],

                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_07_59',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_09_54',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_11_56',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_14_00'],

                # 'AUCCCR-clusters': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34'],

            }
        },
        'Reddit small (politics)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-politics->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_18-02-2023_21_58',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_09_02',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_09_53',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_10_47',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_11_40'],

                'Sparse': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_18-02-2023_22_50',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_16_24',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_18_16',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_20_15',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_22-02-2023_22_14'],

                'Sparse-clustered': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_18-02-2023_22_53',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_08_22',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_10_18',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_12_30',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_22-02-2023_14_31'],

                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-02-2023_23_11',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_15_59',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_18_00',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_20_13',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_21-02-2023_22_23'],

                # 'KMeans': ['conns/exp1/small/P2PAgent_100A_100E_50B_kmeans(directed-3)_19-02-2023_00_32'],

                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_07_59',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_09_54',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_11_56',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_14_00'],

                # 'AUCCCR-clusters': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34'],
            }
        },
        'Reddit big (leagueoflegends)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-leagueoflegends->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_19-02-2023_01_49',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_14_52',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_17_10',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_19_50',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_22_32'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_19-02-2023_06_27'],
                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_19-02-2023_06_27'],
                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_19-02-2023_07_37'],
                # 'KMeans': ['conns/exp1/big/P2PAgent_100A_100E_50B_kmeans(directed-3)_19-02-2023_06_40'],
                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_19_14',
                          #  'conns/exp1/big/',
                          #  'conns/exp1/big',
                          #  'conns/exp1/big',
                          #  'conns/exp1/big',]
                           ],

                # 'AUCCCR-clusters': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_20-02-2023_02_58'],
            }
        },
        'Reddit big (politics)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-politics->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_19-02-2023_02_37',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_16_05',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_19_18',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_22_30',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_21-02-2023_01_28'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_19-02-2023_06_27'],
                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_19-02-2023_06_27'],
                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_19-02-2023_07_37'],
                # 'KMeans': ['conns/exp1/big/P2PAgent_100A_100E_50B_kmeans(directed-3)_19-02-2023_06_40'],

                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_19_14',
                          #  'conns/exp1/big/',
                          #  'conns/exp1/big',
                          #  'conns/exp1/big',
                          #  'conns/exp1/big',]
                           ],
                # 'AUCCCR-clusters': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_20-02-2023_02_58'],
            }
        },

        'BERT - Reddit small (leagueoflegends)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-leagueoflegends->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_07_46'],
                'Sparse': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_20-02-2023_11_40'],
                'Sparse-clustered': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_20-02-2023_16_21'],
                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_20-02-2023_21_58'],
                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_09_42'],
            }
        },
        'BERT - Reddit small (politics)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-politics->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_07_49'],
                'Sparse': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_20-02-2023_11_40'],
                'Sparse-clustered': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_20-02-2023_16_21'],
                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_20-02-2023_21_58'],
                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_09_42'],
            }
        },
        'BERT - Reddit big (leagueoflegends)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-leagueoflegends->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_19-02-2023_22_37'],
                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_21-02-2023_07_43'],
                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_21-02-2023_17_30'],
                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_22-02-2023_05_28'],
                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_00_25'],
            }
        },
        'BERT - Reddit big (politics)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-politics->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_03_04'],
                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_21-02-2023_07_43'],
                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_21-02-2023_17_30'],
                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_22-02-2023_05_28'],
                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_00_25'],
            }
        },
    }
    side_by_side(viz,
                 fig_size=(10*2, 8),
                 n_rows=2,
                 axis_lim=[
                    {'y': [0, 8], 'step': 1},
                    {'y': [0, 8], 'step': 1},
                    {'y': [0, 11], 'step': 1},
                    {'y': [0, 11], 'step': 1},
                    {'y': [0, 9], 'step': 1},
                    {'y': [0, 9], 'step': 1},
                    {'y': [0, 13], 'step': 1},
                    {'y': [0, 13], 'step': 1},
                 ])
    for vk, vv in viz.items():
        print(vk)
        for k, v in vv['viz'].items():
            t = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1]
            accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'], agg_fn=np.array)[2]
            avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in accs]))
            print('\t-', k, round(max(t), 2), "Avg-Agent-Max", round(float(avg_agents_max), 2))

    """
    vk = 'Reddit small (leagueoflegends)'
    vv = viz[vk]
    k = 'Solo'
    v = vv['viz'][k]
    accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'], agg_fn=np.array)[2]
    sim = accs[0]
    sim = np.array(sim)
    np.max(sim, axis=0)"""


def neigh():
    from matplotlib.colors import ListedColormap
    from p2p.graph_manager import GraphManager, DummyNode, build_from_clusters, build_from_classes

    def plot_graph(gm=None, mx=None):
        if mx is None:
            mx = gm.as_numpy_array()
        # remove self connections
        np.fill_diagonal(mx, 0)
        mx[mx > 0] = 1
        for i in range(int(mx.shape[0]/2)):
            mx[i][mx[i] > 0] = 3
            mx[i, int(mx.shape[0]/2):][(mx[i] > 0)[int(mx.shape[0]/2):]] = 2
        for i in range(int(mx.shape[0]/2), mx.shape[0]):
            mx[i, :int(mx.shape[0]/2)][(mx[i] > 0)[:int(mx.shape[0]/2)]] = 2
        plt.pcolormesh(mx, cmap=ListedColormap(['white', 'blue', 'green', 'red']))
        plt.plot([50, 50], [0, 100], [0, 100], [50, 50], color='lightgray')
        plt.xlabel('Agent ID')
        plt.ylabel('Agent ID')
    plot_graph(GraphManager('sparse_clusters',  [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=3, cluster_conns=0))
    plot_graph(GraphManager('sparse',  [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=3))
    plot_graph(GraphManager('sparse_clusters',  [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=2, cluster_conns=1))

    plot_graph(mx=np.array(read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10')))
    plot_graph(mx=np.array(read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34')))

    plot_graph(mx=np.array(read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-02-2023_23_11')))

    plot_graph(mx=np.array(read_graph('conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_22-02-2023_05_28')))


if __name__ == '__main__':
    exp_1()


# Single run
# Reddit small (leagueoflegends)
# 	- Solo 7.1 Avg-Agent-Max 10.5
# 	- Sparse 7.37 Avg-Agent-Max 10.21
# 	- Sparse-clustered 7.15 Avg-Agent-Max 10.13
# 	- Acc 7.14 Avg-Agent-Max 10.09
# Reddit small (politics)
# 	- Solo 6.91 Avg-Agent-Max 10.03
# 	- Sparse 7.23 Avg-Agent-Max 10.17
# 	- Sparse-clustered 7.4 Avg-Agent-Max 10.28
# 	- Acc 7.16 Avg-Agent-Max 10.44
# Reddit big (leagueoflegends)
# 	- Solo 8.8 Avg-Agent-Max 10.43
# 	- Sparse 9.39 Avg-Agent-Max 10.93
# 	- Sparse-clustered 9.09 Avg-Agent-Max 10.66
# 	- Acc 9.43 Avg-Agent-Max 10.94
# Reddit big (politics)
# 	- Solo 10.22 Avg-Agent-Max 11.8
# 	- Sparse 10.34 Avg-Agent-Max 11.85
# 	- Sparse-clustered 10.14 Avg-Agent-Max 11.77
# 	- Acc 10.28 Avg-Agent-Max 12.03
# BERT - Reddit small (leagueoflegends)
# 	- Solo 7.51 Avg-Agent-Max 9.35
# 	- Sparse 7.98 Avg-Agent-Max 9.55
# 	- Sparse-clustered 8.09 Avg-Agent-Max 9.79
# 	- Acc 8.2 Avg-Agent-Max 9.81
# BERT - Reddit small (politics)
# 	- Solo 6.73 Avg-Agent-Max 9.19
# 	- Sparse 7.24 Avg-Agent-Max 9.52
# 	- Sparse-clustered 7.35 Avg-Agent-Max 9.51
# 	- Acc 7.29 Avg-Agent-Max 9.43
# BERT - Reddit big (leagueoflegends)
# 	- Solo 10.01 Avg-Agent-Max 11.51
# 	- Sparse 10.71 Avg-Agent-Max 11.97
# 	- Sparse-clustered 10.66 Avg-Agent-Max 11.93
# 	- Acc 10.8 Avg-Agent-Max 12.1
# BERT - Reddit big (politics)
# 	- Solo 12.13 Avg-Agent-Max 13.67
# 	- Sparse 12.79 Avg-Agent-Max 13.98
# 	- Sparse-clustered 12.46 Avg-Agent-Max 13.7
# 	- Acc 12.71 Avg-Agent-Max 13.86
