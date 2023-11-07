from plot.visualize import side_by_side, plt, resolve_timeline, parse_timeline, read_graph, read_agents
from scipy.stats import ttest_ind
from data.metrics.gini import gini
import numpy as np


def print_table_2(viz, start_epoch=20, metric='avg'):
    ### Model Conns Lol Politics
    ### Gru
    ###       Solo  0   0
    print("----Table----")
    print("\\begin{table*}[b]\n\\centering\n\\footnotesize\n\\begin{tabular}{l l c c}\n\\hline")
    g1, g2 = np.unique([k[k.index('(') + 1:-1] for k in viz.keys()])
    print(
        "\\textbf{Model (dataset)} & \\textbf{Connections} &  \\textbf{leagueoflegends} &  \\textbf{politics}  \\\\\n\\hline")
    baselines = {}
    for model in ['GRU', 'BERT']:
        for group in ['small', 'big']:
            print(model + f" ({group})" + "  & & & \\\\\n\\hline")
            for col in ['Solo', 'Sparse', 'Sparse-clustered', 'Acc', 'Acc (val)', 'AUCCCR']:
                print(f"& {col} ", end='')
                for g in [g1, g2]:
                    k = f"{model} - {group} ({g})"
                    v = viz[k]
                    vv = v['viz'][col]
                    if metric == 'avg':
                        t, accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'])[1:]
                        max_a = round(max(t), 2)
                    else:
                        agent_accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'], agg_fn=np.array)[2]
                        avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in agent_accs]))
                        max_a = round(float(avg_agents_max), 2)
                        t = np.mean([np.max(np.array(sim), axis=0) for sim in agent_accs], axis=-1)
                        start_epoch = 0

                    # b_key = col + "-" + model + ' (' + group + ')'
                    b_key = f"{g} -> {model} ({group})"
                    if b_key not in baselines:
                        print("&", max_a, " " if g == g1 else '\\\\', end='')
                        if col == 'Solo':  # 'Sparse':
                            baselines[b_key] = [t[start_epoch:], max_a]
                    else:
                        baseline = baselines[b_key]
                        rel_inc = round((max_a - baseline[1]) / baseline[1] * 100, 2)
                        p_val = ttest_ind(baseline[0], t[start_epoch:])[1]
                        p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format(
                            '+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                        if rel_inc > 0:
                            p_text = "\\textbf{" + p_text + "}"
                        if p_val < 0.05:
                            p_text += " \\textbf{**}"
                        print("&", p_text, " " if g == g1 else '\\\\', end='')
                print()
            print('\\hline')

    print("\\end{tabular}\n\\caption{\label{tbl:name} " + metric + ".}\n\\end{table*}")


def neigh():
    from matplotlib.colors import ListedColormap
    from p2p.graph_manager import GraphManager, DummyNode

    def plot_graph(gm=None, mx=None):
        if mx is None:
            mx = gm.as_numpy_array()
        # remove self connections
        np.fill_diagonal(mx, 0)
        mx[mx > 0] = 1
        for i in range(int(mx.shape[0] / 2)):
            mx[i][mx[i] > 0] = 3
            mx[i, int(mx.shape[0] / 2):][(mx[i] > 0)[int(mx.shape[0] / 2):]] = 2
        for i in range(int(mx.shape[0] / 2), mx.shape[0]):
            mx[i, :int(mx.shape[0] / 2)][(mx[i] > 0)[:int(mx.shape[0] / 2)]] = 2
        plt.pcolormesh(mx, cmap=ListedColormap(['white', 'blue', 'green', 'red']))
        plt.plot([50, 50], [0, 100], [0, 100], [50, 50], color='lightgray')
        plt.xlabel('Agent ID')
        plt.ylabel('Agent ID')

    plot_graph(GraphManager('sparse_clusters', [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=3,
                            cluster_conns=0))
    plot_graph(GraphManager('sparse', [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=3))
    plot_graph(GraphManager('sparse_clusters', [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=2,
                            cluster_conns=1))
    # SMALL
    plot_graph(mx=np.array(
        read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10')))  # AUCCCR
    plot_graph(mx=np.array(
        read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34')))  # AUCCCR clusters
    plot_graph(mx=np.array(read_graph(
        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_08_46')))  # Accuracy connections
    plot_graph(mx=np.array(read_graph(
        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_00_28')))  # Accuracy val connections


def plot_communication():
    def plot_graph(path):
        agents = read_agents(path)
        mx = np.zeros((len(agents), len(agents)))
        for ak, av in agents.items():
            # Flatten list
            sent = [subitem for item in av['sent_to'] for subitem in item]
            values, counts = np.unique(sent, return_counts=True)
            mx[int(ak), values] = counts
        plt.imshow(mx, cmap="GnBu")
        plt.colorbar()
        plt.show()

    # 2 Clusters #
    # Oracle
    # plot_graph()
    # Sparse
    # plot_graph()
    # Sparse clusters
    # plot_graph()
    # AUCCCR
    # plot_graph()
    # DAC


ROT_2C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_27'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_03'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_33'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_56'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_20'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_00_40'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_18'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_28'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_36'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_34'
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_44'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_13'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_23_00'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_55'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_30'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_08-10-2023_22_02'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_04_10'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_11_06'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_34'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_13_19'
                     ],
        }
    },
}


ROT_4C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_22'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_04'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_22'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_35'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_43'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_00_57'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_18_(1)'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_05'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_27'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_29'
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_38'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_15'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_04'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_27'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_24'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_23_19'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_44'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_10_35'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_26'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_54'
                     ],
        }
    },
}


SWAP_2C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_20'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_01'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_20'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_52'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_38'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_01_05'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_03'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_08'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_25'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_27'
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_22'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_06'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_14'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_17'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_13'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_23_25'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_52'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_11_05'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_23'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_49'
                     ],
        }
    },
}




SWAP_4C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_10'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_06'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_06'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_39'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_09'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_01_45'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_47'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_07'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_21'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_29_(1)'
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_28'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_13_54'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_18'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_05'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_12'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_22_43'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_42'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-09-2023_14_38'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_24'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_54_(1)'
                     ],
        }
    },
}


PART_2C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_21-09-2023_23_16'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_05'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_17'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_36'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_15_21'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_02_24'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_10'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_58'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_08'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_24'
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_13_44'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_13_05'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_33'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_52'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_11_30'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_00_53'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_09_58'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-09-2023_14_45'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_18'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_05_24'
                     ],
        }
    },
}


PRACT_4C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_07'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_57_(1)'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_22_48'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_15'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_17_00'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_07-10-2023_01_32'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_42'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_37'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_01'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_17_57'
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_16'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_03'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_05'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_27'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_18_52'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_04_53'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_09_49'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_06'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_12'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_04_58'
                     ],
        }
    },
}


PATHO_4C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_02'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_57'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_22_55'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_14'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_02'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_02_43'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_40'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_27'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_18_53'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_17_59'
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_14'
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_10'
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_24'
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_24'
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_18_56'
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_03_17'
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_03_33'
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_28-09-2023_06_15'
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_09'
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_05_04'
                     ],
        }
    },
}


def plot_cifar_2C_rot():
    # Rotations = [0, 180]
    # Examples=400
    info = {
        # '0\\degree': 'cifar10-c0->test_model-sparse_categorical_accuracy',
        # '180\\degree': 'cifar10-c1->test_model-sparse_categorical_accuracy',
        'Train size=400': 'test_model-sparse_categorical_accuracy'
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_27'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_03'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_33'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_56'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_20'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_00_40'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_18'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_28'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_36'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_34'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}],
                 # title="Trainset size=400"
                 )

    # Examples=100
    info = {
        # '0\\degree': 'cifar10-c0->test_model-sparse_categorical_accuracy',
        # '180\\degree': 'cifar10-c1->test_model-sparse_categorical_accuracy',
        'Train size=100': 'test_model-sparse_categorical_accuracy'
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_44'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_13'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_23_00'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_55'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_30'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_08-10-2023_22_02'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_04_10'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_11_06'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_34'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_13_19'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [30, 44], 'step': 2},
                     {'y': [30, 44], 'step': 2}],
                 # title="Trainset size=100"
                 )


def plot_cifar_4C_rot():
    # Rotations = [0, 90, 180, 270]
    info = {
        # '0\\degree': 'cifar10-c0->test_model-sparse_categorical_accuracy',
        # '90\\degree': 'cifar10-c1->test_model-sparse_categorical_accuracy',
        # '180\\degree': 'cifar10-c2->test_model-sparse_categorical_accuracy',
        # '270\\degree': 'cifar10-c3->test_model-sparse_categorical_accuracy',
        'Train size=400': 'test_model-sparse_categorical_accuracy'
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_22'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_04'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_22'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_35'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_43'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_00_57'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_18_(1)'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_05'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_27'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_29'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [40, 54], 'step': 2},
                     # {'y': [40, 54], 'step': 2},
                     # {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])

    info = {
        'Train size=100': 'test_model-sparse_categorical_accuracy'
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_38'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_15'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_04'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_27'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_24'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_23_19'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_44'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_10_35'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_26'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_54'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     {'y': [40, 54], 'step': 2}])


def plot_cifar_2C_swap():
    # label_swaps = [[], [0, 2]]
    info = {
        'N=400': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_20'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_01'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_20'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_52'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_38'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_01_05'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_03'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_08'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_25'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_27'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])
    info = {
        'N=100': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_22'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_06'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_14'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_17'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_13'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_23_25'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_52'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_11_05'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_23'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_49'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     {'y': [40, 54], 'step': 2}])


def plot_cifar_4C_swap():
    # label_swaps = [[], [[0, 1]], [[2, 3]], [[4, 5]]]
    info = {
        'Swap=1, N=400': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_10'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_06'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_06'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_39'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_09'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_01_45'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_47'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_07'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_21'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_29_(1)'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])

    info = {
        'Swap=1, N=100': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_28'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_13_54'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_18'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_05'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_12'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_22_43'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_42'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-09-2023_14_38'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_24'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_54_(1)'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])


def plot_cifar_2C_part():
    # label_pertitions = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]
    info = {
        'N=400': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_21-09-2023_23_16'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_05'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_17'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_36'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_15_21'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_02_24'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_10'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_58'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_08'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_24'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])

    info = {
        'N=100': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_13_44'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_13_05'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_33'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_52'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_11_30'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_00_53'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_09_58'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-09-2023_14_45'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_18'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_05_24'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[
                     # {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])


def plot_cifar_pathological():
    # pathological non-IID
    info = {
        'N=400': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_02'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_57'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_22_55'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_14'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_02'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_02_43'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_40'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_27'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_18_53'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_17_59'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    info = {
        'N=100': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_14'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_10'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_24'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_24'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_18_56'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_03_17'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_03_33'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_28-09-2023_06_15'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_09'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_05_04'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])


def plot_cifar_practical():
    # practical non-IID
    info = {
        'N=400': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_07'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_57_(1)'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_22_48'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_15'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_17_00'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_07-10-2023_01_32'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_42'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_37'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_01'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_17_57'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [50, 70], 'step': 2}])

    info = {
        'N=100': 'test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_16'
                   ],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_03'
                   ],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_05'
                   ],
        'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_27'
                ],
        'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_18_52'
                      ],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_04_53'
                  ],
        'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_09_49'
                ],
        'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_06'
                     ],
        'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_12'
                     ],
        'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_04_58'
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])


def plot_reddit_small():
    # LoL <-> Politics
    info = {
        'LoL-Politics': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_13-10-2023_13_40',
                   ],
        'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_45',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_05-11-2023_23_38',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # LoL <-> NBA
    info = {
        'LoL-NBA': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_15',
                   ],
        'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_46',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_05-11-2023_23_54',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # LoL <-> Bitcoin
    info = {
        'LoL-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_12',
                   ],
        'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_25',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_05-11-2023_21_51',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # Politics <-> NBA
    info = {
        'Politics-NBA': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_10',
                   ],
        'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_25_(1)',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_28-10-2023_06_16',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # Politics <-> Bitcoin
    info = {
        'Politics-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_02',
                   ],
        'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_14_42',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_28-10-2023_01_57',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],

    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # NBA <-> Bitcoin
    info = {
        'NBA-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_20_57',
                   ],
        'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_14_39',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_28-10-2023_01_35',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],

    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # LoL-Politics-NBA-Bitcoin
    info = {
        'LoL-Politics-NBA-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_200A_1000E_50B_sparse_clusters(directed-3)_18-10-2023_14_27',
                   ],
        'Sparse': ['conns/reddit/GossipPullAgent_200A_1000E_50B_sparse(directed-3)_20-10-2023_21_42',
                   ],
        'DAC': ['conns/reddit/DacAgent_200A_1000E_50B_sparse(directed-3)_21-10-2023_16_51',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])


def plot_reddit_large():
    # LoL <-> Politics
    info = {
        'LoL-Politics': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/',
                   ],
        'Sparse': ['conns/reddit/',
                   ],
        'AUCCCR': ['conns/reddit/',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # LoL <-> NBA
    info = {
        'LoL-NBA': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/',
                   ],
        'Sparse': ['conns/reddit/',
                   ],
        'AUCCCR': ['conns/reddit/',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # LoL <-> Bitcoin
    info = {
        'LoL-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/',
                   ],
        'Sparse': ['conns/reddit/',
                   ],
        'AUCCCR': ['conns/reddit/',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # Politics <-> NBA
    info = {
        'Politics-NBA': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/',
                   ],
        'Sparse': ['conns/reddit/',
                   ],
        'AUCCCR': ['conns/reddit/',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # Politics <-> Bitcoin
    info = {
        'Politics-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/',
                   ],
        'Sparse': ['conns/reddit/',
                   ],
        'AUCCCR': ['conns/reddit/',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # NBA <-> Bitcoin
    info = {
        'NBA-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/',
                   ],
        'Sparse': ['conns/reddit/',
                   ],
        'AUCCCR': ['conns/reddit/',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])

    # LoL-Politics-NBA-Bitcoin
    info = {
        'LoL-Politics-NBA-Bitcoin': 'test_model-accuracy_no_oov',
    }
    data = {
        'Oracle': ['conns/reddit/GossipPullAgent_200A_300E_50B_sparse_clusters(directed-3)_31-08-2023_22_15',
                   ],
        'Sparse': ['conns/reddit/',
                   ],
        'AUCCCR': ['conns/reddit/',
                   ],
        'DAC': ['conns/reddit/',
                ],
        'D-Cliques': ['conns/reddit/',
                      ],
        'DiPLe': ['conns/reddit/',
                  ],
        'L2C': ['conns/reddit/',
                ],
        'PANMGrad': ['conns/reddit/',
                     ],
        'PANMLoss': ['conns/reddit/',
                     ],
        'PENS': ['conns/reddit/',
                 ],
    }
    plot_results(info, data, n_rows=1,
                 axis_lim=[{'y': [40, 54], 'step': 2}])


def plot_results(info, data, n_rows=1, axis_lim=None, title=None, metric='avg', analyze='accs'):
    # analyze in [all, plot, accs, table, msgs, gini]
    viz = {
        lbl: {
            'x_axis': 'epoch',
            'metric': d_metric,
            'viz': data
        } for lbl, d_metric in info.items()
    }
    if analyze in ['all', 'plot']:
        side_by_side(viz, n_rows=n_rows, axis_lim=axis_lim)
    # """
    if analyze in ['all', 'accs']:
        for vk, vv in viz.items():
            print(vk)
            for k, v in vv['viz'].items():
                t = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1]
                accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'], agg_fn=np.array)[2]
                avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in accs]))
                start, end = identify_stable_region(t)

                print('\t-', k.ljust(10),
                      round(max(t), 2), f"({round(max(t[start:end]), 2)}% - {start},{end-start})".ljust(13),
                      "\tAvg-Agent-Max", str(round(float(avg_agents_max), 2)).ljust(5))

    if analyze in ['all', 'table']:
        create_table(viz, title=title, metric=metric, print_means=False)

    if analyze in ['all', 'msgs']:
        # plot_comms(viz)
        print("==== Messages ====\n\t\tReceived\tSent")
        for vk, vv in viz.items():
            for k, v in vv['viz'].items():
                print(k)
                msgs(v, verbose=2)

    if analyze in ['all', 'gini']:
        print("=== GINI ===")
        for vk, vv in viz.items():
            print(vk)
            for k, v in vv['viz'].items():
                gini_coef, avg_msgs = msgs(v, verbose=0)
                print('\t', k.ljust(10), "{:.3f} ({})".format(gini_coef, round(avg_msgs)))


def find_longest_consecutive_streak(numbers):
    if numbers is None or len(numbers) == 0:
        return []

    longest_streak = []
    current_streak = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_streak.append(numbers[i])
        else:
            if len(current_streak) > len(longest_streak):
                longest_streak = current_streak.copy()
            current_streak = [numbers[i]]

    # Check if the last streak is longer than the longest streak
    if len(current_streak) > len(longest_streak):
        longest_streak = current_streak

    return longest_streak


def identify_stable_region(values, threshold=1):
    # Calculate the derivative
    accuracy_derivative = np.diff(values)

    # Find the stable region indices where the derivative is close to zero
    stable_region_indices = np.where(np.abs(accuracy_derivative) < threshold)[0] + 1  # Add 1 to align with epoch numbers
    stable_region_indices = find_longest_consecutive_streak(list(stable_region_indices))

    """
    print("Stable Region Indices:", stable_region_indices)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    # Highlight the stable region
    plt.axvline(x=stable_region_indices[0], color='red', linestyle='--', label='Stable Region Start')
    plt.axvline(x=stable_region_indices[-1], color='green', linestyle='--', label='Stable Region End')
    plt.legend()
    plt.show()
    """
    return stable_region_indices[0], stable_region_indices[-1]


def msgs(path, verbose=0):
    from plot.visualize import read_agents
    if verbose:
        print('ID\tReceived\tSent')
    rec, sent = [], []
    if isinstance(path, str):
        path = [path]
    all_vals = {'gini': [], 'sent': []}
    for p in path:
        agents = read_agents(p)
        for k, v in agents.items():
            if verbose > 2:
                print(k, '\t', sum(v['useful_msg']), '\t', sum(v['sent_msg']))
            rec.append(sum(v['useful_msg']))
            sent.append(sum(v['sent_msg']))
        if verbose > 2:
            print("-----------")
        if verbose > 1:
            print(f"\t- Min\t{np.min(rec)}\t{np.min(sent)}")
            print(f"\t- Max\t{np.max(rec)}\t{np.max(sent)}")
            print(f"\t- Avg\t{np.mean(rec)}+-{np.std(rec).round()}\t{np.mean(sent)}+-{np.std(sent).round()}")
            print(f"\t- Gini\t{gini(sent)}")
        all_vals['gini'].append(gini(sent))
        all_vals['sent'].append(np.mean(sent))

    return float(np.mean(all_vals['gini'])), float(np.mean(all_vals['sent']))


def plot_comms(viz):
    # from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size': 20})
    for vk, vv in viz.items():
        for k, v in vv['viz'].items():
            # v = ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_44']
            agents = read_agents(v[0])
            mx = np.zeros((len(agents), len(agents)))
            for i in range(len(agents)):
                for com in agents[str(i)]['sent_to']:
                    for j in com:
                        mx[i, j] += 1
            plt.pcolormesh(mx, cmap='Blues') # , cmap=ListedColormap(['white', 'blue']))
            plt.plot([50, 50], [0, 100], [0, 100], [50, 50], color='lightgray')
            plt.xlabel('Agent ID')
            plt.ylabel('Agent ID')
            cbar = plt.colorbar()
            cbar.set_label("# of agent sent messages")
            plt.tight_layout()
            plt.savefig(f"plot/plots/{k}.pdf")
            plt.clf()


def create_table(viz, title=None, start_epoch=20, metric='both', print_means=True):
    # Method C1-Cn Mean (std)

    print("----Table----")
    print("\\begin{table*}[b]\n\\centering\n%\\footnotesize\n\\begin{tabular}{l " + ' '.join(
        ['c'] * len(viz)) + " c}\n\\hline")

    if title:
        print("\\multicolumn{" + str(len(viz) + 2) + "}{c}{\\textbf{" + title + "}} \\\\")

    print("\\textbf{Method} & " + ' '.join(
        ["\\textbf{" + k + "} &" for k in viz.keys()]) + "\\textbf{Mean}  \\\\\n\\hline")
    seen = set()
    dist_items = [x for x in [subitem for item in [v['viz'].keys() for v in viz.values()] for subitem in item] if
                  not (x in seen or seen.add(x))]
    baselines = {}
    means = {}
    a_means = {}
    for item in dist_items:
        means[item] = []
        a_means[item] = []
        if item not in ['Oracle', 'Solo']:
            print(item, end='')
        for k, v in viz.items():
            vv = v['viz'][item]
            agents_max, max_a = 0, 0
            if metric == 'both':
                t, accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'])[1:]
                max_a = round(max(t), 2)

                agent_accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'], agg_fn=np.array)[2]
                avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in agent_accs]))
                agents_max = round(float(avg_agents_max), 2)
                # t = np.mean([np.max(np.array(sim), axis=0) for sim in agent_accs], axis=-1)
            elif metric == 'avg':
                t, accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'])[1:]
                max_a = round(max(t), 2)
            else:
                agent_accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'], agg_fn=np.array)[2]
                avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in agent_accs]))
                max_a = round(float(avg_agents_max), 2)
                t = np.mean([np.max(np.array(sim), axis=0) for sim in agent_accs], axis=-1)

            means[item].append(max_a)
            a_means[item].append(agents_max)

            if item == 'Oracle' or item == 'Solo':
                baselines['Oracle-' + k] = (t[start_epoch:], max_a)
                baselines['Oracle-a-' + k] = (t[start_epoch:], agents_max)
                continue
            elif item == 'Sparse':
                baselines[k] = (t[start_epoch:], max_a)
                num = "{:.2f}".format(round(max_a, 2))
                if agents_max > 0:
                    num += " ({:.2f})".format(round(agents_max, 2))
                print(f" & {num}", end='')
                if k == list(viz.keys())[-1]:
                    if print_means:
                        nm = "{:.2f}".format(round(float(np.mean(means[item])), 2))
                        if sum(a_means[item]) > 0:
                            nm += " ({:.2f})".format(round(float(np.mean(a_means[item])), 2))
                        print(f" & {nm}", end='')
                    print(" \\\\")
            else:
                baseline = baselines[k]
                rel_inc = round((max_a - baseline[1]) / baseline[1] * 100, 2)
                """
                p_val = ttest_ind(baseline[0], t[start_epoch:])[1]
                p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                if rel_inc > 0:
                    p_text = "\\textbf{" + p_text + "}"
                if p_val < 0.05:
                    p_text += " \\textbf{**}"
                """
                num = "{:.2f}".format(round(max_a, 2))  # + (" \\textbf{**}" if p_val<0.05 else "")
                if agents_max > 0:
                    num += " ({:.2f})".format(round(agents_max, 2))
                if rel_inc > 0:
                    num = "\\textbf{" + num + "}"
                print(f" & {num}", end='')
                if k == list(viz.keys())[-1]:
                    if print_means:
                        m = float(np.mean(means[item]))
                        nm = "{:.2f}".format(round(m, 2))
                        if sum(a_means[item]) > 0:
                            nm += " ({:.2f})".format(round(float(np.mean(a_means[item])), 2))
                        if m > float(np.mean(means['Sparse'])):
                            nm = "\\textbf{" + nm + "}"
                        print(f" & {nm}", end='')
                    print(" \\\\")

    print('\\arrayrulecolor{gray}\\hline')
    print("\\textcolor[gray]{.5} {Oracle}", end='')
    for k in viz.keys():
        num = "{:.2f}".format(round(baselines['Oracle-' + k][1], 2))
        if baselines['Oracle-a-' + k][1] > 0:
            num += " ({:.2f})".format(round(baselines['Oracle-a-' + k][1], 2))
        print(" & \\textcolor[gray]{.5} {" + str(num) + '}', end='')
        if k == list(viz.keys())[-1]:
            if print_means:
                nm = "{:.2f}".format(round(float(np.mean(means['Oracle'])), 2))
                if sum(a_means['Oracle']) > 0:
                    nm += " ({:.2f})".format(round(float(np.mean(a_means['Oracle'])), 2))
                print("& \\textcolor[gray]{.5} {" + str(nm) + "}", end='')
            print(" \\\\")
    print("\\end{tabular}\n\\caption{\\label{tbl:name} " + metric + ".}\n\\end{table*}")


def do_parse():
    import os, json
    order = ["'rotations': [0, 180]", "'rotations': [0, 90, 180, 270]", "'label_swaps': [[], [[0, 2]]]",
             "'label_swaps': [[], [[0, 1]], [[2, 3]], [[4, 5]]]", "'label_partitions': [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]",
             "'mode': 'pathological non-IID'", "'mode': 'practical non-IID'"
             ]
    fnames = [f for f in os.listdir('/Users/robert/Downloads/') if f.endswith('.json')]
    f_info = {}
    for filename in fnames:
        with open('/Users/robert/Downloads/' + filename, "r") as infile:
            f_json = json.loads(infile.read())
        info = f_json['info']
        dp = info['agent_data_pars']
        d_info = dp[dp.index('32, ') + 4:-2]
        seed = info['sim_pars']['seed']
        f_info[d_info + " " + seed] = filename.replace('.json', '')

    def custom_sort(k):
        s = k[-4:]
        key = k[:-4]
        l = key.split(", '")[0]
        r = str(order.index(l)) + k[len(l):] + s
        return r

    for k in sorted(f_info, key=custom_sort):
        print(k, '\n\t' + f_info[k], '\n')


if __name__ == '__main__':
    pass
    # plot_cifar_2C_rot()


