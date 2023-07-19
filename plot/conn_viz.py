from plot.visualize import side_by_side, plt, resolve_timeline, parse_timeline, read_graph
from scipy.stats import ttest_ind
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


def plot_cifar_2C_rot():
    # Rotations = [0, 180]
    info = {
        '0\\degree': 'cifar10-c0->test_model-sparse_categorical_accuracy',
        '180\\degree': 'cifar10-c1->test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_300E_32B_sparse_clusters(directed-3)_14-07-2023_21_27'],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_300E_32B_sparse(directed-3)_14-07-2023_21_30'],
        'Sparse-clusters': ['conns/cifar10/GossipPullAgent_100A_300E_32B_sparse_clusters(directed-2)_14-07-2023_23_28'],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_300E_32B_aucccr(directed-3)_18-07-2023_19_28'],
        'Dac': ['conns/cifar10/DacAgent_100A_300E_32B_sparse(directed-3)_15-07-2023_01_32'],
        'DClique': ['conns/cifar10/DCliqueAgent_100A_300E_32B_d-cliques(directed-3)_17-07-2023_20_03'],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_300E_32B_sparse(directed-3)_17-07-2023_23_25'],
        # 'L2C': ['conns/cifar10/'],
        'PanmGrad': ['conns/cifar10/PanmAgent_100A_300E_32B_sparse(directed-3)_18-07-2023_21_19'],
        'PanmLoss': ['conns/cifar10/PanmAgent_100A_300E_32B_sparse(directed-3)_18-07-2023_22_13'],
        'Pens': ['conns/cifar10/PensAgent_100A_300E_32B_sparse(directed-3)_15-07-2023_01_28'],
    }
    plot_results(info, data, n_rows=2,
                 axis_lim=[
                     {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])


def plot_cifar_4C_rot():
    # Rotations = [0, 90, 180, 270]
    info = {
        '0\\degree': 'cifar10-c0->test_model-sparse_categorical_accuracy',
        '90\\degree': 'cifar10-c1->test_model-sparse_categorical_accuracy',
        '180\\degree': 'cifar10-c2->test_model-sparse_categorical_accuracy',
        '270\\degree': 'cifar10-c3->test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/'],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_300E_32B_sparse(directed-3)_15-07-2023_12_49'],
        'Sparse-clusters': ['conns/cifar10/'],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_300E_32B_aucccr(directed-3)_18-07-2023_22_51'],
        'Dac': ['conns/cifar10/DacAgent_100A_300E_32B_sparse(directed-3)_15-07-2023_13_53'],
        'DClique': ['conns/cifar10/DCliqueAgent_100A_300E_32B_d-cliques(directed-3)_16-07-2023_07_21'],
        'DiPLe': ['conns/cifar10/DiPLeAgent_100A_300E_32B_sparse(directed-3)_17-07-2023_00_54'],
        # 'L2C': ['conns/cifar10/'],
        'PanmGrad': ['conns/cifar10/PanmAgent_100A_300E_32B_sparse(directed-3)_18-07-2023_23_17'],
        'PanmLoss': ['conns/cifar10/PanmAgent_100A_300E_32B_sparse(directed-3)_19-07-2023_00_01'],
        'Pens': ['conns/cifar10/PensAgent_100A_300E_32B_sparse(directed-3)_15-07-2023_13_32'],
    }
    plot_results(info, data, n_rows=2,
                 axis_lim=[
                     {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])


def plot_cifar_2C_swap():
    # label_swaps = [[], [0, 2]]
    info = {
        'None': 'cifar10-c0->test_model-sparse_categorical_accuracy',
        '0-2': 'cifar10-c1->test_model-sparse_categorical_accuracy',
    }
    data = {
        'Oracle': ['conns/cifar10/GossipPullAgent_100A_300E_32B_sparse_clusters(directed-3)_18-07-2023_18_29'],
        'Sparse': ['conns/cifar10/GossipPullAgent_100A_300E_32B_sparse(directed-3)_18-07-2023_18_18'],
        'Sparse-clusters': ['conns/cifar10/GossipPullAgent_100A_300E_32B_sparse_clusters(directed-2)_18-07-2023_18_27'],
        'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_300E_32B_aucccr(directed-3)_19-07-2023_01_27'],
        'Dac': ['conns/cifar10/DacAgent_100A_300E_32B_sparse(directed-3)_18-07-2023_19_41'],
        # 'DClique': ['conns/cifar10/'],
        # 'DiPLe': ['conns/cifar10/'],
        # 'L2C': ['conns/cifar10/'],
        'PanmGrad': ['conns/cifar10/PanmAgent_100A_300E_32B_sparse(directed-3)_19-07-2023_02_34'],
        'PanmLoss': ['conns/cifar10/PanmAgent_100A_300E_32B_sparse(directed-3)_19-07-2023_03_41'],
        'Pens': ['conns/cifar10/PensAgent_100A_300E_32B_sparse(directed-3)_18-07-2023_19_22'],
    }
    plot_results(info, data, n_rows=2,
                 axis_lim=[
                     {'y': [40, 54], 'step': 2},
                     {'y': [40, 54], 'step': 2}])


def plot_results(info, data, n_rows=1, axis_lim=None):
    viz = {
        lbl: {
            'x_axis': 'epoch',
            'metric': metric,
            'viz': data
        } for lbl, metric in info.items()
    }
    side_by_side(viz, n_rows=n_rows, axis_lim=axis_lim)
    for vk, vv in viz.items():
        print(vk)
        for k, v in vv['viz'].items():
            t = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1]
            accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'], agg_fn=np.array)[2]
            avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in accs]))
            print('\t-', k, round(max(t), 2), "Avg-Agent-Max", round(float(avg_agents_max), 2))

    create_table(viz)

    """
    print("==== Messages ====")
    for vk, vv in viz.items():
        for k, v in vv['viz'].items():
            print(k)
            msgs(v)
        break
    # """


def msgs(path, verbose=False):
    from plot.visualize import read_agents
    if verbose:
        print('ID\tReceived\tSent')
    rec, sent = [], []
    if isinstance(path, str):
        path = [path]
    for p in path:
        agents = read_agents(p)
        for k, v in agents.items():
            if verbose:
                print(k, '\t', sum(v['useful_msg']), '\t', sum(v['sent_msg']))
            rec.append(sum(v['useful_msg']))
            sent.append(sum(v['sent_msg']))
        if verbose:
            print("-----------")
    print(f"\t- Min\t{np.min(rec)}\t{np.min(sent)}")
    print(f"\t- Max\t{np.max(rec)}\t{np.max(sent)}")
    print(f"\t- Avg\t{np.mean(rec)}+-{np.std(rec).round()}\t{np.mean(sent)}+-{np.std(sent).round()}", )


def create_table(viz, title=None, start_epoch=20, metric='avg'):
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
    for item in dist_items:
        means[item] = []
        if item not in ['Oracle', 'Solo']:
            print(item, end='')
        for k, v in viz.items():
            vv = v['viz'][item]
            if metric == 'avg':
                t, accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'])[1:]
                max_a = round(max(t), 2)
            else:
                agent_accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'], agg_fn=np.array)[2]
                avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in agent_accs]))
                max_a = round(float(avg_agents_max), 2)
                t = np.mean([np.max(np.array(sim), axis=0) for sim in agent_accs], axis=-1)

            means[item].append(max_a)

            if item == 'Oracle' or item == 'Solo':
                baselines['Oracle-' + k] = (t[start_epoch:], max_a)
                continue
            elif item == 'Sparse':
                baselines[k] = (t[start_epoch:], max_a)
                num = "{:.2f}".format(round(max_a, 2))
                print(f" & {num}", end='')
                if k == list(viz.keys())[-1]:
                    nm = "{:.2f}".format(round(float(np.mean(means[item])), 2))
                    print(f" & {nm} \\\\")
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
                if rel_inc > 0:
                    num = "\\textbf{" + num + "}"
                print(f" & {num}", end='')
                if k == list(viz.keys())[-1]:
                    m = float(np.mean(means[item]))
                    nm = "{:.2f}".format(round(m, 2))
                    if m > float(np.mean(means['Sparse'])):
                        nm = "\\textbf{" + nm + "}"
                    print(f" & {nm} \\\\")

    print('\\arrayrulecolor{gray}\\hline')
    print("\\textcolor[gray]{.5} {Oracle}", end='')
    for k in viz.keys():
        num = "{:.2f}".format(round(baselines['Oracle-' + k][1], 2))
        print(" & \\textcolor[gray]{.5} {" + str(num) + '}', end='')
        if k == list(viz.keys())[-1]:
            nm = "{:.2f}".format(round(float(np.mean(means['Oracle'])), 2))
            print("& \\textcolor[gray]{.5} {" + str(nm) + "} \\\\")
    print("\\end{tabular}\n\\caption{\\label{tbl:name} " + metric + ".}\n\\end{table*}")


if __name__ == '__main__':
    plot_cifar_2C_swap()
