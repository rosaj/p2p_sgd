from plot.visualize import side_by_side, plt, resolve_timeline, parse_timeline, read_graph
from scipy.stats import ttest_ind
import numpy as np


def exp_1_reddit():
    colors = ['r', 'g', 'b', 'indigo', 'orange', 'black']
    viz = {
        'GRU - small (leagueoflegends)': {
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

                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_08_46',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_10_55',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_12_58',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_15_07',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_17_17'],

                'Acc (val)': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_00_28',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_02_53',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_05_28',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_08_14',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_11_33'],

                # 'KMeans': ['conns/exp1/small/P2PAgent_100A_100E_50B_kmeans(directed-3)_19-02-2023_00_32'],

                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_07_59',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_09_54',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_11_56',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_14_00'],

                # 'AUCCCR-clusters': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34'],

            }
        },
        'GRU - small (politics)': {
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

                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_08_46',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_10_55',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_12_58',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_15_07',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_17_17'],

                'Acc (val)': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_00_28',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_02_53',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_05_28',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_08_14',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_11_33'],

                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_07_59',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_09_54',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_11_56',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_21-02-2023_14_00'],

                # 'AUCCCR-clusters': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34'],
            }
        },
        'GRU - big (leagueoflegends)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-leagueoflegends->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_19-02-2023_01_49',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_14_52',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_17_10',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_19_50',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_22_32'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_19-02-2023_06_27',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_27-02-2023_22_53',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_28-02-2023_11_11',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_28-02-2023_16_32',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_01-03-2023_03_29'],

                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_19-02-2023_06_27',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_18_14',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_23_29',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_03-03-2023_05_02',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_03-03-2023_10_56'],

                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_08_46',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_20_20',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_26-02-2023_02_51',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_26-02-2023_17_39',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_00_51'],

                'Acc (val)': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_05_29',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_14_54',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_23_27',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_11-03-2023_07_31',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_11-03-2023_16_00'],

                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_19_14',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_12_01',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_17_49',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_23_35',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_24-02-2023_05_40'],

                # 'AUCCCR-clusters': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_20-02-2023_02_58'],
            }
        },
        'GRU - big (politics)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-politics->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_19-02-2023_02_37',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_16_05',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_19_18',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_22_30',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_21-02-2023_01_28'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_19-02-2023_06_27',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_27-02-2023_22_53',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_28-02-2023_11_11',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_28-02-2023_16_32',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_01-03-2023_03_29'],

                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_19-02-2023_06_27',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_18_14',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_23_29',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_03-03-2023_05_02',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_03-03-2023_10_56'],

                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_08_46',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_20_20',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_26-02-2023_02_51',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_26-02-2023_17_39',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_00_51'],

                'Acc (val)': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_05_29',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_14_54',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_23_27',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_11-03-2023_07_31',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_11-03-2023_16_00'],

                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_19_14',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_12_01',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_17_49',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_23_35',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_24-02-2023_05_40'],

                # 'AUCCCR-clusters': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_20-02-2023_02_58'],
            }
        },

        'BERT - small (leagueoflegends)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-leagueoflegends->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_07_46',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_11_37',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_13_06',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_14_39',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_16_07'],

                'Sparse': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_20-02-2023_11_40',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_23-02-2023_20_14',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_23-02-2023_23_43',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_24-02-2023_03_27',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_24-02-2023_07_08'],

                'Sparse-clustered': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_20-02-2023_16_21',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_10_43',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_14_46',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_18_18',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_22_49'],

                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_01_10',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_06_25',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_11_43',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_17_19',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_26-02-2023_15_16'],

                'Acc (val)': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_10_13',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_13_32',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_17_01',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_20_39',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_19-03-2023_00_18'],

                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_09_42',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_25-02-2023_18_02',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_25-02-2023_21_36',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_26-02-2023_01_32',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_26-02-2023_05_25'],
            }
        },
        'BERT - small (politics)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-politics->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_07_49',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_11_43',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_13_15',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_14_50',
                         'conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_23-02-2023_16_27'],

                'Sparse': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_20-02-2023_11_40',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_23-02-2023_20_14',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_23-02-2023_23_43',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_24-02-2023_03_27',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_sparse(directed-3)_24-02-2023_07_08'],

                'Sparse-clustered': ['conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_20-02-2023_16_21',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_10_43',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_14_46',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_18_18',
                                     'conns/exp1/small/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_24-02-2023_22_49'],

                'Acc': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_01_10',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_06_25',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_11_43',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_25-02-2023_17_19',
                        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_26-02-2023_15_16'],

                'Acc (val)': ['conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_10_13',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_13_32',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_17_01',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_18-03-2023_20_39',
                              'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_19-03-2023_00_18'],

                'AUCCCR': ['conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_09_42',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_25-02-2023_18_02',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_25-02-2023_21_36',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_26-02-2023_01_32',
                           'conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_26-02-2023_05_25'],
            }
        },
        'BERT - big (leagueoflegends)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-leagueoflegends->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_19-02-2023_22_37',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_26-02-2023_21_47',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_27-02-2023_02_14',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_27-02-2023_06_29',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_27-02-2023_10_59'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_21-02-2023_07_43',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_06-03-2023_07_29',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_06-03-2023_18_23',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_07-03-2023_04_37',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_07-03-2023_16_14'],

                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_21-02-2023_17_30',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_28-02-2023_17_57',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_01_00',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_10_42',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_22_32'],

                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_05-03-2023_20_40',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_03-03-2023_13_19',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_04-03-2023_03_30',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_04-03-2023_16_28',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_05-03-2023_06_57'],

                'Acc (val)': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_16-03-2023_00_36',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_16-03-2023_12_19',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_17-03-2023_00_47',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_17-03-2023_12_11',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_17-03-2023_23_32'],

                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_00_25',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_12-03-2023_07_25',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_12-03-2023_19_25',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_13-03-2023_04_54',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_13-03-2023_19_15'],
            }
        },
        'BERT - big (politics)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-nwp-politics->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_20-02-2023_03_04',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_26-02-2023_22_11',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_27-02-2023_04_30',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_27-02-2023_09_15',
                         'conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_27-02-2023_15_27'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_21-02-2023_07_43',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_06-03-2023_07_29',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_06-03-2023_18_23',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_07-03-2023_04_37',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_07-03-2023_16_14'],

                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_21-02-2023_17_30',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_28-02-2023_17_57',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_01_00',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_10_42',
                                     'conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_02-03-2023_22_32'],

                'Acc': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_05-03-2023_20_40',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_03-03-2023_13_19',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_04-03-2023_03_30',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_04-03-2023_16_28',
                        'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_05-03-2023_06_57'],

                'Acc (val)': ['conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_16-03-2023_00_36',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_16-03-2023_12_19',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_17-03-2023_00_47',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_17-03-2023_12_11',
                              'conns/exp1/big/P2PAgent_100A_100E_50B_acc_conns(directed-3)_17-03-2023_23_32'],

                'AUCCCR': ['conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_23-02-2023_00_25',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_12-03-2023_07_25',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_12-03-2023_19_25',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_13-03-2023_04_54',
                           'conns/exp1/big/P2PAgent_100A_100E_50B_aucccr(directed-3)_13-03-2023_19_15'],
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
    print_table(viz, metric='avg')
    print_table(viz, metric='max agent acc')
    """
    for vk, vv in viz.items():
        print(vk)
        for k, v in vv['viz'].items():
            t = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1]
            accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'], agg_fn=np.array)[2]
            avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in accs]))
            print('\t-', k, round(max(t), 2), "Avg-Agent-Max", round(float(avg_agents_max), 2))
    """


def print_table(viz, start_epoch=20, metric='avg'):
    print("----Table----")
    print("\\begin{table*}[b]\n\\centering\n\\footnotesize\n\\begin{tabular}{l l c c c c}\n\\hline")
    g1, g2 = np.unique([k[k.index('(')+1:-1] for k in viz.keys()])
    print("\\textbf{Group} & \\textbf{Connections} &  \\textbf{GRU (small)} &  \\textbf{GRU (big)} & \\textbf{BERT (small)} &  \\textbf{BERT (big)}  \\\\\n\\hline")
    # print("\\textbf{Model} & \\textbf{Collaboration} &  \\textbf{" + g1 + "} & \\textbf{"+g2+"}\n\\hline")

    for g in [g1, g2]:
        print(g + "  & & & & & \\\\\n\\hline")
        baselines = {}
        for col in ['Solo', 'Sparse', 'Sparse-clustered', 'Acc', 'Acc (val)', 'AUCCCR']:
            print(f"& {col} & ", end='')
            for model in ['GRU', 'BERT']:
                for group in ['small', 'big']:
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

                    b_key = model + ' (' + group + ')'
                    if b_key not in baselines:
                        print(max_a, " & " if b_key != 'BERT (big)' else ' \\\\', end='')
                        if col == 'Sparse':
                            baselines[b_key] = [t[start_epoch:], max_a]
                    else:
                        baseline = baselines[b_key]
                        rel_inc = round((max_a - baseline[1]) / baseline[1] * 100, 2)
                        p_val = ttest_ind(baseline[0], t[start_epoch:])[1]
                        p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                        if rel_inc > 0:
                            p_text = "\\textbf{" + p_text + "}"
                        if p_val < 0.05:
                            p_text += " \\textbf{**}"
                        print(p_text, " & " if b_key != 'BERT (big)' else ' \\\\', end='')
            print()
        print('\\hline')

    print("\\end{tabular}\n\\caption{\label{tbl:name} " + metric + ".}\n\\end{table*}")


def exp_1_stackoverflow():
    colors = ['r', 'g', 'b', 'indigo', 'orange', 'black']
    viz = {
        'GRU - small (python)': {
            'x_axis': 'epoch',
            'colors': colors[:1],
            'metric': 'stackoverflow-nwp-python->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_10-03-2023_21_15'],

            }
        },
        'GRU - small (javascript)': {
            'x_axis': 'epoch',
            'colors': colors[:1],
            'metric': 'stackoverflow-nwp-javascript->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_10-03-2023_18_47'],
            }
        },
        'GRU - big (python)': {
            'x_axis': 'epoch',
            'colors': colors[:3],
            'metric': 'stackoverflow-nwp-python->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_11-03-2023_21_07'],
                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_28-03-2023_08_39'],
                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_28-03-2023_14_52']
            }
        },
        'GRU - big (javascript)': {
            'x_axis': 'epoch',
            'colors': colors[:3],
            'metric': 'stackoverflow-nwp-javascript->test_model-accuracy_no_oov',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_11-03-2023_20_14'],
                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_28-03-2023_08_39'],
                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_28-03-2023_14_52']
            }
        },
        'BERT - small (python)': {
            'x_axis': 'epoch',
            'colors': colors[:1],
            'metric': 'stackoverflow-nwp-python->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_21-03-2023_10_52'],
            }
        },
        'BERT - small (javascript)': {
            'x_axis': 'epoch',
            'colors': colors[:1],
            'metric': 'stackoverflow-nwp-javascript->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/small/P2PAgent_50A_100E_50B_sparse(directed-3)_21-03-2023_10_03'],
            }
        },
        'BERT - big (python)': {
            'x_axis': 'epoch',
            'colors': colors[:2],
            'metric': 'stackoverflow-nwp-python->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_22-03-2023_07_44'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_29-03-2023_18_40'],

                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_26-03-2023_16_48'],
            }
        },
        'BERT - big (javascript)': {
            'x_axis': 'epoch',
            'colors': colors[:2],
            'metric': 'stackoverflow-nwp-javascript->test_model-sparse_categorical_accuracy',
            'viz': {
                'Solo': ['conns/exp1/big/P2PAgent_50A_100E_50B_sparse(directed-3)_22-03-2023_07_57'],

                'Sparse': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse(directed-3)_29-03-2023_18_40'],

                'Sparse-clustered': ['conns/exp1/big/P2PAgent_100A_100E_50B_sparse_clusters(directed-2)_26-03-2023_16_48'],
            }
        },
    }
    side_by_side(viz,
                 fig_size=(10*2, 8),
                 n_rows=2,
                 axis_lim=[
                    {'y': [0, 12], 'step': 1},
                    {'y': [0, 13], 'step': 1},
                    {'y': [0, 16], 'step': 1},
                    {'y': [0, 17], 'step': 1},
                    {'y': [0, 14], 'step': 1},
                    {'y': [0, 14], 'step': 1},
                    {'y': [0, 18], 'step': 1},
                    {'y': [0, 18], 'step': 1},
                 ])
    for vk, vv in viz.items():
        print(vk)
        for k, v in vv['viz'].items():
            t = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1]
            accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'], agg_fn=np.array)[2]
            avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in accs]))
            print('\t-', k, round(max(t), 2), "Avg-Agent-Max", round(float(avg_agents_max), 2))


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

    plot_graph(mx=np.array(read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10')))  # AUCCCR
    plot_graph(mx=np.array(read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34')))  # AUCCCR clusters

    plot_graph(mx=np.array(read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_08_46')))  # Accuracy connections


if __name__ == '__main__':
    exp_1_stackoverflow()
