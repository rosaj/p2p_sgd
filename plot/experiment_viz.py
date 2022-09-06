from plot.visualize import side_by_side, show, plot_graph, resolve_timeline, parse_timeline, read_json
from scipy.stats import mannwhitneyu, ttest_ind
import numpy as np

ALG_NAME = 'P2P-BN'


def plot_graphs():
    plot_graph(
        {
            'Ring (undirected)': 'graphs/P2PAgent_10A_0E_50B_1V_ring(undirected)_N10_NB2_TV-1_04-01-2022_16_11',
            'Fully connected (undirected)': 'graphs/P2PAgent_10A_0E_50B_1V_complete(undirected)_N10_NB9_TV-1_04-01-2022_16_13',
            'Sparse (undirected)': 'graphs/P2PAgent_10A_0E_50B_1V_sparse(undirected)_N10_NB3_TV-1_04-01-2022_17_06',

            'Ring (directed)': 'graphs/P2PAgent_10A_0E_50B_1V_ring(directed)_N10_NB1_TV-1_04-01-2022_17_17',
            'Fully connected (directed)': 'graphs/P2PAgent_10A_0E_50B_1V_complete(directed)_N10_NB9_TV-1_04-01-2022_17_18',
            'Sparse (directed)': 'graphs/P2PAgent_10A_0E_50B_1V_sparse(directed)_N10_NB3_TV-1_04-01-2022_17_19',
        }, n_rows=2
    )


def plot_experiment_1():
    colors = ['r', 'g', 'b', 'indigo']
    side_by_side(
        {
            'Ring undirected (non-BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_00',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_49',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_03_22',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',
                    ],
                    'D$^2$': [
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',
                    ],
                    'SGP': [
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',
                    ],
                },
            },
            'Ring undirected (BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_54',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_12_08',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_42',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_12',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_14',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_12_40',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_05',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                    ],
                    'D$^2$': [
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_01_09',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_21_25',
                    ],
                    'SGP': [
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_27',
                    ],
                },
            },
            'Ring directed (non-BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_17_58',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_15',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_48',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_20_21',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_19_08',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',
                    ],
                    'D$^2$': [
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',
                    ],
                    'SGP': [
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',
                    ],
                },
            },
            'Ring directed (BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_07',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_32',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_54',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_14',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_20',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_58',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_05_32',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_16_53',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_09',
                    ],
                    'D$^2$': [
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_50',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_19',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_31',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_14_03',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_48',
                    ],
                    'SGP': [
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_46',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_36',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_02',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_39',
                    ],
                },
            },
        }, fig_size=(12 / 2, 12 / 3.75 * 2), n_rows=2)


def plot_experiment_1_graph():
    plot_graph(
        {
            'Undirected ring': 'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
            'Directed ring': 'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
        }
    )


def plot_experiment_2():
    colors = ['r', 'g', 'b', 'indigo']
    side_by_side({
        'Fixed sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_12_58',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_17_52',
                ],
                'GoSGD': [
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_00_26',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_20_50',
                ],
                'D$^2$': [
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_53',
                ],
                'SGP': [
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_47',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                #  ]
            }
        },
        'Fixed sparse (directed)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_13-03-2022_00_54',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-03-2022_23_26',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                ],
                'D$^2$': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                ],
                'SGP': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                # ]
            }
        },
        'Varying sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_12_09',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_16_55',
                ],
                'GoSGD': [
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_19_59',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_23_44',
                ],
                'D$^2$': [
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_20_50',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_00',
                ],
                'SGP': [
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_21_12',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_15',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                #  ]
            }
        },
        'Varying sparse (directed)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_10_28',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_13_51',
                ],
                'GoSGD': [
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_18_05',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_22_10',
                ],
                'D$^2$': [
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_30',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_02_32',

                ],
                'SGP': [
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_27',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_03_12',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                #  ]
            }
        },
    }, fig_size=(12 / 2, 12 / 3.75 * 2), n_rows=2)


def plot_experiment_1_and_2():
    colors = ['r', 'g', 'b', 'indigo']
    side_by_side(
        {
            'Ring undirected (non-BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_00',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_49',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_03_22',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',
                    ],
                    'D$^2$': [
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',
                    ],
                    'SGP': [
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',
                    ],
                },
            },
            'Ring undirected (BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_54',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_12_08',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_42',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_12',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_14',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_12_40',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_05',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                    ],
                    'D$^2$': [
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_01_09',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_21_25',
                    ],
                    'SGP': [
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_27',
                    ],
                },
            },
            'Ring directed (non-BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_17_58',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_15',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_48',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_20_21',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_19_08',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',
                    ],
                    'D$^2$': [
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',
                    ],
                    'SGP': [
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',
                    ],
                },
            },
            'Ring directed (BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_07',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_32',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_54',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_14',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_20',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_58',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_05_32',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_16_53',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_09',
                    ],
                    'D$^2$': [
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_50',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_19',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_31',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_14_03',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_48',
                    ],
                    'SGP': [
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_46',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_36',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_02',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_39',
                    ],
                },
            },
            'Fixed sparse (undirected)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
                        'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
                        'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
                        'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_12_58',
                        'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_17_52',
                    ],
                    'GoSGD': [
                        'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
                        'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
                        'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
                        'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_00_26',
                        'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_20_50',
                    ],
                    'D$^2$': [
                        'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
                        'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
                        'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',
                        'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                        'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_53',
                    ],
                    'SGP': [
                        'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
                        'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
                        'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',
                        'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                        'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_47',
                    ],
                    # 'FL': [
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                    #  ]
                }
            },
            'Fixed sparse (directed)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                        'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                        'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                        'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_13-03-2022_00_54',
                        'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-03-2022_23_26',
                    ],
                    'GoSGD': [
                        'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                        'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                        'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                        'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                        'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                    ],
                    'D$^2$': [
                        'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                        'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                        'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                        'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                        'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                    ],
                    'SGP': [
                        'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                        'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                        'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                        'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                        'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                    ],
                    # 'FL': [
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                    # ]
                }
            },
            'Varying sparse (undirected)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
                        'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
                        'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',
                        'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_12_09',
                        'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_16_55',
                    ],
                    'GoSGD': [
                        'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
                        'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
                        'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',
                        'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_19_59',
                        'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_23_44',
                    ],
                    'D$^2$': [
                        'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
                        'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
                        'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',
                        'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_20_50',
                        'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_00',
                    ],
                    'SGP': [
                        'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
                        'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
                        'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',
                        'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_21_12',
                        'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_15',
                    ],
                    # 'FL': [
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                    #  ]
                }
            },
            'Varying sparse (directed)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
                        'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
                        'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',
                        'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_10_28',
                        'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_13_51',
                    ],
                    'GoSGD': [
                        'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
                        'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
                        'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',
                        'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_18_05',
                        'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_22_10',
                    ],
                    'D$^2$': [
                        'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
                        'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
                        'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',
                        'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_30',
                        'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_02_32',

                    ],
                    'SGP': [
                        'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
                        'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
                        'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',
                        'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_27',
                        'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_03_12',
                    ],
                    # 'FL': [
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    #    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                    #  ]
                }
            },
        }, fig_size=(12 / 2, 12 / 3.75 * 4), n_rows=4)


def exp_1_avg_msg_acc():
    viz = {
        'D$^2$': [
            # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
            # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
            # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
            # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
            # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',

            'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
            'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
            'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
            'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_01_09',
            'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_21_25',

            # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
            # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
            # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
            # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
            # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',

            'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_50',
            'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_19',
            'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_31',
            'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_14_03',
            'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_48',

        ],
        'GoSGD': [
            # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
            # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
            # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
            # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
            # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',

            'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_12',
            'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_14',
            'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_12_40',
            'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_05',
            'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',

            # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
            # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
            # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
            # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
            # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',

            'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_20',
            'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_58',
            'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_05_32',
            'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_16_53',
            'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_09',
        ],
        'SGP': [
            # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
            # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
            # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
            # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
            # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',

            'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
            'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
            'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
            'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
            'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_27',

            # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
            # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
            # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
            # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
            # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',

            'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_46',
            'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_36',
            'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_02',
            'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_39',
        ],
        ALG_NAME: [
            # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_00',
            # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_16',
            # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_49',
            # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
            # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_03_22',

            'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
            'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_54',
            'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_12_08',
            'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_42',
            'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',

            # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_17_58',
            # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_15',
            # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_48',
            # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_20_21',
            # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_19_08',

            'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
            'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_07',
            'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_32',
            'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_54',
            'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_14',
        ],
    }

    print("Average number of communications per each epoch")
    for k, v in viz.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))


def exp_2_avg_msg_acc():
    viz = {
        ALG_NAME: [
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_12_58',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_17_52',

            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_13-03-2022_00_54',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-03-2022_23_26',

            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_12_09',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_16_55',

            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_10_28',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_13_51',
        ],
        'D$^2$': [
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_53',

            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',

            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_20_50',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_00',

            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_30',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_02_32',

        ],
        'GoSGD': [
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_00_26',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_20_50',

            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',

            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_19_59',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_23_44',

            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_18_05',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_22_10',

        ],
        'SGP': [
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_47',

            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',

            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_21_12',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_15',

            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_27',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_03_12',

        ],
        'FL': [
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
        ]
    }

    print("Average number of communications per each epoch and avg top accuracy")
    for k, v in viz.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))


def plot_experiment_2_graph():
    plot_graph(
        {
            'Undirected sparse': 'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
            'Directed sparse': 'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
        }, node_size=100,
    )


def plot_experiment_3():
    colors = ['red', 'coral', 'orange', 'blue', 'cornflowerblue', 'g', 'purple', 'limegreen', 'violet']
    viz = {
        '': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME + ' (1000)': [
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_29-12-2021_09_51',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_26-12-2021_05_47',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_23_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_10_46',
                ],
                ALG_NAME + ' (500)': [
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_16_01',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_16-12-2021_11_21',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_07_45',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_13_52',
                ],
                ALG_NAME + ' (100)': [
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_05',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_22',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_45',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_17_11',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_18_27',
                ],
                'GoSGD (500)': [
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_18-06-2022_03_34',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_20-06-2022_01_42',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_13-06-2022_15_01',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-06-2022_05_17',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_21-06-2022_10_02',
                ],
                'GoSGD (100)': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                ],
                'D$^2$ (500)': [
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_17-06-2022_23_24',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_25-06-2022_18_34',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_23-06-2022_03_19',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_03-07-2022_03_36',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_13_45',
                ],
                'SGP (500)': [
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_25-06-2022_23_49',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_14_17',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_16_30',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_05-07-2022_05_11',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_04-07-2022_18_36',
                ],
                'D$^2$ (100)': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                ],
                'SGP (100)': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                ]
            }
        },
    }

    side_by_side(viz, fig_size=(5, 5))
    viz = viz['']['viz']
    # 5 points
    accs = {}
    for k, v in viz.items():
        accs[k] = []
        for avg_acc in [parse_timeline(None, val)[1] for val in v]:
            avg_acc = np.array(avg_acc)[20:]
            avg_max = np.max(avg_acc)
            accs[k].append(avg_max)
        accs[k] = np.array(accs[k]).flatten()

    avg_accs = {}
    for k, v in viz.items():
        _, acc, _ = parse_timeline(None, v)
        avg_accs[k] = acc

    for name in [ALG_NAME, 'GoSGD', 'D$^2$', 'SGP']:
        print(name + ' (100 <-> 500) Max100:', max(avg_accs[name + ' (100)']), 'Max500:',
              max(avg_accs[name + ' (500)']), 'TTEST:', ttest_ind(accs[name + ' (100)'], accs[name + ' (500)']))


def exp_bn_ft():
    side_by_side({
        'Sparse (directed)': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (300)': [
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_01_48',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_04_53',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_07_17',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_48',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_00',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_53',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_10_56',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_04_15',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_23',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_40',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_12_07',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_18_02',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_19_20',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_21_05',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_20_24',
                ],
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_02_41',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_01',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_24',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_22',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_31',
                ],
            }
        },
        'Sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (300)': [
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_35',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_19_17',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_43',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_01_25',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_02_49',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_15',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_54',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_22-01-2022_00_46',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_11',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_09',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_07_21',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_06_24',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_04_12',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_20_34',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_21_18',
                ],
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_16_50',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_07',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_33',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_16',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_20',
                ],
            }
        },
    }, fig_size=(7, 7 / 2))


def bn_ft_significance():
    viz = {
        ALG_NAME + ' (100 - no BN FT) - directed': [
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_02_41',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_01',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_24',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_22',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_31',
        ],
        ALG_NAME + ' (300 - no BN FT) - directed': [
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_12_07',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_18_02',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_19_20',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_21_05',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_20_24',
        ],
        ALG_NAME + ' (100 - no BN FT) - undirected': [
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_16_50',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_07',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_33',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_16',
            'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_20',
        ],
        ALG_NAME + ' (300 - no BN FT) - undirected': [
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_07_21',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_06_24',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_04_12',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_20_34',
            'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_21_18',

        ],
        ALG_NAME + ' (100) - directed': [
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_53',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_10_56',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_04_15',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_23',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_40',
        ],
        ALG_NAME + ' (300) - directed': [
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_01_48',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_04_53',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_07_17',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_48',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_00',
        ],
        ALG_NAME + ' (100) - undirected': [
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_15',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_54',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_22-01-2022_00_46',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_11',
            'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_09',
        ],
        ALG_NAME + ' (300) - undirected': [
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_35',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_19_17',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_43',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_01_25',
            'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_02_49',
        ],
    }
    """
    accs = {}
    for k, v in viz.items():
        accs[k] = []
        for a_acc in [parse_timeline(None, val, agg_fn=lambda x: np.array(x))[1] for val in v]:
            a_acc = np.array(a_acc)[20:]
            a_max = np.max(a_acc, axis=0)
            accs[k].append(a_max)
        accs[k] = np.array(accs[k]).flatten()

    half_len = int(len(accs)/2)
    names = list(accs.keys())
    for i in range(half_len):
        print(names[i], "<->", names[i+half_len], ttest_ind(accs[names[i]], accs[names[i+half_len]]))
    """
    # 5 points
    accs = {}
    for k, v in viz.items():
        accs[k] = []
        for avg_acc in [parse_timeline(None, val)[1] for val in v]:
            avg_acc = np.array(avg_acc)[20:]
            avg_max = np.max(avg_acc)
            accs[k].append(avg_max)
        accs[k] = np.array(accs[k]).flatten()

    half_len = int(len(accs) / 2)
    names = list(accs.keys())
    for i in range(half_len):
        print(names[i], "<->", names[i + half_len], ttest_ind(accs[names[i]], accs[names[i + half_len]]))

    """
    accs = []
    for k, v in viz.items():
        _, acc, _ = parse_timeline(None, v)
        accs.append(acc)

    half_len = int(len(accs)/2)
    names = list(viz.keys())
    for i in range(half_len):
        print(names[i], "<->", names[i+half_len], mannwhitneyu(accs[i][20:], accs[i+half_len][20:]))

    print('-------------------')
    print('Directed')
    dir_ind = [0, 1, 4, 5]
    for i in dir_ind:
        for j in dir_ind:
            print(names[i], "<->", names[j], mannwhitneyu(accs[i][20:], accs[j][20:]))

    print('-------------------')
    print('Undirected')
    undir_ind = [2, 3, 6, 7]
    for i in undir_ind:
        for j in undir_ind:
            print(names[i], "<->", names[j], mannwhitneyu(accs[i][20:], accs[j][20:]))

    print("Accuracy gain")
    for i in range(len(names)):
        print(names[i], max(accs[i]))
    """


def independent_acc():
    viz = {
        'agents': [
            'experiment_2/indp/P2PAgent_100A_100E_50B_4V_21-02-2022_21_20',
            'experiment_2/indp/P2PAgent_100A_100E_50B_4V_22-02-2022_00_01',
            'experiment_2/indp/P2PAgent_100A_100E_50B_4V_22-02-2022_03_13',
            'experiment_2/indp/P2PAgent_100A_100E_50B_4V_15-03-2022_02_21',
            'experiment_2/indp/P2PAgent_100A_100E_50B_4V_15-03-2022_03_14',
        ],
        'single': [
            'experiment_2/indp/Single_100A_100E_50B_4V_22-02-2022_11_44',
            'experiment_2/indp/Single_100A_100E_50B_4V_22-02-2022_12_26',
            'experiment_2/indp/Single_100A_100E_50B_4V_22-02-2022_13_17',
            'experiment_2/indp/Single_100A_100E_50B_4V_15-03-2022_00_42',
            'experiment_2/indp/Single_100A_100E_50B_4V_14-03-2022_23_38',
        ]
    }

    for k, v in viz.items():
        max_acc = []
        for filename in v:
            data = read_json(filename)['agents']
            if isinstance(data, list):
                data = np.array(data)
                m_ac = [max(data[:, i]) for i in range(data.shape[1])]
            else:
                m_ac = [max(data[key]['test_model-accuracy_no_oov']) for key in data.keys()]
            max_acc.append(m_ac)
        print(k, "{:.3%}".format(np.average(max_acc)))


def review_exp_ring():
    side_by_side({
        'MNIST-IID (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_08-07-2022_03_44',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_13-07-2022_12_35',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_13-07-2022_12_20',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_08-07-2022_06_08',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_13-07-2022_10_55',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_13-07-2022_11_09'
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_08-07-2022_03_47',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_03_09',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_03_17',
                ],
            },
        },
        'MNIST-Pathological (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_12-07-2022_10_23',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_14-07-2022_07_27',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_14-07-2022_07_20',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_12-07-2022_07_05',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_05_59',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_05_35',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_12-07-2022_10_23',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_08_02',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_07_42',
                ],
            },
        },
        'MNIST-Practical (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_12-07-2022_03_58',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_15-07-2022_01_31',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_15-07-2022_01_49',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_12-07-2022_10_02',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_23_51',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_23_45',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_12-07-2022_04_29',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_15-07-2022_02_26',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_15-07-2022_02_33',
                ],
            },
        },
        'Reddit (Unirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                    'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                    'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                    # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',
                    # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
                ],
                'GoSGD': [
                    'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                    'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                    'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                    # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
                    # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',
                ],
                'SGP': [
                    'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                    'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                    'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                    # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',
                    # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
                ],
            },
        },
        'MNIST-IID (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_08-07-2022_04_03',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_13-07-2022_12_23',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_13-07-2022_12_22',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_08-07-2022_05_43',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_13-07-2022_10_45',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_13-07-2022_11_12',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_08-07-2022_05_10',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_14-07-2022_03_04',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_14-07-2022_03_03',
                ],
            },
        },
        'MNIST-Pathological (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_12-07-2022_10_22',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_16-07-2022_08_53',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_16-07-2022_09_14',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_12-07-2022_07_04',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_16-07-2022_07_17',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_16-07-2022_06_58',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_12-07-2022_09_57',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_16-07-2022_08_24',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_16-07-2022_08_17',
                ],
            },
        },
        'MNIST-Practical (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_12-07-2022_03_46',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_16-07-2022_05_07',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_16-07-2022_05_03',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_12-07-2022_09_29',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_15-07-2022_02_07',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_15-07-2022_03_36',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_12-07-2022_08_19',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_15-07-2022_02_28',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_15-07-2022_02_06',
                ],
            },
        },
        'Reddit (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                    'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                    'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                    # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
                    # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',
                ],
                'GoSGD': [
                    'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                    'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                    'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                    # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',
                    # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
                ],
                'SGP': [
                    'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                    'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                    'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                    # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',
                    # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
                ],
            },
        }
    }, n_rows=2, fig_size=(15, 7), axis_lim=[{'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]},
                                             {'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]}])


def review_exp_sparse():
    side_by_side({
        'MNIST-IID (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_09-07-2022_03_38',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_18',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_11',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_08-07-2022_10_21',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_07_35',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_07_46',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_08-07-2022_10_40',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_49',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_22',
                ],
                # 'GoSGD': 'rev/GoSGDAgent_100A_100E_32B_random(directed-1)_08-07-2022_06_12',
            },
        },
        'MNIST-Pathological (Unirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_12-07-2022_13_15',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_16-07-2022_05_33',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_16-07-2022_05_30',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_12-07-2022_12_34',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_03_44',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_03_44',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_12-07-2022_17_44',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_04_59',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_04_57',
                ],
            },
        },
        'MNIST-Practical (Unirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_12-07-2022_09_07',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_17-07-2022_03_23',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_17-07-2022_03_31',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_12-07-2022_07_59',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_17-07-2022_01_31',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_17-07-2022_01_23',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_12-07-2022_08_51',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_17-07-2022_02_37',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_17-07-2022_02_36',
                ],
            },
        },
        'Reddit (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_08_01',
                    'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_05_42',
                    'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_12_07',
                    # 'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_09_18',
                    # 'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_07',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_17-07-2022_23_07',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_17-07-2022_22_19',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_01_01',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_09',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_12_53',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_02_10',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_26',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_02_15',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_45',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_17-07-2022_20_02',
                ],
            }
        },
        'MNIST-IID (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_09-07-2022_03_18',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_59',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_45',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_14',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_51',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_41',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_34',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_38',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_03_41',
                ],
                # 'GoSGD': 'rev/GoSGDAgent_100A_100E_32B_random(directed-1)_08-07-2022_06_12',
            },
        },
        'MNIST-Pathological (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_12-07-2022_19_19',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_18-07-2022_18_00',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_19-07-2022_07_28',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_12-07-2022_12_38',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_16_48',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_16_43',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_16-07-2022_04_30',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_17_19',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_17_55',
                ],
            },
        },
        'MNIST-Practical (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_12-07-2022_08_34',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_18-07-2022_18_24',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_18-07-2022_18_14',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_12-07-2022_08_02',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_16_36',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_16_22',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_12-07-2022_08_56',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_17_18',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_18-07-2022_17_21',
                ],
            },
        },
        'Reddit (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_50B_sparse(directed-3)_18-07-2022_17_58',
                    'rev/D2Agent_100A_100E_50B_sparse(directed-3)_18-07-2022_17_13',
                    'rev/D2Agent_100A_100E_50B_sparse(directed-3)_18-07-2022_20_59',
                    # 'rev/D2Agent_100A_100E_50B_sparse(directed-3)_19-07-2022_16_03',
                    # 'rev/D2Agent_100A_100E_50B_sparse(directed-3)_19-07-2022_01_54',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_08_24',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_07_29',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_11_22',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_09_08',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_06_27',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_06_47',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_06_21',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_10_05',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_09_10',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_05_24',
                ],
            }
        },
    }, n_rows=2, fig_size=(15, 7), axis_lim=[{'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]},
                                             {'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]}])


if __name__ == '__main__':
    review_exp_sparse()
