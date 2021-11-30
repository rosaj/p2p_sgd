from plot.visualize import side_by_side, show, plot_graph


def plot_experiment_1():
    side_by_side(
        {
            'non-BN model': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                        'experiment_1/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                        'experiment_1/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                    ],
                    'GoSGD': [
                        'experiment_1/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_10_02',
                        'experiment_1/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_17_07',
                        'experiment_1/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_20_05',
                    ],
                    'SGP': [
                        'experiment_1/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                        'experiment_1/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                        'experiment_1/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                    ],
                },
            },
            'BN model': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
                        'experiment_1/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
                        'experiment_1/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
                    ],
                    'GoSGD': [
                        'experiment_1/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_11_46',
                        'experiment_1/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_16_07',
                        'experiment_1/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_08',
                    ],
                    'SGP': [
                        'experiment_1/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
                        'experiment_1/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
                        'experiment_1/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
                    ],
                },
            },
        }
    )


def plot_experiment_1_graph():
    plot_graph(
        {
            'Undirected ring': 'experiment_1/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_22-11-2021_01_30',
        }
    )


def plot_experiment_2():
    side_by_side({
        'Fixed sparse (undirected)': {
            'x_axis': 'epoch',
            'viz': {
                'P2P': [
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
                ],
                'D$^2$': [
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',
                ],
                'GoSGD': [
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
                ],
                'SGP': [
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                       ]
            }
        },
        'Fixed sparse (directed)': {
            'x_axis': 'epoch',
            'viz': {
                'P2P': [
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                       ]
            }
        },
    })


def plot_experiment_2_graph():
    plot_graph(
        {
            'Undirected sparse': 'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
            'Directed sparse': 'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
        }
    )


if __name__ == '__main__':
    plot_experiment_2()
