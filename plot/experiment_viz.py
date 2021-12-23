from plot.visualize import side_by_side, show, plot_graph, resolve_timeline


def plot_experiment_1():
    side_by_side(
        {
            'Ring undirected (non-BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                    ],
                    'SGP': [
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                    ],
                    'P2P': [
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_00',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_49',
                    ],
                },
            },
            'Ring undirected (BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_12',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_14',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_12_40',
                    ],
                    'SGP': [
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
                    ],
                    'P2P': [
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_54',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_12_08',
                    ],
                },
            },
            'Ring directed (non-BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                    ],
                    'SGP': [
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                    ],
                    'P2P': [
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_17_58',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_15',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_48',
                    ],
                },
            },
            'Ring directed (BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_50',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_19',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_31',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_20',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_58',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_05_32',
                    ],
                    'SGP': [
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_46',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_36',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_33',
                    ],
                    'P2P': [
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_07',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_32',
                    ],
                },
            },
        }, fig_size=(15, 4))


def plot_experiment_1_graph():
    plot_graph(
        {
            'Undirected ring': 'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
            'Directed ring': 'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
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
                'D$^2$': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                ],
                'SGP': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                ]
            }
        },
        'Varying sparse (undirected)': {
            'x_axis': 'epoch',
            'viz': {
                'P2P': [
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',
                ],
                'D$^2$': [
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',
                ],
                'GoSGD': [
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',
                ],
                'SGP': [
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                ]
            }
        },
        'Varying sparse (directed)': {
            'x_axis': 'epoch',
            'viz': {
                'P2P': [
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',

                ],
                'D$^2$': [
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',

                ],
                'GoSGD': [
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',
                ],
                'SGP': [
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                ]
            }
        },
    }, fig_size=(15, 4))


def exp_2_avg_msg_count():
    viz = {
        'P2P': [
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',

            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',

            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',

            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',
        ],
        'D$^2$': [
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',

            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',

            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',

            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',

        ],
        'GoSGD': [
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',

            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',

            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',

            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',

        ],
        'SGP': [
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',

            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',

            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',

            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',

        ],
        'FL': [
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
        ]
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


def plot_experiment_2_graph():
    plot_graph(
        {
            'Undirected sparse': 'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
            'Directed sparse': 'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
        }
    )


def plot_experiment_3():
    side_by_side({
        'Results': {
            'x_axis': 'epoch',
            'viz': {
                'P2P (100)': [
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                ],
                'P2P (500)': [
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_16_01',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_16-12-2021_11_21',
                ],
                'P2P (1000)': [
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
                ],
                'FL (100)': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                ],
                'FL (500)': [
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_06-12-2021_12_24',
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_06-12-2021_17_21',
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_06-12-2021_22_20',
                ],
                'FL (1000)': [
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_09-12-2021_18_57',
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_10-12-2021_05_17',
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_10-12-2021_16_09',
                ]
            }
        },
        'Per-agent messages': {
            'x_axis': 'acomms',
            'viz': {
                'P2P (100)': [
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                ],
                'P2P (500)': [
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_16_01',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_16-12-2021_11_21',
                ],
                'P2P (1000)': [
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
                ]
            }
        }
    })


if __name__ == '__main__':
    plot_experiment_3()
