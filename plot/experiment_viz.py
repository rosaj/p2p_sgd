from plot.visualize import side_by_side, show


def plot_experiment_1():
    side_by_side(
        {
            'non-BN model': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': 'experiment_1/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_22-11-2021_01_30',
                    'GoSGD': 'experiment_1/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB1_TV-1_21-11-2021_02_46',
                    'SGP': 'experiment_1/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_22-11-2021_22_46',
                },
            },
            'BN model': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': 'experiment_1/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_22-11-2021_11_57',
                    'GoSGD': 'experiment_1/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB1_TV-1_21-11-2021_11_19',
                    'SGP': 'experiment_1/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_23-11-2021_11_11',
                },
            },
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
                    ],
                'GoSGD': [
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
                    ],
                'SGP': [
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
                    ]
            }
        },
        'Individual': {
            'x_axis': 'epoch',
            'viz': {
                'P2P1': 'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
                'P2P2': 'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
                'P2P3': 'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
                'GoSGD1': 'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
                'GoSGD2': 'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
                'SGP': 'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
            }
        },
    })


if __name__ == '__main__':
    plot_experiment_2()
