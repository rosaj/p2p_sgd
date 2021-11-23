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
                'x_axis': 'comms',
                'viz': {
                    'D$^2$': 'experiment_1/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_22-11-2021_11_57',
                    'GoSGD': 'experiment_1/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB1_TV-1_21-11-2021_11_19',
                    'SGP': 'experiment_1/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_23-11-2021_11_11',
                },
            },
        }
    )


if __name__ == '__main__':
    plot_experiment_1()
