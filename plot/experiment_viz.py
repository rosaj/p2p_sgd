from plot.visualize import side_by_side, show


def plot_experiment_1():
    side_by_side(
        {
            'non-BN model': {
                'x_axis': 'epoch',
                'viz': {
                    'GoSGD': 'experiment_1/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-11-2021_02_46',
                },
            },
            'BN model': {
                'x_axis': 'epoch',
                'viz': {
                    'GoSGD': 'experiment_1/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-11-2021_11_19',
                },
            },
        }
    )


if __name__ == '__main__':
    plot_experiment_1()
