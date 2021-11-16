import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import json

LABELS = {
    "epoch": "Epoch",
    "comms": "# of Communications",
    "acomms": "# of Agent Communications",
    "examples": "Examples",
    "round": "Round"
}


def read_json(filename):
    with open('log/' + filename + '.json', "r") as infile:
        f_json = json.loads(infile.read())
    return f_json


def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # '%.2f%s'
    return '%i%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def parse_timeline(name, filename, x_axis='Examples', agg_fn=np.average):
    data = read_json(filename)
    acc, v_acc, t_acc = [], [], []
    if filename.startswith('p2p'):
        mk = 'shared' if 'gru' in name else 'shared'
        rounds = list(range(len(data[list(data.keys())[0]]['train_' + mk])))
        total_examples = sum([data[a_key]['train_len'] for a_key in list(data.keys())])
        x_time = [] if x_axis != 'Round' else rounds
        for rind in rounds:
            val = [data[a_id]['val_' + mk][rind] for a_id in data.keys()]
            test = [data[a_id]['test_' + mk][rind] for a_id in data.keys()]
            # v_acc.append(agg_fn(val) * 100)
            t_acc.append(agg_fn(test) * 100)
            acc.append(agg_fn(np.average([val, test], axis=0)) * 100)

            examples = sum([data[a_key]['examples'][rind] for a_key in list(data.keys())])
            if x_axis == 'epoch':
                x_time.append(round(examples / total_examples))
            elif x_axis == 'examples':
                x_time.append(examples)
            elif x_axis == 'comms':
                v = filename.split('_')
                x_time.append(int(v[1].replace('A', '')) * int(v[2].replace('N', '')) * rind)
            elif x_axis == 'acomms':
                v = filename.split('_')
                x_time.append(int(v[2].replace('N', '')) * rind)
    else:
        x_time = [] if x_axis != 'Round' else [int(k) for k in data.keys()]
        for dk, dv in data.items():
            # v_acc.append(agg_fn(dv['Valid']) * 100)
            t_acc.append(agg_fn(dv['Test']) * 100)

            acc.append(agg_fn(np.average([dv['Valid'], dv['Test']], axis=0)) * 100)
            if x_axis == 'epoch':
                x_time.append(dv['Epoch'])
            elif x_axis == 'examples':
                x_time.append(dv['Examples'])
            elif x_axis == 'comms':
                x_time.append(int(filename.split('_')[2].replace('TR', '')) * 2 * int(dk))

    return x_time, t_acc


def show(viz_dict, x_axises=tuple(['comms']), agg_fn=np.average):
    fig, axs = plt.subplots(1, len(x_axises))
    if len(x_axises) < 2:
        axs = [axs]
    legend = []
    for ax, x_axis in zip(axs, x_axises):
        x_axis2 = None
        if isinstance(x_axis, list):
            x_axis, x_axis2 = x_axis

        for k, v in viz_dict.items():
            x_time, t_acc = parse_timeline(k, v, x_axis, agg_fn)
            ax.plot(x_time, t_acc)
            legend.append(k)
            if x_axis2 is not None:
                x_time2, t_acc2 = parse_timeline(k, v, x_axis2, agg_fn)
                xmin, xmax = x_time[0], x_time[-1]

                def x_lim_fn(x):
                    left_pct = (xmin - x[0]) / xmax
                    right_pct = (x[1] - xmax) / xmax
                    left = x_time2[0] - left_pct * x_time2[-1]
                    right = right_pct * x_time2[-1] + x_time2[-1]
                    return [left, right]

                ax2 = ax.secondary_xaxis('top', functions=(x_lim_fn, lambda x: x))
                ax2.xaxis.set_major_formatter(FuncFormatter(human_format))
                ax2.set_xlabel(LABELS[x_axis2])

        ax.set_xlabel(LABELS[x_axis])
        # ax.set_title('Plot')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.set_ylabel('Test accuracy (%)')
        ax.grid()
        ax.legend(legend, loc='lower right')
        # ax.yaxis.set_major_locator(MultipleLocator())
    # plt.savefig('/Users/robert/Desktop/acomms.svg', format='svg', dpi=1200)
    plt.show()


if __name__ == '__main__':
    show({

        'P2P (A=100, N=2)': 'p2p_100A_2N_101E_average_50B_-1CDS_4Vb_1Vc copy',
        # 'FL   (A=100, S=10)': 'fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples',

        'P2P (A=500, N=2)': 'p2p_500A_2N_101E_average_50B_-1CDS_4Vb_1Vc',
        # 'FL   (A=500, S=10)': 'fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples',

        # 'FL   (A=1000, S=10)': 'fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples',
        'P2P (A=1000, N=2)': 'p2p_1000A_2N_101E_average_50B_-1CDS_4Vb_1Vc',


        # 'P2P_100_2_V4': 'p2p_100A_2N_100E_average_50B_-1CDS_4Vb_1Vc',
        # 'P2P_100_2_V4 (2)': 'p2p_100A_2N_101E_average_50B_-1CDS_4Vb_1Vc copy',
        # 'P2P_100_2_V4-Mom_Exp': 'p2p_100A_2N_101E_average_50B_-1CDS_4Vb_1Vc',
        # 'P2P_100_2_V4-Mom_Exp2': 'p2p_100A_2N_102E_average_50B_-1CDS_4Vb_1Vc',
        # 'P2P_500_2_V4': 'p2p_500A_2N_100E_average_50B_-1CDS_4Vb_1Vc',
        # 'P2P_500_2_V4-2': 'p2p_500A_2N_101E_average_50B_-1CDS_4Vb_1Vc',
        # 'P2P_1000_2_V4-101E': 'p2p_1000A_2N_101E_average_50B_-1CDS_4Vb_1Vc',
        # 'P2P_1000_2_V4': 'p2p_1000A_2N_100E_average_50B_-1CDS_4Vb_1Vc',
        # 'FL_100-10_V2-exam': 'fl_100C_10TR_2V(0_005S-0_005C)_50E_num_examples',
        # 'FL_500-10_V2-exam': 'fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples',
        # 'FL_1000-10_V2-exam': 'fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples',


        # 'FL_500-10_V4-exam': 'fl_500C_10TR_4V(0_005S-0_005C)_100E_num_examples',
        # 'FL_500-10_V2(001)-exam-gru': 'fl_500C_10TR_2V(0_001S-0_001C)_100E_num_examples',
        # 'FL_500-10_V2-unif': 'fl_500C_10TR_2V(0_005S-0_005C)_100E_uniform',

        # 'FL_500-10_V2-exam-gru': 'fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples',
        # 'FL_500-10_V2(001)-unif-gru': 'fl_500C_10TR_2V(0_001S-0_001C)_100E_uniform',
        # 'FL_500-10_V2(001-005)-exam-gru': 'fl_500C_10TR_2V(0_001S-0_005C)_100E_num_examples',

        # 'FL_500-50_V2-unif-gru': 'fl_500C_50TR_2V(0_005S-0_005C)_100E_uniform',
        # 'FL_500-50_V2(001)-unif-gru': 'fl_500C_50TR_2V(0_001S-0_001C)_100E_uniform',
        # 'FL_500-50_V2(001)-exam-gru': 'fl_500C_50TR_2V(0_001S-0_001C)_100E_num_examples',


        # 'FL_500-100_V2-unif-gru': 'fl_500C_100TR_2V(0_005S-0_005C)_100E_uniform',
        # 'FL_500-300_V2-unif-gru': 'fl_500C_300TR_2V(0_005S-0_005C)_100E_uniform'
         },
         x_axises=[['epoch', 'comms'], 'acomms'],
         # 'round', 'examples', 'epoch', 'comms', 'acomms'
    )

