import numpy as np
from datetime import datetime
from datetime import timedelta
import spiceypy as spice
import furnsh_all_kernels
import matplotlib.patches as patch
import pandas as pd
from plotly import graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as scio


orbit_flag = 7
AU = 1.496e11  # unit: m
solar_radius = 6.955e8  # unit: m
my_path = 'D://Desktop/Impact_Rate/Orbit0' + str(orbit_flag) + '_WISPR_Impact_Region_statistics/impact_rate.txt'
perihelion_07 = datetime.strptime('2021-01-17/17:40', '%Y-%m-%d/%H:%M')
perihelion_02 = datetime.strptime('2019-04-04/22:39', '%Y-%m-%d/%H:%M')
perihelion_08 = datetime.strptime('2021-04-29/08:48', '%Y-%m-%d/%H:%M')
perihelion = perihelion_07
orbit07_perihelion_height = 0.09*1.496e11
orbit07_aphelion_height = np.sqrt(39869303.1361**2 + 115212032.1**2 + 7813761.028**2) * 1e3
orbit07_semimajor_axis = (orbit07_aphelion_height + orbit07_perihelion_height) / 2
orbit07_eccentric = (orbit07_aphelion_height - orbit07_perihelion_height) / \
                    (orbit07_aphelion_height + orbit07_perihelion_height)
kepler_distance = orbit07_semimajor_axis * (1- orbit07_eccentric**2)
parallel_path = 'D://Desktop/Impact_Rate/Orbit07_WISPR_Impact_Region_statistics/not_on_PSP_case/' \
                'all_cases_not_on_PSP.txt'


def get_kepler_threshold(file_path, time_format='%Y-%m-%dT%H:%M:%S'):
    fun_i = 0
    file = open(file_path)
    all_str = file.read().splitlines()
    total_number = len(all_str)
    all_time = []
    temp_psp_pos = 0
    for fun_j in range(total_number):
        temp_str = all_str[fun_j].split(' ')
        all_time.append(datetime.strptime(temp_str[0], time_format))
    while fun_i < len(all_time):
        et = spice.datetime2et(all_time[fun_i])
        temp_psp_pos, light_time = spice.spkpos('SPP', et, 'SPP_HCI', 'NONE', 'SUN')
        temp_dis = np.sqrt(temp_psp_pos[0]**2 + temp_psp_pos[1]**2 + temp_psp_pos[2]**2) * 1e3
        if temp_dis < kepler_distance:
            break
        fun_i = fun_i + 1
    inbound_thre = all_time[fun_i] - perihelion
    while fun_i < len(all_time):
        et = spice.datetime2et(all_time[fun_i])
        temp_psp_pos, light_time = spice.spkpos('SPP', et, 'SPP_HCI', 'NONE', 'SUN')
        temp_dis = np.sqrt(temp_psp_pos[0]**2 + temp_psp_pos[1]**2 + temp_psp_pos[2]**2) * 1e3
        if temp_dis >= kepler_distance:
            break
        fun_i = fun_i + 1
    outbound_thre = all_time[fun_i-1] - perihelion
    return inbound_thre, outbound_thre


def read_data(file_path, time_format='%Y-%m-%dT%H:%M:%S'):
    """
    :param file_path: (str)
    :param time_format: (str) the format of time strings in the input file.
    :return: time_stamp: (1*n datetime array)
            impact_rate: (1*n ndarray) unit: counts per hour
            exp_time: (1*n ndarray) unit: second
    """
    fun_f = open(file_path)
    all_str = fun_f.read().splitlines()
    case_num = len(all_str)
    time_stamp = []
    impact_rate = np.zeros([case_num], dtype='float64')
    exp_time = np.zeros([case_num], dtype='float64')
    all_num = np.zeros([6, case_num], dtype=float)
    total_num = np.zeros([case_num], dtype=int)
    for fun_i in range(case_num):
        temp_str = all_str[fun_i].split(' ')
        temp_num = temp_str[2].split('+')
        time_stamp.append(datetime.strptime(temp_str[0], time_format))
        impact_rate[fun_i] = float(temp_str[1]) / eval(temp_str[3]) * 3600
        total_num[fun_i] = int(temp_str[4])
        all_num[0, fun_i] = int(temp_num[0]) / total_num[fun_i] * total_num[fun_i]
        all_num[1, fun_i] = int(temp_num[1]) / total_num[fun_i] * total_num[fun_i]
        all_num[2, fun_i] = int(temp_num[2]) / total_num[fun_i] * total_num[fun_i]
        all_num[3, fun_i] = int(temp_num[3]) / total_num[fun_i] * total_num[fun_i]    # Counts
        # all_num[0, fun_i] = int(temp_num[0]) / total_num[fun_i]
        # all_num[1, fun_i] = int(temp_num[1]) / total_num[fun_i]
        # all_num[2, fun_i] = int(temp_num[2]) / total_num[fun_i]
        # all_num[3, fun_i] = int(temp_num[3]) / total_num[fun_i]   # Normalized Counts
        all_num[4, fun_i] = total_num[fun_i]
        all_num[5, fun_i] = float(temp_str[1])
        exp_time[fun_i] = eval(temp_str[3])
    fun_f.close()
    return time_stamp, impact_rate, all_num, exp_time


def read_parallel(file_path, time_format='%Y-%m-%dT%H:%M:%S'):
    """
        :param file_path: (str)
        :param time_format: (str) the format of time strings in the input file.
        :return: time_stamp: (1*n datetime array)
        """
    fun_f = open(file_path)
    all_str = fun_f.read().splitlines()
    case_num = len(all_str)
    time_stamp = []
    impact_rate = np.zeros([case_num], dtype='float64')
    for fun_i in range(case_num-1):
        temp_str = all_str[fun_i].strip(' ')
        time_stamp.append(datetime.strptime(temp_str, time_format))
    fun_f.close()
    return time_stamp


def read_fields_data(file_path='D://Desktop/Impact_Rate/Orbit07_WISPR_Impact_Region_statistics/FIELDS_TDSmax_Rate/'
                               'time_and_rate.mat'):
    """
    :param file_path: The path of the .mat file
    :return:
    """
    data = scio.loadmat(file_path)
    return data['day_time_delta'][0], data['num1'][0]


def main_function():
    my_datetime, my_rate, all_num, my_exptime = read_data(my_path)
    normalized_num = np.zeros((all_num.shape[0]-1, all_num.shape[1]), dtype=float)
    wispr_rate = np.zeros_like(normalized_num)
    for fun_i in range(all_num[0].size):
        normalized_num[:, fun_i] = all_num[0:4, fun_i] / all_num[4, fun_i]
        wispr_rate[:, fun_i] = all_num[0:4, fun_i] / my_exptime[fun_i] * 3600  # unit: counts/hour
    my_datetime_num = []
    threshold = get_kepler_threshold(my_path)
    threshold_num = np.zeros(2)
    for i in range(2):
        threshold_num[i] = threshold[i].total_seconds()/3600/24
    para_datetime = read_parallel(parallel_path)
    for fun_i in range(len(my_datetime)):
        my_datetime_num.append((my_datetime[fun_i]).total_seconds()/3600/24)
    fields_data = np.array(read_fields_data())
    my_fig = plt.figure(figsize=(19, 9))
    ax_1 = my_fig.add_subplot(2, 1, 1)
    # ax_2 = my_fig.add_subplot(3, 1, 2)
    ax_3 = my_fig.add_subplot(2, 1, 2)
    ax_33 = ax_3.twinx()
    ax_33.tick_params(axis='y', color='orange', labelsize=15, labelcolor='orange')
    ax_3.set_ylim(0, 1.1)
    my_delta_1 = timedelta(days=15).total_seconds()/3600/24
    my_delta_2 = timedelta(days=10).total_seconds()/3600/24
    my_delta_3 = timedelta(days=5).total_seconds()/3600/24
    ax_3.set_xlim(-my_delta_1, my_delta_1)
    ax_3.set_ylabel('Normalized Counts', fontsize=15)
    ax_3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_3.set_xticks([-my_delta_1, -my_delta_2, -my_delta_3,
                     0, my_delta_3, my_delta_2, my_delta_1])
    ax_3.set_xticklabels(['-15', '-10', '-5', '0', '5', '10', '15'])
    ax_3.set_xlabel('Days after perihelion', fontsize=15)
    ax_33.set_ylabel('[counts / hour]', fontsize=15, color='orange')
    ax_1.tick_params(labelsize=15)
    ax_3.tick_params(labelsize=15)
    # ax_2.tick_params(labelsize=15)
    ax_1.set_title('Orbit 0'+str(orbit_flag)+' Perihelion'+'2021-01-17/17:40', fontsize=20)
    ax_3.plot([0, 0], [0, 1.2], '--', c='black')
    ax_3.plot([threshold_num[0], threshold_num[0]], [0, 1.2], '-.', c='purple', linewidth=2)
    ax_3.plot([threshold_num[1], threshold_num[1]], [0, 1.2], '-.', c='purple', linewidth=2)
    ax_1.plot([0, 0], [0, 15], '--', c='black')
    # ax_2.set_xlim(-my_delta_1, my_delta_1)
    # ax_2.set_ylim(0, my_exptime.max()*1.2/60)
    # ax_2.set_xticks([-my_delta_1, -my_delta_2, -my_delta_3,
    #                  0, my_delta_3, my_delta_2, my_delta_1])
    # ax_2.set_xticklabels(['', '', '', '', '', '', ''])
    # ax_2.set_ylabel('Exposure Time' + '\n' + '[minutes]', fontsize=15)
    # ax_2.plot([0, 0], [0, my_exptime.max()*1.2/60], '--', c='black')
    # ax_2.scatter(my_datetime_num, my_exptime/60, marker='x')
    ax_1.set_xlim(-my_delta_1, my_delta_1)
    ax_1.set_xticklabels(['', '', '', '', '', '', ''])
    ax_1.set_xticks([-my_delta_1, -my_delta_2, -my_delta_3,
                     0, my_delta_3, my_delta_2, my_delta_1])
    ax_1.set_ylabel('Counts', fontsize=15)
    ax_1.set_ylim(0, 15)
    ax_1.set_yticks([0, 3, 6, 9, 12, 15])
    ax_1.set_yticklabels(['0', '3', '6', '9', '12', '15'])
    ax_1_width = []
    for fun_i in range(len(my_datetime_num)):
        if fun_i == 0:
            ax_1_width.append(1)
        elif fun_i == 1:
            ax_1_width.append(my_datetime_num[fun_i+1] - my_datetime_num[fun_i])
        elif fun_i < len(my_datetime_num) - 1:
            ax_1_width.append((my_datetime_num[fun_i+1]/2+my_datetime_num[fun_i]/2) -
                              (my_datetime_num[fun_i]/2+my_datetime_num[fun_i-1]/2))
        else:
            ax_1_width.append(my_datetime_num[fun_i] - my_datetime_num[fun_i-1])
    for fun_i in range(len(my_datetime_num)):
        if fun_i == 0:
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, 0),
                                       width=ax_1_width[fun_i], height=all_num[0, fun_i], color='tab:green',
                                       label='Radial-Pattern Streaks: Converged point cannot be on S/C', alpha=0.8)
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, all_num[0, fun_i]),
                                       ax_1_width[fun_i], all_num[1, fun_i], color='tab:red',
                                       label='Radial-Pattern Streaks: Converged point can be on S/C', alpha=0.8)
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, all_num[0, fun_i] + all_num[1, fun_i]),
                ax_1_width[fun_i], all_num[2, fun_i], color='tab:blue',
                label='Non-Radial-Pattern Streaks: Streaks parallel to each other', alpha=0.8)
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2,
                                        all_num[0, fun_i] + all_num[1, fun_i] + all_num[2, fun_i]),
                                       ax_1_width[fun_i], all_num[3, fun_i], color='tab:grey',
                                       label='Non-Radial-Pattern Streaks: Streaks distributed randomly', alpha=0.8)
            ax_1.add_patch(temp_rec)
        elif fun_i == 1:
            temp_rec = patch.Rectangle((3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1], 0),
                                       width=ax_1_width[fun_i], height=all_num[0, fun_i], color='tab:green', alpha=0.8)
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1], all_num[0, fun_i]),
                ax_1_width[fun_i], all_num[1, fun_i], color='tab:red', alpha=0.8)
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1],
                 all_num[0, fun_i] + all_num[1, fun_i]),
                ax_1_width[fun_i], all_num[2, fun_i], color='tab:blue', alpha=0.8)
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1],
                                        all_num[0, fun_i] + all_num[1, fun_i] + all_num[2, fun_i]),
                                       ax_1_width[fun_i], all_num[3, fun_i], color='tab:grey', alpha=0.8)
            ax_1.add_patch(temp_rec)
        else:
            if all_num[0, fun_i] != 0:
                temp_rec = patch.Rectangle((1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1], 0),
                                           width=ax_1_width[fun_i], height=all_num[0, fun_i], color='tab:green')
                ax_1.add_patch(temp_rec)
            if all_num[1, fun_i] != 0:
                temp_rec = patch.Rectangle(
                    (1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1], all_num[0, fun_i]),
                    width=ax_1_width[fun_i], height=all_num[1, fun_i], color='tab:red', alpha=0.8)
                ax_1.add_patch(temp_rec)
            if all_num[2, fun_i] != 0:
                temp_rec = patch.Rectangle(
                    (1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1],
                     all_num[0, fun_i] + all_num[1, fun_i]),
                    width=ax_1_width[fun_i], height=all_num[2, fun_i], color='tab:blue', alpha=0.8)
                ax_1.add_patch(temp_rec)
            if all_num[3, fun_i] != 0:
                temp_rec = patch.Rectangle((1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1],
                                            all_num[0, fun_i] + all_num[1, fun_i] + all_num[2, fun_i]),
                                           width=ax_1_width[fun_i], height=all_num[3, fun_i], color='tab:grey',
                                           alpha=0.8
                                           )
                ax_1.add_patch(temp_rec)
    ax_1.legend(fontsize=12, loc='upper right')

    for fun_i in range(len(my_datetime_num)):
        if fun_i == 0:
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, 0),
                                       width=ax_1_width[fun_i], height=normalized_num[0, fun_i], color='tab:green'
                                       , alpha=0.8
                                       )
            ax_3.add_patch(temp_rec)
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, normalized_num[0, fun_i]),
                                       ax_1_width[fun_i], normalized_num[1, fun_i], color='tab:red', alpha=0.8)
            ax_3.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, normalized_num[0, fun_i] + normalized_num[1, fun_i]),
                ax_1_width[fun_i], normalized_num[2, fun_i], color='tab:blue', alpha=0.8)
            ax_3.add_patch(temp_rec)
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2,
                                        normalized_num[0, fun_i] + normalized_num[1, fun_i] + normalized_num[2, fun_i]),
                                       ax_1_width[fun_i], normalized_num[3, fun_i], color='tab:grey', alpha=0.8)
            ax_3.add_patch(temp_rec)
        elif fun_i == 1:
            temp_rec = patch.Rectangle((3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1], 0),
                                       width=ax_1_width[fun_i], height=normalized_num[0, fun_i], color='tab:green',
                                       alpha=0.8)
            ax_3.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1], normalized_num[0, fun_i]),
                ax_1_width[fun_i], normalized_num[1, fun_i], color='tab:red', alpha=0.8)
            ax_3.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1],
                 normalized_num[0, fun_i] + normalized_num[1, fun_i]),
                ax_1_width[fun_i], normalized_num[2, fun_i], color='tab:blue', alpha=0.8)
            ax_3.add_patch(temp_rec)
            temp_rec = patch.Rectangle((3 / 2 * my_datetime_num[fun_i] - 1 / 2 * my_datetime_num[fun_i + 1],
                                        normalized_num[0, fun_i] + normalized_num[1, fun_i] + normalized_num[2, fun_i]),
                                       ax_1_width[fun_i], normalized_num[3, fun_i], color='tab:grey', alpha=0.8)
            ax_3.add_patch(temp_rec)
        else:
            if normalized_num[0, fun_i] != 0:
                temp_rec = patch.Rectangle((1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1], 0),
                                           width=ax_1_width[fun_i], height=normalized_num[0, fun_i], color='tab:green'
                                           , alpha=0.8)
                ax_3.add_patch(temp_rec)
            if normalized_num[1, fun_i] != 0:
                temp_rec = patch.Rectangle(
                    (1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1], normalized_num[0, fun_i]),
                    width=ax_1_width[fun_i], height=normalized_num[1, fun_i], color='tab:red', alpha=0.8)
                ax_3.add_patch(temp_rec)
            if normalized_num[2, fun_i] != 0:
                temp_rec = patch.Rectangle(
                    (1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1],
                     normalized_num[0, fun_i] + normalized_num[1, fun_i]),
                    width=ax_1_width[fun_i], height=normalized_num[2, fun_i], color='tab:blue', alpha=0.8)
                ax_3.add_patch(temp_rec)
            if normalized_num[3, fun_i] != 0:
                temp_rec = patch.Rectangle((1 / 2 * my_datetime_num[fun_i] + 1 / 2 * my_datetime_num[fun_i - 1],
                                            normalized_num[0, fun_i] + normalized_num[1, fun_i] + normalized_num[
                                                2, fun_i]),
                                           width=ax_1_width[fun_i], height=normalized_num[3, fun_i], color='tab:grey'
                                           , alpha=0.8)
                ax_3.add_patch(temp_rec)

    ax_33.scatter(fields_data[0], fields_data[1], c='orange', label='Occurance Rate Calculated by TDS data')
    ax_33.legend(fontsize=12)
    ax_33.set_ylim(0, 220)
    # ax_1.plot(my_datetime_num, all_num[0], color='tab:green')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0], color='tab:red')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0]+all_num[2], color='tab:blue')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0]+all_num[2]+all_num[3], color='tab:grey')
    my_fig.tight_layout(h_pad=-1.1)
    # plt.savefig('D://Desktop/trial.eps', dpi=300)
    plt.show()


if __name__ == "__main__":
    main_function()

