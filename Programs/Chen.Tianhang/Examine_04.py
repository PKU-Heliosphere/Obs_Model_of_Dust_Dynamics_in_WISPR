'''
This script is to visualize the PSP impact rate through WISPR streak-storm cases.
'''

import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.patches as patch
import pandas as pd
from plotly import graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt


orbit_flag = 7
my_path = '/INPUT_FILE_FOLDER_PATH/Orbit0' + str(orbit_flag) + '_WISPR_Impact_Region_statistics/impact_rate.txt'
perihelion_07 = datetime.strptime('2021-01-17/17:40', '%Y-%m-%d/%H:%M')
perihelion_02 = datetime.strptime('2019-04-04/22:39', '%Y-%m-%d/%H:%M')
perihelion_08 = datetime.strptime('2021-04-29/08:48', '%Y-%m-%d/%H:%M')
perihelion = perihelion_07
orbit07_perihelion_height = 0.09*1.496e11
orbit07_aphelion_height = np.sqrt(39869303.1361**2 + 115212032.1**2 + 7813761.028**2) * 1e3
orbit07_semimajor_axis = (orbit07_aphelion_height + orbit07_perihelion_height) / 2
orbit07_eccentric = (orbit07_aphelion_height - orbit07_perihelion_height) / \
                    (orbit07_aphelion_height + orbit07_perihelion_height)
# parallel_path = 'INPUT_FIEL_FOLDER_PATH\all_cases_not_on_PSP.txt'


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
    all_num = np.zeros([4, case_num], dtype=float)
    total_num = np.zeros([case_num], dtype=int)
    for fun_i in range(case_num):
        temp_str = all_str[fun_i].split(' ')
        temp_num = temp_str[2].split('+')
        time_stamp.append(datetime.strptime(temp_str[0], time_format)-perihelion)
        impact_rate[fun_i] = float(temp_str[1]) / eval(temp_str[3]) * 3600
        total_num[fun_i] = int(temp_str[4])
        all_num[0, fun_i] = int(temp_num[0]) / total_num[fun_i]
        all_num[1, fun_i] = int(temp_num[1]) / total_num[fun_i]
        all_num[2, fun_i] = int(temp_num[2]) / total_num[fun_i]
        all_num[3, fun_i] = int(temp_num[3]) / total_num[fun_i]
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


def main_function():
    my_datetime, my_rate, all_num, my_exptime = read_data(my_path)
    my_datetime_num = []
    para_datetime = read_parallel(parallel_path)
    for fun_i in range(len(my_datetime)):
        my_datetime_num.append((my_datetime[fun_i]).total_seconds()/3600/24)
    my_fig = plt.figure()
    ax_1 = my_fig.add_subplot(3, 1, 1)
    ax_2 = my_fig.add_subplot(3, 1, 2)
    ax_3 = my_fig.add_subplot(3, 1, 3)
    ax_3.plot(my_datetime_num, my_rate, '-x', c='#E3170D')
    ax_2.plot(my_datetime_num, my_exptime, '-')
    ax_3.set_ylim(0, my_rate.max()*10/9)
    my_delta_1 = timedelta(days=15).total_seconds()/3600/24
    my_delta_2 = timedelta(days=10).total_seconds()/3600/24
    my_delta_3 = timedelta(days=5).total_seconds()/3600/24
    ax_3.set_xlim(-my_delta_1, my_delta_1)
    ax_3.set_xticks([-my_delta_1, -my_delta_2, -my_delta_3,
                     0, my_delta_3, my_delta_2, my_delta_1])
    ax_3.set_xticklabels(['-15', '-10', '-5', '0', '5', '10', '15'])
    ax_3.set_xlabel('Days after perihelion', fontsize=15)
    ax_3.set_ylabel('impact rate $[counts / hour]$', fontsize=15)
    ax_1.tick_params(labelsize=15)
    ax_3.tick_params(labelsize=15)
    ax_2.tick_params(labelsize=15)
    ax_1.set_title('Orbit_0'+str(orbit_flag)+' Perihelion'+'2021-01-17/17:40', fontsize=20)
    ax_3.plot([0, 0], [0, my_rate.max()*10/9], '--', c='black')

    ax_2.set_xlim(-my_delta_1, my_delta_1)
    ax_2.set_ylim(0, my_exptime.max())
    ax_2.set_xticks([-my_delta_1, -my_delta_2, -my_delta_3,
                     0, my_delta_3, my_delta_2, my_delta_1])
    ax_2.set_xticklabels(['', '', '', '', '', '', ''])
    ax_2.set_ylabel('exposure $[second]$', fontsize=15)
    ax_2.plot([0, 0], [0, my_exptime.max()], '--', c='black')
    ax_1.set_xlim(-my_delta_1, my_delta_1)
    ax_1.set_xticklabels(['', '', '', '', '', '', ''])
    ax_1.set_xticks([-my_delta_1, -my_delta_2, -my_delta_3,
                     0, my_delta_3, my_delta_2, my_delta_1])
    ax_1.set_ylabel('Normalized Counts', fontsize=15)
    ax_1.set_ylim(0, 1.1)
    ax_1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_1_width = []
    for fun_i in range(len(my_datetime_num)):
        if fun_i is 0:
            ax_1_width.append(1)
        elif fun_i is 1:
            ax_1_width.append(my_datetime_num[fun_i+1] - my_datetime_num[fun_i])
        elif fun_i < len(my_datetime_num) - 1:
            ax_1_width.append((my_datetime_num[fun_i+1]/2+my_datetime_num[fun_i]/2) -
                              (my_datetime_num[fun_i]/2+my_datetime_num[fun_i-1]/2))
        else:
            ax_1_width.append(my_datetime_num[fun_i] - my_datetime_num[fun_i-1])
    print(ax_1_width)
    for fun_i in range(len(my_datetime_num)):
        if fun_i is 0:
            temp_rec = patch.Rectangle((my_datetime_num[fun_i]-ax_1_width[fun_i]/2, 0),
                                       width=ax_1_width[fun_i], height=all_num[0, fun_i], color='tab:green',
                                       label='Radial-Pattern Streaks: Converged point cannot be on S/C')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, all_num[0, fun_i]),
                                       ax_1_width[fun_i], all_num[1, fun_i], color='tab:red',
                                       label='Radial-Pattern Streaks: Converged point can be on S/C')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, all_num[0, fun_i]+all_num[1, fun_i]),
                                       ax_1_width[fun_i], all_num[2, fun_i], color='tab:blue',
                                       label='Non-Radial-Pattern Streaks: Streaks parallel to each other')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((my_datetime_num[fun_i] - ax_1_width[fun_i] / 2, all_num[0, fun_i]+all_num[1, fun_i]+all_num[2, fun_i]),
                                       ax_1_width[fun_i], all_num[3, fun_i], color='tab:grey',
                                       label='Non-Radial-Pattern Streaks: Streaks distributed randomly')
            ax_1.add_patch(temp_rec)
        elif fun_i is 1:
            temp_rec = patch.Rectangle((3/2*my_datetime_num[fun_i] - 1/2*my_datetime_num[fun_i+1], 0),
                                       width=ax_1_width[fun_i], height=all_num[0, fun_i], color='tab:green')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((3/2*my_datetime_num[fun_i] - 1/2*my_datetime_num[fun_i+1], all_num[0, fun_i]),
                                       ax_1_width[fun_i], all_num[1, fun_i], color='tab:red')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (3/2*my_datetime_num[fun_i] - 1/2*my_datetime_num[fun_i+1], all_num[0, fun_i] + all_num[1, fun_i]),
                ax_1_width[fun_i], all_num[2, fun_i], color='tab:blue')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((3/2*my_datetime_num[fun_i] - 1/2*my_datetime_num[fun_i+1],
                                        all_num[0, fun_i] + all_num[1, fun_i] + all_num[2, fun_i]),
                                       ax_1_width[fun_i], all_num[3, fun_i], color='tab:grey')
            ax_1.add_patch(temp_rec)
        else:
            temp_rec = patch.Rectangle((1/2*my_datetime_num[fun_i]+1/2*my_datetime_num[fun_i-1], 0),
                                       width=ax_1_width[fun_i], height=all_num[0, fun_i], color='tab:green')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (1/2*my_datetime_num[fun_i]+1/2*my_datetime_num[fun_i-1], all_num[0, fun_i]),
                ax_1_width[fun_i], all_num[1, fun_i], color='tab:red')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle(
                (1/2*my_datetime_num[fun_i]+1/2*my_datetime_num[fun_i-1],
                 all_num[0, fun_i] + all_num[1, fun_i]),
                ax_1_width[fun_i], all_num[2, fun_i], color='tab:blue')
            ax_1.add_patch(temp_rec)
            temp_rec = patch.Rectangle((1/2*my_datetime_num[fun_i]+1/2*my_datetime_num[fun_i-1],
                                        all_num[0, fun_i] + all_num[1, fun_i] + all_num[2, fun_i]),
                                       ax_1_width[fun_i], all_num[3, fun_i], color='tab:grey')
            ax_1.add_patch(temp_rec)
    ax_1.legend(fontsize=10, loc='upper right')
    # ax_1.plot(my_datetime_num, all_num[0], color='tab:green')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0], color='tab:red')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0]+all_num[2], color='tab:blue')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0]+all_num[2]+all_num[3], color='tab:grey')

    plt.show()


main_function()
