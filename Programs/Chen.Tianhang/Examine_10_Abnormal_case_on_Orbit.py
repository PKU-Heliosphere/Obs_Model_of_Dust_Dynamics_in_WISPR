"""
This script is used to visualise all cases that the converged points of streaks in WISPR are not on SpaceCraft on the
orbit of PSP.
"""
import spiceypy as spice
import numpy as np
import matplotlib.patches as patch
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import copy
import os
from datetime import datetime
from datetime import timedelta
import sklearn.cluster as cluster
import scipy.optimize as opt

import furnsh_all_kernels      # all spice kernels needed are described in this file.

INPUT_path = 'D://Desktop/Impact_Rate/Orbit07_WISPR_Impact_Region_statistics/all_impact_origin.txt'
AU = 1.496e8  # unit: km


def datetime64_to_datetime(datetime64_array):
    """
    :param datetime64_array: must be 1d numpy datetime64 array.
    :return: datetime array.
    """
    datetime_array = []
    ref_datetime64 = np.datetime64('1970-01-01T08:00:00')
    for the_time in datetime64_array:
        the_timestamp = (the_time - ref_datetime64) / np.timedelta64(1, 's')
        datetime_array.append(datetime.fromtimestamp(the_timestamp))
    return datetime_array


def cal_pos_psp(spice_epoch):
    psp_pos_hci = spice.spkpos('spp', spice_epoch, 'SPP_HCI', 'NONE', 'SUN')  # unit: km
    target_pos = np.array(psp_pos_hci[0], dtype=np.float64) / AU
    return target_pos


def read_case_pos(input_file_path):
    """
    :param input_file_path:
    :return:
    """
    fun_file = open(input_file_path, 'r')
    impact_pos_str = fun_file.read().splitlines()
    num_pos = len(impact_pos_str)
    case_pos = np.zeros([num_pos, 3], dtype=np.float64)   # in HCI frame, unit: AU
    all_time = []
    is_not_on_sc = [False]*num_pos
    fun_file.close()
    for fun_i in range(num_pos):
        temp_str = impact_pos_str[fun_i].split(', ')
        temp_len = len(temp_str)
        if temp_str[0][0] == 'w':
            continue
        if temp_len == 6:
            is_not_on_sc[fun_i] = True
        all_time.append(datetime.strptime(temp_str[3], '%Y-%m-%dT%H:%M:%S.%f'))
    fun_file.close()
    for fun_i in range(num_pos):
        case_pos[fun_i] = cal_pos_psp(spice.datetime2et(all_time[fun_i]))

    return case_pos, is_not_on_sc


def get_orbit_pos(time_start, time_end, t_resolution):
    """
    :param time_start: a string like 2021-01-01T00:00:00 showing the start time of orbit
    :param time_end: ...
    :param t_resolution: unit: hour
    :return:
    """
    datetime64_array = np.arange(time_start, time_end, np.timedelta64(t_resolution, 'h'), dtype=np.datetime64)
    datetime_array = datetime64_to_datetime(datetime64_array)
    total_num = datetime64_array.size
    pos_array_psp = np.zeros([total_num, 3], dtype=np.float64)
    for time_index in range(total_num):
        this_epoch = spice.datetime2et(datetime_array[time_index])
        pos_array_psp[time_index] = cal_pos_psp(this_epoch)
    return pos_array_psp


def fun_main():
    time_start = '2021-01-01T00:00:00'
    time_end = '2021-02-01T00:00:00'
    time_resolution = 4   # unit: hr
    orbit07_pos = get_orbit_pos(time_start, time_end, time_resolution)
    case_pos, is_not_no_sc = read_case_pos(INPUT_path)
    sun_pos = np.zeros([3], dtype=np.float64)
    fig_1 = plt.figure(figsize=(7, 7))
    ax = fig_1.add_subplot(111)
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_xlim(-0.43, 0.23)
    ax.set_ylim(-0.55, 0.15)
    ax.scatter(0, 0, s=60, marker='o', color='orange', label='Sun')
    ax.plot(orbit07_pos[:, 0], orbit07_pos[:, 1], color='black')
    case_num = len(is_not_no_sc)
    legend_flag = [False, False]
    for a_case in range(case_num):
        if is_not_no_sc[a_case]:
            if not legend_flag[0]:
                ax.scatter(case_pos[a_case, 0], case_pos[a_case, 1],
                           marker='o', color='tab:red', label='Converged point not on S/C')
                legend_flag[0] = True
            else:
                ax.scatter(case_pos[a_case, 0], case_pos[a_case, 1],
                           marker='o', color='tab:red')
        else:
            if not legend_flag[1]:
                ax.scatter(case_pos[a_case, 0], case_pos[a_case, 1],
                           marker='o', color='tab:blue', label='Converged point on S/C', alpha=0.5)
                legend_flag[1] = True
            else:
                ax.scatter(case_pos[a_case, 0], case_pos[a_case, 1],
                           marker='o', color='tab:blue', alpha=0.5)
    plt.legend()
    plt.show()


fun_main()

