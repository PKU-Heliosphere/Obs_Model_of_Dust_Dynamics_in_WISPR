import spiceypy as spice
import pyvista as pv
import numpy as np
import matplotlib.patches as patch
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.ticker import MultipleLocator,FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import copy
import os
from datetime import datetime
from datetime import timedelta
import sklearn.cluster as cluster
import cv2
import scipy.optimize as opt
from spacepy.pycdf import CDF

import furnsh_all_kernels      # all spice kernels needed are described in this file.
from Examine_07_impact_rate import my_path, get_kepler_threshold, read_data

# try
psp_3d_model_path = 'D://Microsoft Download/Formal Files/data file/3d_model/ParkerSolarProbe/stl/ParkerSolarProbe.stl'
output_path = 'D://Desktop/PSP_Projected_Area/'
ref_file_path = 'D://Desktop/PSP_Projected_Area/ref.png'
param_file_path = 'D://Desktop/alpha_param.txt'
M_s = 1.989e30  # unit: kg
G = 6.674e-11  # unit: N*m^2/kg^2
mu_s = 6.67e-11*M_s
AU = 1.496e8  # unit: km
R_s = 6.955e5  # unit: km
peri = 20.3 * 6.963e8
orbit_a = (mu_s / 4 / np.pi**2 * (112.5*24*3600)**2) ** (1/3)  # unit: m
orbit_eccentric = (orbit_a - peri) / orbit_a
r_kepler = orbit_a * (1 - orbit_eccentric ** 2)
perihelion_time_orb07 = '2021-01-17T17:40:00'
peri_datetime = datetime.strptime(perihelion_time_orb07, '%Y-%m-%dT%H:%M:%S')


def read_file(file_path):
    file = open(file_path, 'r')
    all_str = file.read().splitlines()
    total_number = len(all_str)
    area = np.zeros((2, total_number-1), dtype=float)
    rela_speed = np.zeros_like(area, dtype=float)
    helio_dis = np.zeros(total_number-1, dtype=float)
    for fun_j in range(1, total_number):
        temp_str = all_str[fun_j].split('\t')
        area[0, fun_j-1] = eval(temp_str[0])
        area[1, fun_j-1] = eval(temp_str[1])
        rela_speed[0, fun_j-1] = eval(temp_str[2])
        rela_speed[1, fun_j-1] = eval(temp_str[3])
        helio_dis[fun_j-1] = eval(temp_str[4])
    file.close()
    return area, rela_speed, helio_dis


def projection(f_point, original_vec_sc):
    """
    :param f_point: 3D Coordinates of all points that need projecting in S/C frame system. (unit: m)(n*3 ndarray)
    :param original_vec_sc: The velocity unit vector of the stream in S/C frame. (3 ndarray)
    :return: 2D Coordinates of all points that have been projected in a plane vertical to 'original_vec_sc'.
    """
    if (np.sum(original_vec_sc**2) - 1) >= 1e-3:
        print('The vector input in funtion \'projection()\' is not a unit one!')
        return
    f_point_ref = np.zeros_like(f_point)
    num_point = f_point.shape[0]
    project_distance = np.zeros(num_point, dtype=float)
    output_point = np.zeros((num_point, 2), dtype=float)
    f_point_ref[0] = f_point[0] + original_vec_sc * 10
    project_distance = - (original_vec_sc[0] * (f_point[:, 0] - f_point_ref[0, 0]) +
                          original_vec_sc[1] * (f_point[:, 1] - f_point_ref[0, 1]) +
                          original_vec_sc[2] * (f_point[:, 2] - f_point_ref[0, 2])) / (np.sum(original_vec_sc**2))
    for fun_i in range(1, num_point):
        f_point_ref[fun_i, :] = f_point[fun_i, :] + original_vec_sc * project_distance[fun_i]

    base_vec = np.zeros((2, 3), dtype=float)
    base_vec[0] = (f_point_ref[1] - f_point_ref[0]) / np.sqrt(np.sum((f_point_ref[1] - f_point_ref[0])**2))
    base_vec[1] = np.cross(original_vec_sc, base_vec[0])
    for fun_i in range(1, num_point):
        if np.abs(base_vec[1, 0]) < 1e-3 and np.abs(base_vec[1, 1]) < 1e-3:
            output_point[fun_i, 1] = (f_point_ref[fun_i, 2] - f_point_ref[0, 2]) / base_vec[1, 2]
            if np.abs(base_vec[0, 0] < 1e-3):
                output_point[fun_i, 0] = (f_point_ref[fun_i, 1] - f_point_ref[0, 1]) / base_vec[0, 1]
            else:
                output_point[fun_i, 0] = (f_point_ref[fun_i, 0] - f_point_ref[0, 0]) / base_vec[0, 0]
        elif np.abs(base_vec[0, 0]) < 1e-3 and np.abs(base_vec[0, 1]) < 1e-3:
            output_point[fun_i, 1] = (f_point_ref[fun_i, 2] - f_point_ref[0, 2]) / base_vec[0, 2]
            if np.abs(base_vec[1, 0] < 1e-3):
                output_point[fun_i, 0] = (f_point_ref[fun_i, 1] - f_point_ref[0, 1]) / base_vec[1, 1]
            else:
                output_point[fun_i, 0] = (f_point_ref[fun_i, 0] - f_point_ref[0, 0]) / base_vec[1, 0]
        else:
            output_point[fun_i, 0] = ((f_point_ref[fun_i, 0] - f_point_ref[0, 0]) * base_vec[1, 1] -
                                      (f_point_ref[fun_i, 1] - f_point_ref[0, 1]) * base_vec[1, 0])\
                                    / (base_vec[0, 0] * base_vec[1, 1] - base_vec[0, 1] * base_vec[1, 0])
            output_point[fun_i, 1] = ((f_point_ref[fun_i, 0] - f_point_ref[0, 0]) * base_vec[0, 1] -
                                      (f_point_ref[fun_i, 1] - f_point_ref[0, 1]) * base_vec[0, 0]) \
                                     / (base_vec[1, 0] * base_vec[0, 1] - base_vec[1, 1] * base_vec[0, 0])
    return output_point


def read_point(scale=2.25 / 0.219372 / 2):
    """
    :param scale: A float, 10 by default. Scaling of the spacecraft.
    :return: A trace (go.Mesh3d()) for plotly.
    Note that the z axis in stl file is the true -z axis, x axis is the true y axis and y axis is the true -x axis!
    ——Tianhang Chen, 2022.8.10
    """
    mesh = pv.read(psp_3d_model_path)
    mesh.points = scale * mesh.points / np.max(mesh.points)
    vertices = mesh.points
    np_vertices = np.array([vertices[:, 1], -vertices[:, 0], -vertices[:, 2]])
    np_vertices = np_vertices.T
    # The true coordinates of each point in SC frame
    triangles = mesh.faces.reshape(-1, 4)
    np_triangles = np.array(triangles, dtype=int)
    return np_vertices, np_triangles


def get_rela_vel(date_time, par_pattern):
    """
    :param date_time: Datetime object
    :param par_pattern: alpha, alpha-anti(, beta)
    :return: The relative velocity of particle to spacecraft in S/C frame (unit: m/s)
    and The heliocentric distance (unit: m)
    """
    par_vel_hci = np.zeros(3, dtype=float)
    epoch_time = spice.datetime2et(date_time)
    epoch_time_plus = spice.datetime2et(date_time + timedelta(seconds=1))
    psp_pos_hci = spice.spkpos('spp', epoch_time, 'SPP_HCI', 'NONE', 'SUN')  # unit: km
    psp_pos_hci_plus = spice.spkpos('spp', epoch_time_plus, 'SPP_HCI', 'NONE', 'SUN')  # unit: km
    psp_vel_hci = np.array((psp_pos_hci_plus[0] - psp_pos_hci[0]) / 1) * 1e3  # unit: m/s
    heliocentric_dis = np.sqrt(psp_pos_hci[0][0] ** 2 + psp_pos_hci[0][1] ** 2 + psp_pos_hci[0][2] ** 2)  # unit: km
    par_speed = np.sqrt(G * M_s / heliocentric_dis / 1e3)  # unit: m/s
    if par_pattern == 'alpha':
        par_vel_hci[0] = - par_speed * psp_pos_hci[0][1] / heliocentric_dis
        par_vel_hci[1] = par_speed * psp_pos_hci[0][0] / heliocentric_dis
    elif par_pattern == 'alpha-anti':
        par_vel_hci[0] = par_speed * psp_pos_hci[0][1] / heliocentric_dis
        par_vel_hci[1] = - par_speed * psp_pos_hci[0][0] / heliocentric_dis
    rela_vel_hci = par_vel_hci - psp_vel_hci
    rela_vel_hci[2] = 0
    rela_vel_sc, _ = spice.spkcpt(rela_vel_hci, 'SUN', 'SPP_HCI', epoch_time, 'SPP_SPACECRAFT',
                                  'OBSERVER', 'NONE', 'SUN')
    return np.array(rela_vel_sc[0:3]), heliocentric_dis * 1e3


def retrieve_area(fig_path, ref_fig_path):
    """
    Note that the true area of the reference circle is pi * 6^2 (unit: m^2)
    :param fig_path:
    :param ref_fig_path:
    :return:
    """
    img_sc_cv = cv2.imread(fig_path)
    img_ref_cv = cv2.imread(ref_fig_path)
    pix_x, pix_y, rgb = img_sc_cv.shape
    count_sc = 0
    count_ref = 0
    for fun_i in range(pix_x):
        for fun_j in range(pix_y):
            if img_sc_cv[fun_i, fun_j, 0] != 255:
                count_sc = count_sc + 1
            if img_ref_cv[fun_i, fun_j, 0] != 255:
                count_ref = count_ref + 1
    area = count_sc / count_ref * np.pi * 36
    del img_sc_cv, img_ref_cv, pix_x, pix_y, rgb
    return area


def output_figure(date_time, par_pattern, output_filepath_and_name):
    """
    :param date_time:
    :param par_pattern:
    :param output_filepath_and_name:
    :return: The heliocentric distance of that time (unit: m)
    """
    data_point, data_trg = read_point()
    data_vel, data_distance = get_rela_vel(date_time, par_pattern)
    data_speed = np.sqrt(np.sum(data_vel**2))
    data_point_2d = projection(data_point, data_vel / data_speed)
    my_fig = plt.figure(figsize=(9, 9))
    my_ax = my_fig.add_subplot(111)
    for i in range(len(data_trg[:, 0])):
        the_polygon = patch.Polygon([data_point_2d[data_trg[i, 1]], data_point_2d[data_trg[i, 2]],
                                     data_point_2d[data_trg[i, 3]]], alpha=1, ec=None, fc='tab:blue')
        my_ax.add_patch(the_polygon)
    my_ax.set_xlim(-6, 6)
    my_ax.set_ylim(-6, 6)
    # make axis invisible
    [my_ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]
    my_ax.set_xticks([])
    my_ax.set_yticks([])
    my_fig.savefig(output_filepath_and_name, dpi=200)
    plt.close(my_fig)
    del my_fig, my_ax, the_polygon
    return data_distance


def get_all_const(datetimes, fig_outputpath="./"):
    """
    :param datetimes:
    :param fig_outputpath:
    :return:
    """
    projectarea_pro = []
    projectarea_anti = []
    rela_speed_pro = []
    rela_speed_anti = []
    distance = []
    for the_datetime in datetimes:
        a_rela_vel_pro, the_distance = get_rela_vel(the_datetime, 'alpha')
        a_rela_vel_anti, the_distance = get_rela_vel(the_datetime, 'alpha-anti')
        rela_speed_pro.append(np.sqrt(np.sum(a_rela_vel_pro**2)))
        rela_speed_anti.append(np.sqrt(np.sum(a_rela_vel_anti ** 2)))
        output_file = fig_outputpath + 'time' + the_datetime.strftime("%Y%m%d_%H%M%S") + '_pro.png'
        the_distance = output_figure(the_datetime, 'alpha', output_file)
        projectarea_pro.append(retrieve_area(output_file, ref_file_path))
        output_file = fig_outputpath + 'time' + the_datetime.strftime("%Y%m%d_%H%M%S") + '_anti.png'
        # the_distance = output_figure(the_datetime, 'alpha-anti', output_file)
        # projectarea_anti.append(retrieve_area(output_file, ref_file_path))
        projectarea_anti.append(1)
        # print(the_day, the_distance/AU)
        distance.append(the_distance)
    projectarea_anti = np.array(projectarea_anti)
    projectarea_pro = np.array(projectarea_pro)
    rela_speed_pro = np.array(rela_speed_pro)
    rela_speed_anti = np.array(rela_speed_anti)
    distance = np.array(distance)
    return [projectarea_pro, projectarea_anti], [rela_speed_pro, rela_speed_anti], distance


def residuals(para, y, t, global_area, global_rela_speed, global_heliodis):
    return y - f(para, t, global_area, global_rela_speed, global_heliodis)


def write_file(time_array, txt_output_filepath="./", fig_output_dirpath="./"):
    file = open(txt_output_filepath, 'w')
    file.write('Area_Pro(m^2)\tArea_Anti(m^2)\tSpeed_Pro(m/s)\tSpeed_Anti(m/s)\tHelio_distance(m)\n')
    area, rela_speed, helio_dis = get_all_const(time_array, fig_outputpath=fig_output_dirpath)
    for time_index in range(len(time_array)):
        file.write('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (area[0][time_index], area[1][time_index],
                                                       rela_speed[0][time_index], rela_speed[1][time_index],
                                                       helio_dis[time_index]))
    file.close()


def f(para, index, global_area, global_rela_speed, global_heliodis):
    """
    Unit of the rate: s^-1
    :param para:
    :param index:
    :param global_area:
    :param global_rela_speed:
    :param global_heliodis:
    :return:
    """
    fun_n_0, fun_n_1 = para
    value = fun_n_0 * (AU * 1e3 / global_heliodis[index]) ** 1.3 * global_area[0, index] * \
        global_rela_speed[0, index] ** (1 + fun_n_1)
    # value = (1 - 0) * fun_n_0 * (AU * 1e3 / global_heliodis[index]) \
    #         ** 1.3 * global_area[1, index] * global_rela_speed[1, index]
    return value


def fitting_param(days_array, observing_data_array, args_path):
    """
    :param days_array:
    :param observing_data_array:
    :return:parameters: [0]: n_0, [1]: proportion
    """
    initial_params = np.array([1, 1])
    global_area, global_rela_speed, global_heliodis = read_file(args_path)
    all_params = opt.leastsq(residuals, initial_params,
                             args=(observing_data_array, days_array, global_area, global_rela_speed, global_heliodis),
                             maxfev=6000)
    return all_params[0]


def read_fields_dust(dir_path):
    """
    NOTE: dir_path should lead to a path containing sequential level 3 data (all data during an encounter is the best).
    :param dir_path:
    :return:
    """
    all_epochs = []
    all_rates = []
    all_files = os.listdir(dir_path)
    all_files.sort()
    for each_file in all_files:
        data = CDF(dir_path + '/' + each_file)
        this_epoch = data['psp_fld_l3_dust_V2_rate_epoch']
        this_rate = data['psp_fld_l3_dust_V2_rate_ucc']
        all_epochs.extend(this_epoch)
        all_rates.extend(this_rate)
        data.close()
    return all_epochs, all_rates


def main_function():
    wispr_data = read_data(my_path)
    number_time = len(wispr_data[0])
    wispr_rate = np.sum(wispr_data[2][0:4, :], axis=0)
    # wispr_rate = wispr_data[1]
    all_time = np.zeros(number_time, dtype=float)
    for time_index in range(number_time):
        all_time[time_index] = wispr_data[0][time_index].total_seconds() / 3600 / 24
    number_density = np.zeros_like(all_time)
    index_array = np.linspace(0, number_time-1, number_time, dtype=int)
    # write_file(all_time)
    impact_rate = np.zeros_like(all_time)
    params = fitting_param(index_array, wispr_rate, param_file_path)
    # fun_fig = plt.figure(figsize=(9, 9))
    # fun_ax = fun_fig.add_subplot(111)
    global_area, global_rela_speed, global_heliodis = read_file(param_file_path)
    for simu_index in range(number_time):
        impact_rate[simu_index] = params[0] * (AU * 1e3 / global_heliodis[simu_index]) ** 1.3 * \
                                  global_area[0, simu_index] * \
                                  global_rela_speed[0, simu_index] ** (1 + params[1])
        # impact_rate[simu_index] = + (1 - 0) * params[0] \
        #                           * (AU * 1e3 / global_heliodis[simu_index]) \
        #                           ** 1.3 * global_area[1, simu_index] * global_rela_speed[1, simu_index]
    # fun_ax.plot(all_time, impact_rate, color='tab:red')
    # fun_ax.plot(all_time, wispr_rate, color='tab:blue')
    # print(params)
    # plt.show()
    my_datetime, my_rate, all_num, my_exptime = read_data(my_path)
    normalized_num = np.zeros((all_num.shape[0] - 1, all_num.shape[1]), dtype=float)
    wispr_rate = np.zeros_like(normalized_num)
    for fun_i in range(all_num[0].size):
        normalized_num[:, fun_i] = all_num[0:4, fun_i] / all_num[4, fun_i]
        wispr_rate[:, fun_i] = all_num[0:4, fun_i] / my_exptime[fun_i] * 3600  # unit: counts/hour
    my_datetime_num = []
    threshold = get_kepler_threshold(my_path)
    threshold_num = np.zeros(2)
    for i in range(2):
        threshold_num[i] = threshold[i].total_seconds() / 3600 / 24
    for fun_i in range(len(my_datetime)):
        my_datetime_num.append((my_datetime[fun_i]).total_seconds() / 3600 / 24)
    my_fig = plt.figure()
    ax_1 = my_fig.add_subplot(1, 1, 1)
    ax_33 = ax_1.twinx()
    ax_33.tick_params(axis='y', color='orange', labelsize=15, labelcolor='orange')
    # ax_1.set_ylim(0, 20)
    ax_1.set_ylim(0, 1.5)
    my_delta_1 = timedelta(days=15).total_seconds() / 3600 / 24
    my_delta_2 = timedelta(days=10).total_seconds() / 3600 / 24
    my_delta_3 = timedelta(days=5).total_seconds() / 3600 / 24
    ax_1.set_xlim(-my_delta_1, my_delta_1)
    ax_1.set_ylabel('Normalized Counts', fontsize=15)
    # ax_1.set_yticks([0, 4, 8, 12, 16, 20])
    ax_1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])
    ax_1.set_xticks([-my_delta_1, -my_delta_2, -my_delta_3,
                     0, my_delta_3, my_delta_2, my_delta_1])
    ax_1.set_xticklabels(['-15', '-10', '-5', '0', '5', '10', '15'])
    ax_1.set_xlabel('Days after perihelion', fontsize=15)
    # ax_33.set_ylabel('[counts / hour]', fontsize=15, color='orange')
    ax_1.tick_params(labelsize=15)
    # ax_2.tick_params(labelsize=15)
    ax_1.set_title('Orbit 07' + ' Perihelion' + '2021-01-17/17:40', fontsize=20)
    ax_1.plot([0, 0], [0, 20], '--', c='black')
    # ax_1.plot([threshold_num[0], threshold_num[0]], [0, 20], '-.', c='purple', linewidth=2)
    # ax_1.plot([threshold_num[1], threshold_num[1]], [0, 20], '-.', c='purple', linewidth=2)
    ax_1_width = []
    for fun_i in range(len(my_datetime_num)):
        if fun_i == 0:
            ax_1_width.append(1)
        elif fun_i == 1:
            ax_1_width.append(my_datetime_num[fun_i + 1] - my_datetime_num[fun_i])
        elif fun_i < len(my_datetime_num) - 1:
            ax_1_width.append((my_datetime_num[fun_i + 1] / 2 + my_datetime_num[fun_i] / 2) -
                              (my_datetime_num[fun_i] / 2 + my_datetime_num[fun_i - 1] / 2))
        else:
            ax_1_width.append(my_datetime_num[fun_i] - my_datetime_num[fun_i - 1])
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

    ax_33.plot(all_time, impact_rate, c='orange', lw=2.5,
               label='simulated impact rate of \n'r'prograde $\alpha$-meteoroids')
    ax_33.legend(fontsize=12, loc='upper left')
    # ax_33.set_ylim(0, 1.5 * impact_rate.max())
    ax_33.set_ylim(0, 1.5)
    print(params[1]/3/0.9, params[0])
    # ax_1.plot(my_datetime_num, all_num[0], color='tab:green')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0], color='tab:red')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0]+all_num[2], color='tab:blue')
    # ax_1.plot(my_datetime_num, all_num[1]+all_num[0]+all_num[2]+all_num[3], color='tab:grey')
    # my_fig.tight_layout(h_pad=-1.1)
    # length_1 = len(all_num[4])
    # length_2 = len(impact_rate)
    # the_sum = np.sum(all_num[0:4], axis=0)
    # print(np.max(the_sum[4]), np.max(impact_rate * 3600),
    #       the_sum[(int(length_1 / 2) - 11):(int(length_1 / 2)-5)],
    #       np.min(the_sum[(int(length_1/2)-11):(int(length_1/2)-5)])/np.max(the_sum),
    #       np.min(impact_rate[(int(length_2/2)-10):(int(length_2/2)-5)])/np.max(impact_rate))
    plt.show()
# print(retrieve_area('D://Desktop/period.png', 'D://Desktop/ref.png'))


def fit_7orbits():
    """
    Edited on 2024.04.08, for EGU24 use.  --- Tianhang CHEN
    :return:
    """
    # orbit_nums = [7, 9, 10, 11, 12, 13]
    # orbit_nums = [9, 10, 11, 12, 13]
    # orbit_nums = [10, 11, 12, 13]
    orbit_nums = [13]
    all_datetimes_1d = []
    all_datetimes_2d = []
    all_rates_1d = []
    all_rates_2d = []
    all_fit_rates_2d = []
    for orbit_i in orbit_nums:
        this_output_path = "D:/Desktop/Impact_Rate/Fitting_res/Orbit_%02d/" % orbit_i
        if not os.path.exists(this_output_path):
            os.mkdir(path=this_output_path)
            os.mkdir(path=this_output_path + "Projection_Fig/")
        fun_filepath = "D:/Desktop/Impact_Rate/Orbit%02d_WISPR_Impact_Region_statistics/impact_rate.txt" % orbit_i
        my_datetime, my_rate, all_num, my_exptime = read_data(fun_filepath)
        this_rates = all_num[5] / all_num[4]
        this_number_time = len(my_datetime)
        this_index_array = np.linspace(0, this_number_time - 1, this_number_time, dtype=int)
        all_rates_1d.extend(this_rates)
        all_rates_2d.append(this_rates)
        all_datetimes_1d.extend(my_datetime)
        all_datetimes_2d.append(my_datetime)
        write_file(my_datetime, txt_output_filepath=this_output_path + "alpha_args.txt",
                   fig_output_dirpath=this_output_path + "Projection_Fig/")
        params = fitting_param(this_index_array, this_rates, this_output_path + "alpha_args.txt")
        global_area, global_rela_speed, global_heliodis = read_file(this_output_path + "alpha_args.txt")
        this_fit_rate = np.zeros(this_number_time, dtype=np.float_)
        for simu_index in range(this_number_time):
            this_fit_rate[simu_index] = params[0] * (AU * 1e3 / global_heliodis[simu_index]) ** 1.3 * \
                                        global_area[0, simu_index] * \
                                        global_rela_speed[0, simu_index] ** (1 + params[1])
        all_fit_rates_2d.append(this_fit_rate)


def plot_7orbits():
    orbit_nums = [7, 9, 10, 11, 12, 13]
    perihelions = ["2021-01-17/17:40", "2021-08-09/19:11", "2021-11-21/08:23",
                   "2022-02-25/15:38", "2022-06-01/22:51", "2022-09-06/06:04"]
    perihelions_datetime = [datetime.strptime(perihelions[i], "%Y-%m-%d/%H:%M") for i in range(len(orbit_nums))]
    all_datetimes_2d = []
    all_rates_2d = []
    all_fit_rates_2d = []
    plot_i = 0
    fig, axes = plt.subplots(len(orbit_nums), 1)
    for orbit_i in orbit_nums:
        this_output_path = "D:/Desktop/Impact_Rate/Fitting_res/Orbit_%02d/" % orbit_i
        this_fld_datapath = "D:/Microsoft Download/Formal Files/data file/CDF/FILEDS_Dust/Enc%02d/" % orbit_i
        if not os.path.exists(this_output_path):
            os.mkdir(path=this_output_path)
            os.mkdir(path=this_output_path + "Projection_Fig/")
        fun_filepath = "D:/Desktop/Impact_Rate/Orbit%02d_WISPR_Impact_Region_statistics/impact_rate.txt" % orbit_i
        # Read WISPR rates
        my_datetime, my_rate, all_num, my_exptime = read_data(fun_filepath)
        this_rates = all_num[5] / all_num[4]
        # this_rates = this_rates / np.max(this_rates)
        # Read FIELDS rates
        fld_epochs, fld_rates = read_fields_dust(this_fld_datapath)
        fld_rates = np.array(fld_rates)
        fld_rates_norm = fld_rates / np.max(fld_rates)
        fld_epochs_day = [(fld_epochs[i] - perihelions_datetime[plot_i]).total_seconds() / 3600 / 24
                          for i in range(len(fld_epochs))]
        fld_epochs_day = np.array(fld_epochs_day)
        # Fitting
        this_number_time = len(my_datetime)
        this_index_array = np.linspace(0, this_number_time - 1, this_number_time, dtype=int)
        all_rates_2d.append(this_rates)
        all_datetimes_2d.append(my_datetime)
        params = fitting_param(this_index_array, this_rates, this_output_path + "alpha_args.txt")
        global_area, global_rela_speed, global_heliodis = read_file(this_output_path + "alpha_args.txt")
        this_fit_rate = np.zeros(this_number_time, dtype=np.float_)
        for simu_index in range(this_number_time):
            this_fit_rate[simu_index] = params[0] * (AU * 1e3 / global_heliodis[simu_index]) ** 1.3 * \
                                        global_area[0, simu_index] * \
                                        global_rela_speed[0, simu_index] ** (1 + params[1])
        all_fit_rates_2d.append(this_fit_rate)
        this_plot_time = [(my_datetime[i] - perihelions_datetime[plot_i]).total_seconds() / 3600 / 24
                          for i in range(this_number_time)]
        this_plot_time = np.array(this_plot_time, dtype=np.float_)
        axes[plot_i].text(-9.5, 0.8, "Encounter %02d" % orbit_i, fontsize=16)

        ax_sta_end = []
        # Determination of each bin width
        for fun_i in range(len(this_plot_time)):
            if fun_i == 0:
                if this_plot_time[fun_i + 1] - this_plot_time[fun_i] > 1:
                    ax_sta_end.append([this_plot_time[fun_i] - 0.5, this_plot_time[fun_i] + 0.5])
                else:
                    ax_sta_end.append(
                        [this_plot_time[fun_i] - (this_plot_time[fun_i + 1] - this_plot_time[fun_i]) / 2,
                         this_plot_time[fun_i] + (this_plot_time[fun_i + 1] - this_plot_time[fun_i]) / 2])
            elif fun_i == len(this_plot_time) - 1:
                if this_plot_time[fun_i] - this_plot_time[fun_i - 1] > 1:
                    ax_sta_end.append([this_plot_time[fun_i] - 0.5, this_plot_time[fun_i] + 0.5])
                else:
                    ax_sta_end.append(
                        [this_plot_time[fun_i] - (this_plot_time[fun_i] - this_plot_time[fun_i - 1]) / 2,
                         this_plot_time[fun_i] + (this_plot_time[fun_i] - this_plot_time[fun_i - 1]) / 2])
            else:
                if this_plot_time[fun_i + 1] - this_plot_time[fun_i] > 1:
                    if this_plot_time[fun_i] - this_plot_time[fun_i - 1] > 1:
                        ax_sta_end.append([this_plot_time[fun_i] - 0.5, this_plot_time[fun_i] + 0.5])
                    else:
                        ax_sta_end.append(
                            [this_plot_time[fun_i] - (this_plot_time[fun_i] - this_plot_time[fun_i - 1]) / 2,
                             this_plot_time[fun_i] + (this_plot_time[fun_i] - this_plot_time[fun_i - 1]) / 2])
                else:
                    if this_plot_time[fun_i] - this_plot_time[fun_i - 1] > 1:
                        ax_sta_end.append(
                            [this_plot_time[fun_i] - (this_plot_time[fun_i + 1] - this_plot_time[fun_i]) / 2,
                             this_plot_time[fun_i] + (this_plot_time[fun_i + 1] - this_plot_time[fun_i]) / 2])
                    else:
                        ax_sta_end.append(
                            [this_plot_time[fun_i] - (this_plot_time[fun_i] - this_plot_time[fun_i - 1]) / 2,
                             this_plot_time[fun_i] + (this_plot_time[fun_i + 1] - this_plot_time[fun_i]) / 2])

        # Plot the bins
        for fun_i in range(len(this_plot_time)):
            if fun_i == 0 and plot_i == 0:
                temp_rec = patch.Rectangle((ax_sta_end[fun_i][0], 0),
                                           width=ax_sta_end[fun_i][1] - ax_sta_end[fun_i][0],
                                           height=this_rates[fun_i], color='tab:blue',
                                           label='WISPR Observation', alpha=0.8)
            else:
                temp_rec = patch.Rectangle((ax_sta_end[fun_i][0], 0),
                                           width=ax_sta_end[fun_i][1] - ax_sta_end[fun_i][0],
                                           height=this_rates[fun_i], color='tab:blue', alpha=0.8)
            axes[plot_i].add_patch(temp_rec)
        # Plot FIELDS observation and Fit
        if plot_i == 0:
            # axes[plot_i].scatter(fld_epochs_day, fld_rates_norm, c='red', marker='x', label='FIELDS Observaiton')
            axes[plot_i].plot(this_plot_time, this_fit_rate, c='orange', lw=2.5, label='Fit')
        else:
            # axes[plot_i].scatter(fld_epochs_day, fld_rates_norm, c='red', marker='x')
            axes[plot_i].plot(this_plot_time, this_fit_rate, c='orange', lw=2.5)
        if plot_i == 0:
            axes[plot_i].legend(fontsize=12,  bbox_to_anchor=(0.3, 1., 0.5, 0.25), ncols=2)
        axes[plot_i].tick_params(axis='x', labelsize=14)
        axes[plot_i].tick_params(axis='y', labelsize=12)
        if plot_i < len(orbit_nums) - 1:
            axes[plot_i].set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
            axes[plot_i].set_xticklabels([])
            axes[plot_i].set_yticks([0.25, 0.5, 0.75, 1])
        else:
            axes[plot_i].set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
            axes[plot_i].set_yticks([0, 0.25, 0.5, 0.75, 1])
        if plot_i == 3:
            axes[plot_i].set_ylabel("         Normalized Rate of Occurence", fontsize=18)
        axes[plot_i].set_ylim(0, 1)
        axes[plot_i].set_xlim(-10, 10)
        axes[plot_i].grid(linestyle='--', axis='x')
        wispr_dec = ((np.mean(this_rates[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) +
                      np.mean(this_rates[np.logical_and(this_plot_time > 1, this_plot_time < 2)]) -
                      2 * np.mean(this_rates[np.abs(this_plot_time) < 0.5])) /
                     (np.mean(this_rates[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) +
                      np.mean(this_rates[np.logical_and(this_plot_time > 1, this_plot_time < 2)])))
        fit_dec = ((np.mean(this_fit_rate[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) +
                    np.mean(this_fit_rate[np.logical_and(this_plot_time > 1, this_plot_time < 2)]) -
                    2 * np.mean(this_fit_rate[np.abs(this_plot_time) < 0.5])) /
                   (np.mean(this_fit_rate[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) +
                    np.mean(this_fit_rate[np.logical_and(this_plot_time > 1, this_plot_time < 2)])))
        fld_dec = (np.mean(fld_rates[np.abs(fld_epochs_day) < 0.5]) /
                   np.mean(fld_rates[np.logical_and(fld_epochs_day < -1, fld_epochs_day > -2)]))
        beta_dec = 0

        wispr_asy = (np.abs(np.mean(this_rates[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) -
                           np.mean(this_rates[np.logical_and(this_plot_time > 1, this_plot_time < 2)]))
                     / np.abs(np.mean(this_rates[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) +
                           np.mean(this_rates[np.logical_and(this_plot_time > 1, this_plot_time < 2)])))
        fit_asy = (np.abs(np.mean(this_fit_rate[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) -
                           np.mean(this_fit_rate[np.logical_and(this_plot_time > 1, this_plot_time < 2)]))
                     / np.abs(np.mean(this_fit_rate[np.logical_and(this_plot_time < -1, this_plot_time > -2)]) +
                           np.mean(this_fit_rate[np.logical_and(this_plot_time > 1, this_plot_time < 2)])))
        fld_asy = (np.abs(np.mean(fld_rates[np.logical_and(fld_epochs_day < -1, fld_epochs_day > -2)]) -
                         np.mean(fld_rates[np.logical_and(fld_epochs_day > 1, fld_epochs_day < 2)]))
                   / np.abs(np.mean(fld_rates[np.logical_and(fld_epochs_day < -1, fld_epochs_day > -2)]) +
                         np.mean(fld_rates[np.logical_and(fld_epochs_day > 1, fld_epochs_day < 2)])))
        beta_asy = 0.5
        mat_l = np.mat([[fit_dec - beta_dec], [fit_asy - beta_asy]])
        mat_r = np.mat([[wispr_dec - beta_dec], [wispr_asy - beta_asy]])
        # mat_l = np.mat([[fit_dec, beta_dec], [fit_asy, beta_asy], [1, 1]])
        # mat_r = np.mat([[wispr_dec], [wispr_asy], [1]])
        x = np.linalg.lstsq(mat_l, mat_r, rcond=None)
        print("Orbit %02d: %.3f, %.3f, %.3f, %.3f" %
              (orbit_i, wispr_dec, fit_dec, wispr_asy, fit_asy))
        print(x[0].item(0), 1 - x[0].item(0))
        plot_i = plot_i + 1
    axes[-1].set_xlabel("Days from Perihelion", fontsize=18)
    plt.subplots_adjust(hspace=0)
    plt.show()


def plot_vel():
    orbit_nums = [7, 9, 10, 11, 12, 13]
    perihelions = ["2021-01-17/17:40", "2021-08-09/19:11", "2021-11-21/08:23",
                   "2022-02-25/15:38", "2022-06-01/22:51", "2022-09-06/06:04"]
    perihelions_datetime = [datetime.strptime(perihelions[i], "%Y-%m-%d/%H:%M") for i in range(len(orbit_nums))]
    all_datetimes_2d = []
    all_rates_2d = []
    all_fit_rates_2d = []
    all_dist_2d_in = []
    all_dist_2d_out = []
    plot_i = 0
    fig, axes = plt.subplots(len(orbit_nums), 1)
    for orbit_i in orbit_nums:
        this_output_path = "D:/Desktop/Impact_Rate/Fitting_res/Orbit_%02d/" % orbit_i
        this_fld_datapath = "D:/Microsoft Download/Formal Files/data file/CDF/FILEDS_Dust/Enc%02d/" % orbit_i
        if not os.path.exists(this_output_path):
            os.mkdir(path=this_output_path)
            os.mkdir(path=this_output_path + "Projection_Fig/")
        fun_filepath = "D:/Desktop/Impact_Rate/Orbit%02d_WISPR_Impact_Region_statistics/impact_rate.txt" % orbit_i
        # Read WISPR rates
        my_datetime, my_rate, all_num, my_exptime = read_data(fun_filepath)
        this_rates = all_num[5] / all_num[4]
        this_rates[this_rates > 1] = 1
        this_number_time = len(my_datetime)
        this_index_array = np.linspace(0, this_number_time - 1, this_number_time, dtype=int)
        all_rates_2d.append(this_rates)
        all_datetimes_2d.append(my_datetime)
        params = fitting_param(this_index_array, this_rates, this_output_path + "alpha_args.txt")
        global_area, global_rela_speed, global_heliodis = read_file(this_output_path + "alpha_args.txt")
        this_fit_rate = np.zeros(this_number_time, dtype=np.float_)
        for simu_index in range(this_number_time):
            this_fit_rate[simu_index] = params[0] * (AU * 1e3 / global_heliodis[simu_index]) ** 1.3 * \
                                        global_area[0, simu_index] * \
                                        global_rela_speed[0, simu_index] ** (1 + params[1])
        all_fit_rates_2d.append(this_fit_rate)
        this_plot_time = np.array([(my_datetime[i] - perihelions_datetime[plot_i]).total_seconds() / 3600 / 24
                                  for i in range(this_number_time)])
        if plot_i == 0:
            axes[plot_i].plot(global_heliodis[this_plot_time < 0] / (R_s * 1e3), this_rates[this_plot_time < 0],
                              '--b', marker='x', label='Inbound')
            axes[plot_i].plot(global_heliodis[this_plot_time > 0] / (R_s * 1e3), this_rates[this_plot_time > 0],
                              '--r', marker='o', label='Outbound')
            axes[plot_i].legend(fontsize=12,  bbox_to_anchor=(0.2, 1.2, 0.5, 0.25), ncols=2)
        else:
            axes[plot_i].plot(global_heliodis[this_plot_time < 0] / (R_s * 1e3), this_rates[this_plot_time < 0],
                              '--b', marker='x')
            axes[plot_i].plot(global_heliodis[this_plot_time > 0] / (R_s * 1e3), this_rates[this_plot_time > 0],
                              '--r', marker='o')
        axes[plot_i].set_xlim(10, 60)
        axes[plot_i].set_xticks(np.arange(10, 65, 5))
        axes[plot_i].set_yticks([0, 0.5, 1])
        axes[plot_i].set_yticklabels(["", "0.5", "1"])
        axes[plot_i].set_ylim(0, 1.1)
        axes[plot_i].grid(linestyle='--', axis='x')
        axes[plot_i].tick_params(labelsize=13)
        axes[plot_i].text(52, 0.9, "Encounter %02d" % orbit_i, fontsize=13)
        if plot_i == 3:
            axes[plot_i].set_ylabel("         Normalized Rate of Occurence", fontsize=16)
        if plot_i < 6 - 1:
            axes[plot_i].set_xticklabels([])
        else:
            axes[plot_i].set_xlabel(r"Heliocentric Distance  [$\mathrm{R_s}$]", fontsize=14)
        plot_i = plot_i + 1
    plt.subplots_adjust(hspace=0)
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # main_function()
    # fit_7orbits()
    # plot_7orbits()
    plot_vel()
