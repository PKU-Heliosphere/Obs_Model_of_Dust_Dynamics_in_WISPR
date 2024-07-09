import os

from sunpy.io import read_file
import sunpy.map
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm
import copy
import pandas as pd
from datetime import datetime
import spiceypy as spice
import scipy.special as spe
import furnsh_all_kernels
import pyvista as pv
from plotly import graph_objects as go
import gc

import pyvista as pv

from Brightness_F_Corona_SCpos_senpai import obtain_VSF0


I_0 = 1361  # W/m^2 Solar Irradiance
AU = 1.49e11  # distance from sun to earth
Rs = 6.96e8  # solar radii
peri_lb = 7.9 * 1e9 / AU
peri_ub = peri_lb * 10
crit_lb = 0.15
crit_ub = 0.50


def n_r(x):
    """
    :param x: normalized radial distance. x = r / r_0
    :return:
    """
    r_in = 5 * Rs / AU
    r_out = 19 * Rs / AU
    nu = 1.3
    return x ** (-nu) * (x > r_out) + x ** (-nu) * (x - r_in) / (r_out - r_in) * np.logical_and(x <= r_out, x >= r_in)


def linear_fit(lin_analysis_l, lin_analysis_I):
    n_analysis = lin_analysis_I.size
    k_fit = (np.sum((lin_analysis_l - np.mean(lin_analysis_l)) * (lin_analysis_I - np.mean(lin_analysis_I))) /
             np.sum((lin_analysis_l - np.mean(lin_analysis_l)) ** 2))
    b_fit = np.mean(lin_analysis_I) - np.mean(lin_analysis_l) * k_fit
    corr_fit = ((np.sum(lin_analysis_l * lin_analysis_I) -
                 n_analysis * np.mean(lin_analysis_l) * np.mean(lin_analysis_I)) /
                np.sqrt((np.sum(lin_analysis_l ** 2) - n_analysis * np.mean(lin_analysis_l) ** 2) *
                        (np.sum(lin_analysis_I ** 2) - n_analysis * np.mean(lin_analysis_I) ** 2)))
    return k_fit, b_fit, corr_fit


def find_eff_length(fov_angle, SC_pos_carr, lin_crit_perc=[0.15, 0.50]):
    # Constants & Geometry
    beta_rad = fov_angle[0]  # 经度方向上的方位角
    gamma_rad = fov_angle[1]  # 纬度方向上的方位角

    cos_elongation = (1 + np.tan(beta_rad) ** 2 + np.tan(gamma_rad) ** 2) ** (-1 / 2)
    elongation_rad = np.arccos(cos_elongation)

    theta_rad = np.linspace(elongation_rad, np.pi, 180)

    x_sc_carr, y_sc_carr, z_sc_carr = SC_pos_carr[0], SC_pos_carr[1], SC_pos_carr[2]
    R_sc = np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2 + z_sc_carr ** 2)

    l_los = R_sc * np.sin(theta_rad - elongation_rad) / np.sin(theta_rad)
    radial_dist_los = R_sc * np.sin(elongation_rad) / np.sin(theta_rad)

    carr2fov_arr = np.array(
        [[-x_sc_carr * z_sc_carr / R_sc ** 2, -y_sc_carr * z_sc_carr / R_sc ** 2, 1 - z_sc_carr ** 2 / R_sc ** 2],
         [y_sc_carr / R_sc, -x_sc_carr / R_sc, 0],
         [x_sc_carr / R_sc, y_sc_carr / R_sc, z_sc_carr / R_sc]])
    carr2fov_arr = np.array(
        [[y_sc_carr / R_sc, -x_sc_carr / R_sc, 0],
         [-x_sc_carr * z_sc_carr / R_sc ** 2, -y_sc_carr * z_sc_carr / R_sc ** 2, 1 - z_sc_carr ** 2 / R_sc ** 2],
         [x_sc_carr / R_sc, y_sc_carr / R_sc, z_sc_carr / R_sc]])
    fov2carr_arr = np.array([
        [-y_sc_carr / np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2),
         x_sc_carr * z_sc_carr / R_sc / np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2), -x_sc_carr / R_sc],
        [x_sc_carr / np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2),
         y_sc_carr * z_sc_carr / R_sc / np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2), -y_sc_carr / R_sc],
        [0, -np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2) / R_sc, -z_sc_carr / R_sc]])
    if x_sc_carr == 0 and y_sc_carr == 0:
        fov2carr_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        fov2carr_arr = np.linalg.inv(carr2fov_arr)

    xp_fov = l_los * np.tan(beta_rad) * cos_elongation
    yp_fov = l_los * np.tan(gamma_rad) * cos_elongation
    zp_fov = - l_los * cos_elongation

    sc_p_carr = np.dot(fov2carr_arr, np.vstack((xp_fov, yp_fov, zp_fov)))
    # sc_p_carr = [-zp_fov, ]
    xp_carr = sc_p_carr[0] + x_sc_carr
    yp_carr = sc_p_carr[1] + y_sc_carr
    zp_carr = sc_p_carr[2] + z_sc_carr

    cos2lat = 1 - zp_carr ** 2 / (xp_carr ** 2 + yp_carr ** 2 + zp_carr ** 2)
    sin2lat = 1 - cos2lat

    # Size & Spatial Distribution of Dust
    a = np.linspace(1e-7, 1e-4, 1000)  # Dust size (m)
    rho = 2.5e6  # g/m^3
    m = (4. / 3) * np.pi * a ** 3 * rho  # g
    c1 = 2.2e3
    c2 = 15.
    c3 = 1.3e-9
    c4 = 1e11
    c5 = 1e27
    c6 = 1.3e-16
    c7 = 1e6
    g1 = 0.306
    g2 = -4.38
    g3 = 2.
    g4 = 4.
    g5 = -0.36
    g6 = 2.
    g7 = -0.85
    Fmr0 = (c1 * m ** g1 + c2) ** g2 \
           + c3 * (m + c4 * m ** g3 + c5 * m ** g4) ** g5 \
           + c6 * (m + c7 * m ** g6) ** g7  # m^-2s^-1
    v0 = 20e3  # m/s
    Nmr0 = 4. * Fmr0 / v0  # m^-3

    # VSF
    VSF0 = []
    print("TTT")
    for i in range(0, len(theta_rad)):
        VSF0.append(obtain_VSF0(a, Nmr0, theta_rad[i]))  # m^-1
    print("QQQ")
    # plt.axes(yscale = 'log')
    # plt.plot(scatter_angle,VSF0)
    # plt.xlabel('Scatter Angle')
    # plt.ylabel('VSF(r0,theta)')
    # plt.show()
    # LOS Integral
    nu = 1.3
    delta_rad = abs(np.mean(np.diff(theta_rad)))

    # I_plot = [I_0 * AU * (1 * np.sin(elongation_rad)) ** (-nu - 1) * \
    #           np.nansum(np.sin(theta_rad[:plot_i]) ** nu * (0.15 + 0.85 * cos2lat[:plot_i] ** 14) *
    #           np.array(VSF0)[:plot_i] * delta_rad) for plot_i in range(theta_rad.size)]
    I_plot = np.array([I_0 * np.nansum((radial_dist_los[:plot_i] / AU) ** (-2) *
                                       n_r(radial_dist_los[:plot_i] / AU) * R_sc * np.sin(elongation_rad) /
                                       np.sin(theta_rad[:plot_i])**2 *
                                       np.exp(-3.8 * np.sqrt(sin2lat[:plot_i]) + 2 * sin2lat[:plot_i]) *
                                       np.array(VSF0)[:plot_i] * delta_rad)
                      for plot_i in range(theta_rad.size)]) / I_0 * 4.50 / 6.61 * 1e-4
    l_los_sta = np.where(I_plot >= np.max(I_plot) * lin_crit_perc[0])[0][0]
    l_los_end = np.where(I_plot > np.max(I_plot) * lin_crit_perc[1])[0][0]

    # I = I_0 * AU * (1 * np.sin(elongation_rad)) ** (-nu - 1) * \
    #     np.nansum(np.sin(theta_rad) ** nu * (0.15 + 0.85 * cos2lat ** 14) * np.array(VSF0) * delta_rad)
    # I = I_0 * AU * (1 * np.sin(elongation_rad)) ** (-nu - 1) * \
    #     np.nansum(
    #         np.sin(theta_rad) ** nu * np.exp(-3.8 * np.sqrt(sin2lat) + 2 * sin2lat) * np.array(VSF0) * delta_rad)
    approx_pos_l = l_los[int((l_los_sta + l_los_end) / 2)]
    approx_len_l = l_los[l_los_end] - l_los[l_los_sta]
    # NOTE that the unit of l is meter !!!!!
    print(approx_pos_l / AU, approx_len_l / AU)
    return approx_pos_l, approx_len_l


def gen_database(r_range=np.array([10 / 225, 100 / 225]), num_r_arr=180,
                 lon_range=np.array([0, 90], dtype=np.float_), num_lon_arr=100,
                 lat_range=np.array([-45, 45], dtype=np.float_), num_lat_arr=100,
                 output_path="D:/Desktop/database/"):
    """
    NOTE that the z axis of camera frame is directed from PSP to the SUN. y axis is perpendicular to the ecliptic plane.
    :param r_range: Heliocentric distance. (Unit: AU)
    :param num_r_arr: Size of array
    :param lon_range: Longitude in Camera Frame (x coordinate).  Unit: deg
    :param num_lon_arr: Size of array
    :param lat_range: Latitude in Camera Frame (y coordinate).   Unit: deg
    :param num_lat_arr: Size of array
    :param output_path:
    :return:
    """

    r_arr = np.linspace(r_range[0], r_range[1], num_r_arr)  # unit: AU
    lon_arr = np.linspace(lon_range[0], lon_range[1], num_lon_arr)
    lat_arr = np.linspace(lat_range[0], lat_range[1], num_lat_arr)
    all_arrs = np.zeros((num_r_arr, num_lon_arr, num_lat_arr, 2))
    for fun_i in range(len(r_arr)):
        for fun_j in range(len(lon_arr)):
            for fun_k in range(len(lat_arr)):
                tmp_pos, tmp_len = find_eff_length([lon_arr[fun_j] * np.pi / 180, lat_arr[fun_k] * np.pi / 180],
                                                   [r_arr[fun_i] * AU,
                                                    0, 0], lin_crit_perc=[crit_lb, crit_ub])
                all_arrs[fun_i, fun_j, fun_k, 0] = tmp_pos
                all_arrs[fun_i, fun_j, fun_k, 1] = tmp_len
                print("当前进度：%d/%d，%d/%d，%d/%d" % (fun_i + 1, len(r_arr), fun_j + 1, len(lon_arr),
                      fun_k + 1, len(lat_arr)))
    np.save(output_path + 'Distance_arr', r_arr)
    np.save(output_path + 'Lon_CMR_arr', lon_arr)
    np.save(output_path + 'Lat_CMR_arr', lat_arr)
    np.save(output_path + 'LOS_database_arr', all_arrs)
    del all_arrs, r_arr, lon_arr, lat_arr
    return 0


def load_database(dir_path="D:/Desktop/database/"):
    dist_arr = np.load(dir_path + "Distance_arr.npy")
    lon_arr = np.load(dir_path + "Lon_CMR_arr.npy")
    lat_arr = np.load(dir_path + "Lat_CMR_arr.npy")
    los_arrs = np.load(dir_path + "LOS_database_arr.npy")
    return dist_arr, lon_arr, lat_arr, los_arrs


def calc_sigma_mean(size_arr, theta_rad):
    wavelength = 550e-9  # m
    alpha = 2 * np.pi * size_arr / wavelength
    albedo = 0.25  # bond albedo
    delta_alpha = np.abs(np.mean(np.diff(alpha)))
    sigma = size_arr ** 2 * np.abs(spe.jv(1, alpha * np.sin(theta_rad))) ** 2 / \
            np.abs(np.sin(theta_rad)) ** 2 + albedo * size_arr ** 2 / 4  # m^2
    rho = 2.5e6  # g/m^3
    m = (4. / 3) * np.pi * size_arr ** 3 * rho  # g
    c1 = 2.2e3
    c2 = 15.
    c3 = 1.3e-9
    c4 = 1e11
    c5 = 1e27
    c6 = 1.3e-16
    c7 = 1e6
    g1 = 0.306
    g2 = -4.38
    g3 = 2.
    g4 = 4.
    g5 = -0.36
    g6 = 2.
    g7 = -0.85
    Fmr0 = (c1 * m ** g1 + c2) ** g2 \
           + c3 * (m + c4 * m ** g3 + c5 * m ** g4) ** g5 \
           + c6 * (m + c7 * m ** g6) ** g7  # m^-2s^-1
    v0 = 20e3  # m/s
    Nmr0 = 4. * Fmr0 / v0  # m^-3
    sigma_mean = (np.nansum((sigma[1:] + sigma[:-1]) / 2 * np.abs(np.diff(Nmr0) / np.diff(alpha)) * delta_alpha) /
                  np.nansum(np.abs(np.diff(Nmr0) / np.diff(alpha)) * delta_alpha))
    return sigma_mean


def gen_sigma_database(size_range, theta_rad_range, database_dirpath="D:/Desktop"):
    sigma_arr = np.zeros_like(theta_rad_range)
    for fun_i in range(theta_rad_range.size):
        print("%.2f" % (fun_i / theta_rad_range.size))
        sigma_arr[fun_i] = calc_sigma_mean(size_range, theta_rad_range[fun_i])
    np.save(database_dirpath + "Sigma_mean", sigma_arr)
    np.save(database_dirpath + "ThetaForSigma", theta_rad_range)
    return 0


def get_density_arr(n_arr: list, fits_filename, output_file_id, database_path="D:/Desktop/database/"):
    fits_data, fits_header = sunpy.io.read_file(fits_filename)[0]
    los_database = load_database(dir_path=database_path)
    sigma_database = np.load(database_path + "Sigma_mean.npy")
    theta_for_sigma = np.load(database_path + "ThetaForSigma.npy")
    fits_datetime_str = fits_header["DATE-OBS"]
    fits_datetime = datetime.strptime(fits_datetime_str, '%Y-%m-%dT%H:%M:%S.%f')
    fits_etime = spice.datetime2et(fits_datetime)
    psp_j2000_pos, light_time = spice.spkpos('SPP', fits_etime, 'ECLIPJ2000', 'NONE', 'SUN')
    psp_j2000_pos = np.array(psp_j2000_pos) * 1e3  # unit: m
    psp_helio_r = np.sqrt(np.sum(psp_j2000_pos**2)) / AU
    fov_x_l = 1920 * 10e-6
    fov_y_l = 2048 * 10e-6
    fov_z_l = 28e-3
    tmp_fov_pos = np.zeros((3,), dtype=np.float_)
    size_range = np.linspace(1, 1e5, 100000) / 1e9  # unit: m
    for x_i in range(0, fits_data.shape[1], 10):
        for y_i in range(0, fits_data.shape[0], 10):
            if np.isnan(fits_data[y_i, x_i]) or fits_data[y_i, x_i] <= 0 or x_i < 35:
                continue
            tmp_fov_pos[0] = x_i / fits_data.shape[1] * fov_x_l - fov_x_l / 2
            tmp_fov_pos[1] = y_i / fits_data.shape[0] * fov_y_l - fov_y_l / 2
            tmp_fov_pos[2] = fov_z_l
            tmp_j2000_pos, _ = spice.spkcpt(tmp_fov_pos, 'SPP', 'SPP_WISPR_INNER', fits_etime,
                                            'ECLIPJ2000', 'OBSERVER', 'NONE', 'SPP')
            tmp_j2000_pos = np.array(tmp_j2000_pos[0:3])  # unit: m
            tmp_pos_lat = np.arcsin(tmp_j2000_pos[2] / np.sqrt(np.sum(tmp_j2000_pos ** 2))) * 180 / np.pi
            tmp_pos_lon = np.arccos(-((tmp_j2000_pos[0] * psp_j2000_pos[0] + tmp_j2000_pos[1] * psp_j2000_pos[1]) /
                                    np.sqrt(np.sum(tmp_j2000_pos[0:2]**2) * np.sum(psp_j2000_pos[0:2]**2))))*180/np.pi
            tmp_pos_elong = np.arccos(-(np.sum(tmp_j2000_pos * psp_j2000_pos) /
                                        np.sqrt(np.sum(tmp_j2000_pos**2) * np.sum(psp_j2000_pos**2)))) * 180 / np.pi
            tmp_R_index = int(((psp_helio_r - los_database[0][0]) / (los_database[0][-1] - los_database[0][0]))
                              * (los_database[0].size - 1))
            tmp_lon_index = int(((tmp_pos_lon - los_database[1][0]) / (los_database[1][-1] - los_database[1][0]))
                                * (los_database[1].size - 1))
            tmp_lat_index = int(((tmp_pos_lat - los_database[2][0]) / (los_database[2][-1] - los_database[2][0]))
                                * (los_database[2].size - 1))
            num_los_dist = 10
            for los_i in range(num_los_dist + 1):
                print("当前进度：", x_i, y_i, los_i)
                if (tmp_R_index >= los_database[3][:, 0, 0, 0].size or
                        tmp_lon_index >= los_database[3][0, :, 0, 0].size or
                        tmp_lat_index >= los_database[3][0, 0, :, 0].size):
                    continue
                this_los_dist = (los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 0] -
                                 los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 1] / 2 +
                                 los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 1] * los_i / num_los_dist)
                # this_los_dist = (los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 0] -
                #                  los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 1] / 4 +
                #                  los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 1] * los_i / 2 / num_los_dist)
                # this_los_dist = los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 0]
                theta_rad = np.arctan(np.sin(tmp_pos_elong * np.pi / 180) /
                                      (this_los_dist / psp_helio_r / AU
                                       - np.cos(tmp_pos_elong * np.pi / 180)))
                if theta_rad < 0:
                    theta_rad = theta_rad + np.pi
                tmp_los_pos = psp_j2000_pos + (tmp_j2000_pos / np.sqrt(np.sum(tmp_j2000_pos**2)) *
                                                 this_los_dist)
                # unit: m
                tmp_helio_r = np.sqrt(np.sum(tmp_los_pos**2))  # unit: m
                tmp_j2000_lat = np.arcsin(tmp_los_pos[2] / tmp_helio_r) * 180 / np.pi  # unit: deg
                tmp_j2000_lot = ((np.arccos(tmp_los_pos[0] / np.sqrt(tmp_los_pos[0]**2 + tmp_los_pos[1]**2))
                                 * 180 / np.pi) * (tmp_los_pos[1] >= 0) +
                                 (360 - np.arccos(tmp_los_pos[0] / np.sqrt(tmp_los_pos[0]**2 + tmp_los_pos[1]**2))
                                 * 180 / np.pi) * (tmp_los_pos[1] < 0))
                # sigma_mean = calc_sigma_mean(size_range, theta_rad)
                tmp_sigma_index = int((theta_rad - theta_for_sigma[0]) / (theta_for_sigma[-1] - theta_for_sigma[0]) *
                                      (theta_for_sigma.size - 1))
                # print(theta_rad / np.pi * 180, this_los_dist / psp_helio_r / AU)
                sigma_mean = sigma_database[tmp_sigma_index]
                tmp_density = (fits_data[y_i, x_i] / (4.50 / 6.61 * 1e-4) * (crit_ub - crit_lb) /
                               los_database[3][tmp_R_index, tmp_lon_index, tmp_lat_index, 1] /
                               sigma_mean * (tmp_helio_r / AU)**2)  # unit: m^-3
                # n_arr.append([tmp_helio_r / AU, tmp_j2000_lat,
                #               np.sqrt(tmp_los_pos[0]**2 + tmp_los_pos[1]**2) / AU, tmp_los_pos[2] / AU,
                #               tmp_density])
                # output_file_id.write("%.6f, %.3f, %.6f, %.6f, %.6E\n" % (tmp_helio_r / AU, tmp_j2000_lat,
                #                                                          np.sqrt(tmp_los_pos[0]**2 + tmp_los_pos[1]**2) / AU,
                #                                                          tmp_los_pos[2] / AU, tmp_density))
                output_file_id.write("%.6f, %.3f, %.3f, %.6f, %.6f, %.6f, %.6E\n"
                                     % (tmp_helio_r / AU, tmp_j2000_lat, tmp_j2000_lot,
                                        tmp_los_pos[0] / AU, tmp_los_pos[1] / AU,
                                        tmp_los_pos[2] / AU, tmp_density))
    del fits_data, sigma_mean, los_database
    gc.collect()
    return n_arr


def get_spatial_distribution(fits_dirpath: str, database_dirpath: str):
    # n_arrs = []
    # output_txt_file = open(database_dirpath + "Enc10/org_density_3d.txt", "a")
    # all_file_dirpath = os.listdir(fits_dirpath)
    # for each_dirpath in all_file_dirpath:
    #     # if each_dirpath[-2] != "1":
    #     #     continue
    #     all_files = os.listdir(fits_dirpath + each_dirpath)
    #     for file_name in all_files:
    #         if file_name[-6] != "1":
    #             continue
    #         n_arrs = get_density_arr(n_arrs, fits_dirpath + each_dirpath + "/" + file_name,
    #                                  output_txt_file,
    #                                  database_path=database_dirpath)
    ############################################################################
    num_x = 1001
    num_y = 1001
    num_z = 501
    num_lat = 361
    num_lon = 721
    plot_x = np.linspace(-0.5, 0.5, num_x)  # unit: AU
    plot_y = np.linspace(-0.5, 0.5, num_y)  # unit: AU
    plot_z = np.linspace(-0.25, 0.25, num_z)  # unit: AU
    plot_r = np.linspace(0, 0.5, num_x)  # unit: AU
    plot_lat = np.linspace(-90, 90, num_lat)  # unit: deg
    plot_lon = np.linspace(0, 360, num_lat)  # unit: deg
    counts = np.zeros((num_x, num_y, num_z), dtype=np.int_)
    densities = np.zeros((num_x, num_y, num_z), dtype=np.float_)
    counts_rhl = np.zeros((num_x, num_lat, num_lon), dtype=np.int_)
    densities_rhl = np.zeros((num_x, num_lat, num_lon), dtype=np.float_)
    line_count = 1
    with open(database_dirpath + "Enc10/org_density_3d.txt", 'r') as org_txt_file:
        while 1:
            this_line = org_txt_file.readline()
            if not this_line:
                break
            this_line = this_line.strip()
            all_num_str = this_line.split(", ")
            # x_index = int((floatn_arrs[fun_i][2] - plot_x[0]) / (plot_x[-1] - plot_x[0]) * num_x)
            # z_index = int((n_arrs[fun_i][3] - plot_z[0]) / (plot_z[-1] - plot_z[0]) * num_z)
            x_index = int((float(all_num_str[3]) - plot_x[0]) / (plot_x[-1] - plot_x[0]) * num_x)
            y_index = int((float(all_num_str[4]) - plot_y[0]) / (plot_y[-1] - plot_y[0]) * num_y)
            z_index = int((float(all_num_str[5]) - plot_z[0]) / (plot_z[-1] - plot_z[0]) * num_z)
            r_index = int((float(all_num_str[0]) - plot_x[0]) / (plot_x[-1] - plot_x[0]) * num_x)
            lat_index = int((float(all_num_str[1]) - plot_lat[0]) / (plot_lat[-1] - plot_lat[0]) * num_lat)
            lon_index = int((float(all_num_str[2]) - plot_lon[0]) / (plot_lon[-1] - plot_lon[0]) * num_lon)
            if lon_index == num_lon:
                lon_index = 0
            counts[x_index, y_index, z_index] = counts[x_index, y_index, z_index] + 1
            counts_rhl[r_index, lat_index, lon_index] = counts_rhl[r_index, lat_index, lon_index] + 1
            # densities[x_index, z_index] = densities[x_index, z_index] + n_arrs[fun_i][4]
            densities[x_index, y_index, z_index] = densities[x_index, y_index, z_index] + float(all_num_str[6])
            densities_rhl[r_index, lat_index, lon_index] = (densities_rhl[r_index, lat_index, lon_index] +
                                                            float(all_num_str[6]))
            line_count = line_count + 1
            print(float(all_num_str[3]), float(all_num_str[4]), line_count)
    densities[counts != 0] = densities[counts != 0] / counts[counts != 0]
    densities[counts == 0] = np.nan
    densities_rhl[counts_rhl != 0] = densities_rhl[counts_rhl != 0] / counts_rhl[counts_rhl != 0]
    densities_rhl[counts_rhl == 0] = np.nan
    # n_arrs = np.array(n_arrs)
    # np.save(database_dirpath + "Enc10/org_density", n_arrs)
    # n_arrs = np.load(database_dirpath + 'Enc10/org_density.npy')
    # print(n_arrs[fun_i][2]/AU, n_arrs[fun_i][3]/AU)

    return plot_x, plot_y, plot_z, densities, plot_r, plot_lat, plot_lon, densities_rhl


if __name__ == "__main__":
    database_path = "D:/Microsoft Download/Formal Files/data file/Formatted Data/NPY/WISPR_brightness_LOS_20240508/"
    # gen_database(output_path=database_path)
    # size_arr = np.linspace(1, 1e5, 100000) / 1e9  # unit: m
    # theta_rad_range = np.linspace(6, 174, 1000) / 180 * np.pi
    # gen_sigma_database(size_arr, theta_rad_range, database_path)

    # x_plot, y_plot, z_plot, density, r_plot, lat_plot, lon_plot, density_rhl = (
    #     get_spatial_distribution(fits_dirpath="D:/Microsoft Download/Formal Files/data file/FITS/"
    #                              "WISPR/L2_Background/WISPR_ENC10_L2b_FITS/",
    #                              database_dirpath=database_path))
    # np.save(database_path + "Enc10/" + "x_plot_3d", x_plot)
    # np.save(database_path + "Enc10/" + "y_plot_3d", y_plot)
    # np.save(database_path + "Enc10/" + "z_plot_3d", z_plot)
    # np.save(database_path + "Enc10/" + "num_density_3d", density)
    # np.save(database_path + "Enc10/" + "r_plot_3d", r_plot)
    # np.save(database_path + "Enc10/" + "lat_plot_3d", lat_plot)
    # np.save(database_path + "Enc10/" + "lon_plot_3d", lon_plot)
    # np.save(database_path + "Enc10/" + "num_density_rhl_3d", density_rhl)
    ########################################################################################################
    x_plot = np.load(database_path + "Enc10/" + "x_plot_3d.npy") * AU / Rs  # unit: Rs
    y_plot = np.load(database_path + "Enc10/" + "y_plot_3d.npy") * AU / Rs  # unit: Rs
    z_plot = np.load(database_path + "Enc10/" + "z_plot_3d.npy") * AU / Rs  # unit: Rs
    density = np.load(database_path + "Enc10/" + "num_density_3d.npy")
    r_plot = np.load(database_path + "Enc10/" + "r_plot_3d.npy") * AU / Rs  # unit: Rs
    lat_plot = np.load(database_path + "Enc10/" + "lat_plot_3d.npy")
    lon_plot = np.load(database_path + "Enc10/" + "lon_plot_3d.npy")
    density_rhl = np.load(database_path + "Enc10/" + "num_density_rhl_3d.npy")
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111)
    # pcm = ax.pcolor(x_plot * AU / Rs, z_plot * AU / Rs, density.T * 1e9)
    # ctr = ax.contour(x_plot * AU / Rs, z_plot * AU / Rs, density.T * 1e9, levels=10, colors='black',
    #                  alpha=0.1)
    # ax.set_xlabel(r"r  [$\mathrm{R_s}$]", fontsize=14)
    # ax.set_ylabel(r"z  [$\mathrm{R_s}$]", fontsize=14)
    # ax.axis("equal")
    # ax.set_xlim(0, 80)
    # ax.set_ylim(-30, 30)
    # c_bar = plt.colorbar(mappable=pcm)
    # c_bar.set_label(label='\n'r'$\mathrm{N_{dust}\quad [km^{-3}]}$', fontsize=14)
    # # c_bar.set_ticks(, fontsize=14)
    # ax.tick_params(labelsize=12)
    #
    # fig_2 = plt.figure(figsize=(15, 7))
    # ax_2 = fig_2.add_subplot(121)
    # for each_lat_i in range(90 - 1, lat_plot.size - 90, 30):
    #     ax_2.plot(np.log10(r_plot * AU / Rs), np.log10(density_rhl[:, each_lat_i] * 1e9), lw=2,
    #               label="%d"r"${}^\circ$" % lat_plot[each_lat_i])
    # ax_2.legend()
    # ax_2.plot(np.linspace(0, 2, 2), np.linspace(0, 2, 2) * (-1.3) + 9.5, "--", color='grey', lw=2)
    # ax_2.text(0.70, 8.2, r"$\mathrm{N\propto r^{-1.3}}$", fontsize=15, color='grey')
    # ax_2.set_xlim(0.5, 2)
    # ax_2.set_ylim(6.5, 8.5)
    # ax_2.set_xlabel(r"$\mathrm{log_{10}\,r\quad [R_s]}$", fontsize=14)
    # ax_2.set_ylabel(r"$\mathrm{log_{10}\,N_{dust}}\quad [km^{-3}]$", fontsize=14)
    # ax_2.set_title("Radial Dependence for Different Latitude", fontsize=15)
    # ax_3 = fig_2.add_subplot(122)
    # for each_r_i in range(60, 300, 50):
    #     ax_3.plot(lat_plot, np.log10(density_rhl[each_r_i, :] * 1e9), lw=2,
    #               label="%d"r" $\mathrm{R_S}$" % (r_plot[each_r_i] * AU / Rs))
    # ax_3.legend()
    # ax_3.set_xlabel(r"$\mathrm{lat\quad [deg]}$", fontsize=14)
    # ax_3.set_ylabel(r"$\mathrm{log_{10}\,N_{dust}}\quad [\mathrm{km^{-3}}]$", fontsize=14)
    # ax_3.set_title("Latitude Dependence for Different r", fontsize=15)
    # ax_2.tick_params(labelsize=12)
    # ax_3.tick_params(labelsize=12)
    # ########################################################################################################
    # max_latitude = np.zeros((r_plot.size,), dtype=np.float_)
    # max_pos = np.zeros((r_plot.size, 2), dtype=np.float_)
    # for fun_i in range(r_plot.size):
    #     if np.all(np.isnan(density_rhl[fun_i, :])):
    #         max_latitude[fun_i] = np.nan
    #         # max_pos[fun_i, :] = np.nan
    #         continue
    #     max_value = np.nanmax(density_rhl[fun_i, :])
    #     max_index = np.where(density_rhl[fun_i, :] == max_value)[0][0]
    #     max_latitude[fun_i] = lat_plot[max_index]
    #     max_pos[fun_i, 0] = r_plot[fun_i] * np.cos(max_latitude[fun_i] * np.pi / 180)
    #     max_pos[fun_i, 1] = r_plot[fun_i] * np.sin(max_latitude[fun_i] * np.pi / 180)
    #
    # fig_3 = plt.figure(figsize=(7, 7))
    # ax_4 = fig_3.add_subplot(111)
    # # ax_4.plot(r_plot * AU / Rs, max_latitude)
    # ax_4.scatter(max_pos[:, 0] * AU / Rs, max_pos[:, 1] * AU / Rs, 10,
    #              label='Position of Symmetry Plane')
    # k_fit, b_fit, corr_fit = linear_fit(max_pos[r_plot * AU / Rs < 36, 0], max_pos[r_plot * AU / Rs < 36, 1])
    # ax_4.plot(np.array([0, 50]), np.array([0, 50]) * k_fit + b_fit, 'r-', lw=2, label='Linear Fit')
    # ax_4.text(10, -5, r"$\Delta \theta$ = %.2f${}^\circ$" % (np.arctan(k_fit) * 180 / np.pi),
    #           fontsize=14, color='red')
    # ax_4.set_xlim(2, 37)
    # ax_4.set_ylim(-20, 20)
    # ax_4.set_xlabel(r"r  [$\mathrm{R_s}$]", fontsize=14)
    # ax_4.set_ylabel(r"z  [$\mathrm{R_s}$]", fontsize=14)
    # ax_4.tick_params(labelsize=12)
    # ax_4.legend(fontsize=13)
    #
    # #######################################################################################################
    # ddz_pos = np.zeros((lat_plot.size, 2), dtype=np.float_)
    # ddz_r = np.zeros((lat_plot.size, ), dtype=np.float_)
    # for fun_i in range(lat_plot.size):
    #     if np.all(np.isnan(density_rhl[:, fun_i])):
    #         ddz_r[fun_i] = np.nan
    #         continue
    #     if lat_plot[fun_i] >= 40 or lat_plot[fun_i] <= -35:
    #         ddz_r[fun_i] = np.nan
    #         continue
    #     max_value = np.nanmax(density_rhl[:, fun_i])
    #     max_index = np.where(density_rhl[:, fun_i] == max_value)[0][0]
    #     ddz_r[fun_i] = r_plot[max_index]
    #     ddz_pos[fun_i, 0] = ddz_r[fun_i] * np.cos(lat_plot[fun_i] * np.pi / 180)
    #     ddz_pos[fun_i, 1] = ddz_r[fun_i] * np.sin(lat_plot[fun_i] * np.pi / 180)
    # print(ddz_r * AU / Rs)
    # ddz_pos[ddz_pos[:, 0] == 0, 0] = np.nan
    # ddz_pos[ddz_pos[:, 1] == 0, 1] = np.nan
    # fig_4 = plt.figure(figsize=(7, 7))
    # ax_5 = fig_4.add_subplot(111)
    # ax_5.plot(ddz_pos[:, 0] * AU / Rs,
    #           ddz_pos[:, 1] * AU / Rs, label='DDZ Boundary')
    # ax_5.plot(10 * np.cos(np.linspace(-np.pi, np.pi, 100)), 10 * np.sin(np.linspace(-np.pi, np.pi, 100)),
    #           '--', lw=2, color='red', label=r'r=10$\mathrm{R_s}$ Circle')
    # ax_5.set_xlim(2, 14)
    # ax_5.set_ylim(-7, 7)
    # ax_5.set_xlabel(r"r  [$\mathrm{R_s}$]", fontsize=14)
    # ax_5.set_ylabel(r"z  [$\mathrm{R_s}$]", fontsize=14)
    # ax_5.tick_params(labelsize=12)
    # ax_5.legend(fontsize=13)
    # plt.show()
    ##################################################################################################
    var_str = "Cumulative Number Density [km^3]"
    spacing = ((x_plot[-1] - x_plot[0]) / (x_plot.size - 1),
               (y_plot[-1] - y_plot[0]) / (y_plot.size - 1),
               (z_plot[-1] - z_plot[0]) / (z_plot.size - 1))
    origin = (x_plot[0], y_plot[0], z_plot[0])
    box_grid = pv.ImageData(dimensions=(x_plot.size, y_plot.size, z_plot.size),
                            spacing=spacing,
                            origin=origin)
    box_grid.point_data[var_str] = density.ravel('F') * 1e9
    p = pv.Plotter()
    vol = p.add_volume(box_grid, cmap='viridis',
                       # Change Colorbar
                       opacity=(0., 0.2, 0.4, 0.6, 0.9),  # Change Opacity Mask
                       # opacity_unit_distance=400,
                       clim=[1e7, 2e8],
                       scalar_bar_args=dict(title_font_size=28, label_font_size=24, color='black'))
    # p.show_bounds(axes_ranges=[-20, 80, -60, 60, -30, 30])
    p.show_grid(axes_ranges=[-20, 80, -60, 60, -30, 30],
                xtitle='X [Rs]', ytitle='Y [Rs]', ztitle='Z [Rs]')
    p.show_axes()
    # p.set_background("white")
    # p.add_title('Heliosphere')
    # p.view_vector((0, -50, 0))
    p.view_vector([-0.75, -1, 0])
    p.show()


