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
from scipy.stats import gaussian_kde
import trace_counting_func as tcf


h = 6.626e-34  # Planck constant
c_0 = 3e8  # the velocity of light in vacuum
k_B = 1.38e-23  # Boltzmann constant
R_s = 6.955e8  # the average radius of SUN.
T_s = 5775  # the temperature of SUN is approximately 5775 K.
psp_3d_model_path = 'D://Microsoft Download/Formal Files/data file/3d_model/ParkerSolarProbe/stl/ParkerSolarProbe.stl'
WISPR_pos = np.array([0.865, -0.249, -0.300], dtype=float)  # the position of WISPR onboard PSP in spacecraft frame.
transfer_factor = 2.25 / 0.219372 / 2
the_step = 500


# def generate_par(pattern_flag):
#     """
#     Different from the two generating function in Examine05, particles that this function generates are fixed,
#     instead of random.
#     :param pattern_flag: 1 means generating parallel-motion particles, 2 means generating divergence-motion particles.
#     :return: fun_state: same as that in function 'generate_par_1()'
#     """
#     num_par = 10
#     fun_state = np.zeros((num_par, 4, 2), dtype=float)
#     fun_radius = np.random.random(num_par) * (1e-5 - 1e-7) + 1e-7
#     fun_state[:, 0, 0] = fun_state[:, 0, 1] = fun_radius
#     if pattern_flag is 1:
#         fun_state[:, 1, 0] = 2*np.array([-5, -3, -2, -6, -1.5,
#                                          -3.2, -4.6, -2.6, -5.5, -3.1],
#                                         dtype=float)
#         fun_state[:, 2, 0] = 2*np.array([-3.32, -2.1, 6.1, -1.2, 2.1,
#                                          -4.4, -6.6, 3.4, 2.6, 5.5],
#                                         dtype=float)
#         fun_state[:, 3, 0] = 2*np.array([10, 13, 16, 20, 11,
#                                          16.3, 12.3, 13.9, 18.1, 10.5],
#                                         dtype=float)
#         fun_velocity = np.array([0.40, 0.1968, -0.8951], dtype=float) * 10
#         fun_state[:, 1, 1] = fun_state[:, 1, 0] + fun_velocity[0] * 1
#         fun_state[:, 2, 1] = fun_state[:, 2, 0] + fun_velocity[1] * 1
#         fun_state[:, 3, 1] = fun_state[:, 3, 0] + fun_velocity[2] * 1
#     return fun_state
#
#
# def coor_transform(fun_state, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
#     """
#     This function transforms the fun_state in WISPR_Inner frame to that in SC frame, in order to plot the 3d figure.
#     :param fun_state: the output array in the function generate_par_1()
#     :param timestr: (str) the time of the figure
#     :param time_format: (str) the format of the time string, such as 'yyyy-MM-DDThh:mm:ss'.
#     :return: the similar array to fun_state, but in SC frame.
#     """
#     etime = spice.datetime2et(datetime.strptime(timestr, time_format))
#     tar_state = np.zeros((len(fun_state[:, 0, 0]), 4, 2), dtype=float)
#     for fun_i in range(len(fun_state[:, 0, 0])):
#         tar_start_state, _ = spice.spkcpt([fun_state[fun_i, 1, 0], fun_state[fun_i, 2, 0], fun_state[fun_i, 3, 0]],
#                                           'SPP', 'SPP_WISPR_INNER', etime, 'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
#         tar_end_state, _ = spice.spkcpt([fun_state[fun_i, 1, 1], fun_state[fun_i, 2, 1], fun_state[fun_i, 3, 1]],
#                                         'SPP', 'SPP_WISPR_INNER', etime, 'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
#         tar_state[fun_i, 1:4, 0] = tar_start_state[0:3]
#         tar_state[fun_i, 1:4, 1] = tar_end_state[0:3]
#         tar_state[fun_i, 0, :] = fun_state[fun_i, 0, :]
#     return tar_state
#
#
# def cal_elongation(coor_wispr_i, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
#     """
#     This function calculates the elongation (rad).
#     elongation: the angle from PSP between the particle and SUN.
#     :param coor_wispr_i: the coordinate of particle in WISPR_Inner frame
#     :param timestr: ...
#     :param time_format: ... (the same as above)
#     :return:the elongation
#     """
#     etime = spice.datetime2et(datetime.strptime(timestr, time_format))
#     coor_sc, _ = spice.spkcpt(coor_wispr_i, 'SPP', 'SPP_WISPR_INNER', etime,
#                                       'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
#     coor_sc = np.array(coor_sc, dtype=float)
#     cos_elogation = np.dot(coor_sc[0:3], np.array([0, 0, 1], dtype=float)) / np.linalg.norm(coor_sc[0:3])
#     elongation = np.arccos(cos_elogation)
#     return elongation
#
#
# def solar_spectrum(wavelength):
#     """
#     :param wavelength: unit: m
#     :return: the irradience of sun (unit: W/m^2)
#     """
#     irradience_0 = 2 * h * c_0**2 / wavelength**5 / (np.exp(h*c_0/wavelength/k_B/T_s)-1) * 10e-9 * np.pi
#     # the factor 'pi' comes from Lambert's cosine law.
#     return irradience_0
#
#
# def scattering_intensity(elongation, radius, z_to_obs, d_to_obs, velocity, obs_time, time_format='%Y%m%dT%H:%M:%S'):
#     """
#         This function refers to Mie scattering theory and uses PyMieScatt package.
#         :param elongation: the elongation (unit: degree)
#         :param radius: the radius of the particle (unit: pixel)
#         :param z_to_obs: the distance from particle to observer in z axis of camera frame(unit:m)
#         :param d_to_obs: the distance from particle to observer (unit: m)
#         :param velocity: [1*3 ndarray]the velocity of particle (unit: m/s) in camera frame(i.e. WISPR-Inner frame)
#         :param obs_time: the observing time of the map
#         :param time_format: time string's format
#         :return: the scattering intensity (unit: MSB)
#         """
#     f = 28e-3  # unit: m
#     A_aperture = 42e-6  # unit: m^2
#     A_pixel = 10e-6 ** 2  # unit: m^2
#     deg_to_rad = np.pi / 180
#     qe_transmission = 0.24
#     wavelength = np.linspace(480, 770, 30) * 1e-9  # 480 - 770 nm, unit of it: m
#     elongation_rad = elongation * deg_to_rad
#     etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
#     PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
#     psp_distance = np.sqrt(PSP_pos[0] ** 2 + PSP_pos[1] ** 2 + PSP_pos[2] ** 2) * 1e3
#     # unit in spice is km, while we require it to be m.
#     r_to_SUN = np.sqrt(d_to_obs ** 2 + psp_distance ** 2 - 2 * d_to_obs * psp_distance * np.cos(elongation_rad))
#     sin_scattering_theta = psp_distance * np.sin(elongation_rad) / r_to_SUN
#     albedo = 0.25
#     photon_num = 0
#     for fun_i in range(30):
#         alpha = 2 * np.pi * radius / wavelength[fun_i]
#         bessel = spe.jv(1, alpha * sin_scattering_theta)
#         bessel_length = np.sqrt(bessel.real ** 2 + bessel.imag ** 2)
#         sigma = radius ** 2 * bessel_length ** 2 / np.abs(sin_scattering_theta) ** 2 + albedo * radius ** 2 / 4
#         the_irradiance = solar_spectrum(wavelength[fun_i])
#         max_speed = np.maximum(np.abs(velocity[0]), np.abs(velocity[1]))
#         intensity_W = sigma * the_irradiance * R_s**2 / r_to_SUN**2 * A_aperture / d_to_obs**2
#         # photon_num = photon_num + intensity_W * 10e-6 * d_to_obs / f / velocity / h / c_0 * wavelength[fun_i]
#         photon_num = photon_num + intensity_W * 10e-6 * z_to_obs / f / max_speed / h / c_0 * wavelength[fun_i]
#     total_electron_num = photon_num * qe_transmission
#     intensity_DN = total_electron_num / 2.716
#     intensity_MSB = intensity_DN / 48.64 * 3.93e-14
#     return intensity_MSB
#
#
# def par_in_camera(num_par, fun_state,  single_exp, total_exp, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
#     """
#     :param num_par
#     :param fun_state:
#     :param single_exp:
#     :param total_exp:
#     :param timestr:
#     :param time_format:
#     :return:
#     """
#     f = 28e-3  # unit: m
#     a_pixel = 10e-6  # unit: m
#     fun_step = 5000
#     the_velocity = np.zeros((num_par, 3), dtype=float)
#     trace_3d = np.zeros((num_par, fun_step+1, 3), dtype=float)
#     trace_2d = np.zeros((num_par, fun_step+1, 2), dtype=float)
#     orientation_2d = np.zeros((num_par, 2), dtype=float)
#     # orientation_2d[i] means a unit vector that is vertical to the (i+1)_th particle trace in camera frame (x-y, 2d).
#     width_2d = np.zeros((num_par, fun_step+1), dtype=float)
#     fun_elongation = np.zeros((num_par, fun_step+1), dtype=float)
#     brightness_2d = np.zeros((num_par, fun_step+1), dtype=float)
#     for fun_j in range(num_par):
#         the_velocity[fun_j, :] = (fun_state[fun_j, 1:4, 1] - fun_state[fun_j, 1:4, 0]) / single_exp
#         for fun_i in range(fun_step+1):
#             trace_3d[fun_j, fun_i, :] = fun_state[fun_j, 1:4, 0] + fun_i / fun_step * (fun_state[fun_j, 1:4, 1] - fun_state[fun_j, 1:4, 0])
#             trace_2d[fun_j, fun_i, 0] = trace_3d[fun_j, fun_i, 0] * f / trace_3d[fun_j, fun_i, 2] / a_pixel
#             trace_2d[fun_j, fun_i, 1] = trace_3d[fun_j, fun_i, 1] * f / trace_3d[fun_j, fun_i, 2] / a_pixel
#             width_2d[fun_j, fun_i] = 2 * max(np.sqrt(42e-6) * f / trace_3d[fun_j, fun_i, 2] / a_pixel, 1)
#             temp_dis = np.sqrt(
#                 trace_3d[fun_j, fun_i, 2] ** 2 + trace_3d[fun_j, fun_i, 1] ** 2 + trace_3d[fun_j, fun_i, 0] ** 2)
#             fun_elongation[fun_j, fun_i] = cal_elongation(trace_3d[fun_j, fun_i, :], timestr, time_format=time_format)
#             brightness_2d[fun_j, fun_i] = scattering_intensity(fun_elongation[fun_j, fun_i], fun_state[fun_j, 0, 0],
#                                                                trace_3d[fun_j, fun_i, 2], temp_dis,
#                                                                the_velocity[fun_j], timestr, time_format)
#         orientation_2d[fun_j, 1] = trace_2d[fun_j, -1, 0] - trace_2d[fun_j, 0, 0]
#         orientation_2d[fun_j, 0] = -(trace_2d[fun_j, -1, 1] - trace_2d[fun_j, 0, 1])
#         temp_len = np.sqrt(orientation_2d[fun_j, 0]**2 + orientation_2d[fun_j, 1]**2)
#         orientation_2d[fun_j, :] = orientation_2d[fun_j, :] / temp_len
#     # transform the unit from meter to pixel
#     update_info = [num_par, fun_step, trace_2d, width_2d, orientation_2d, brightness_2d]
#     return update_info, trace_3d


def generate_par(method_flag):
    """
    Different from the two generating function above, particles that this function generates are fixed, instead of random.
    :param method_flag: 1 means generating parallel-motion particles, 2 means generating divergence-motion particles.
    :return: fun_state: same as that in function 'generate_par_1()'
    """
    num_par = 10
    fun_state = np.zeros((num_par, 4, 2), dtype=float)
    fun_radius = np.random.random(num_par) * (1e-5 - 1e-7) + 1e-7
    fun_state[:, 0, 0] = fun_state[:, 0, 1] = fun_radius
    if method_flag == 1:
        fun_state[:, 1, 0] = 2*np.array([-5, -3, -2, -6, -1.5,
                                         -3.2, -4.6, -2.6, -5.5, -3.1],
                                        dtype=float)
        fun_state[:, 2, 0] = 2*np.array([-3.32, -2.1, 6.1, -1.2, 2.1,
                                         -4.4, -6.6, 3.4, 2.6, 5.5],
                                        dtype=float)
        fun_state[:, 3, 0] = 2*np.array([10, 13, 16, 20, 11,
                                         16.3, 12.3, 13.9, 18.1, 10.5],
                                        dtype=float)
        fun_velocity = np.array([0.40, 0.1968, -0.8951], dtype=float) * 10
        fun_state[:, 1, 1] = fun_state[:, 1, 0] + fun_velocity[0] * 1
        fun_state[:, 2, 1] = fun_state[:, 2, 0] + fun_velocity[1] * 1
        fun_state[:, 3, 1] = fun_state[:, 3, 0] + fun_velocity[2] * 1
    return fun_state


def coor_transform(fun_state, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
    This function transforms the fun_state in WISPR_Inner frame to that in SC frame, in order to plot the 3d figure.
    :param fun_state: the output array in the function generate_par_1()
    :param timestr: (str) the time of the figure
    :param time_format: (str) the format of the time string, such as 'yyyy-MM-DDThh:mm:ss'.
    :return: the similar array to fun_state, but in SC frame.
    """
    etime = spice.datetime2et(datetime.strptime(timestr, time_format))
    tar_state = np.zeros((len(fun_state[:, 0, 0]), 4, 2), dtype=float)
    for fun_i in range(len(fun_state[:, 0, 0])):
        tar_start_state, _ = spice.spkcpt([fun_state[fun_i, 1, 0], fun_state[fun_i, 2, 0], fun_state[fun_i, 3, 0]],
                                          'SPP', 'SPP_WISPR_INNER', etime, 'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
        tar_end_state, _ = spice.spkcpt([fun_state[fun_i, 1, 1], fun_state[fun_i, 2, 1], fun_state[fun_i, 3, 1]],
                                        'SPP', 'SPP_WISPR_INNER', etime, 'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
        tar_state[fun_i, 1:4, 0] = tar_start_state[0:3]
        tar_state[fun_i, 1:4, 1] = tar_end_state[0:3]
        tar_state[fun_i, 0, :] = fun_state[fun_i, 0, :]
    return tar_state


def cal_elongation(coor_wispr_i, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
    This function calculates the elongation (rad).
    elongation: the angle from PSP between the particle and SUN.
    :param coor_wispr_i: the coordinate of particle in WISPR_Inner frame
    :param timestr: ...
    :param time_format: ... (the same as above)
    :return:the elongation
    """
    etime = spice.datetime2et(datetime.strptime(timestr, time_format))
    coor_sc, _ = spice.spkcpt(coor_wispr_i, 'SPP', 'SPP_WISPR_INNER', etime,
                              'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    coor_sc = np.array(coor_sc, dtype=float)
    cos_elogation = np.dot(coor_sc[0:3], np.array([0, 0, 1], dtype=float)) / np.linalg.norm(coor_sc[0:3])
    elongation = np.arccos(cos_elogation)
    return elongation


def solar_spectrum(wavelength):
    """
    :param wavelength: unit: m
    :return: the irradience of sun (unit: W/m^2)
    """
    irradience_0 = 2 * h * c_0**2 / wavelength**5 / (np.exp(h*c_0/wavelength/k_B/T_s)-1) * 10e-9 * np.pi
    # the factor 'pi' comes from Lambert's cosine law.
    return irradience_0


def scattering_intensity(elongation, radius, d_to_obs, max_speed, total_exp,
                         obs_time, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
        This method to calculating the scattering brightness refers to Mie scattering theory.
        :param elongation: the elongation (rad).
        :param radius: the radius of the particle (unit: m)
        :param d_to_obs: the distance from particle to observer (unit: m)
        :param max_speed: the maximum projective speed between the velocity in x-axis and y-axis. (unit: m/s)
        :param total_exp: the summed time of each exposure of  a WISPR figure. (unit: s)
        :param obs_time: the observing time of the map
        :param time_format: time string's format
        :return: the scattering intensity (unit: MSB)
        """
    f = 28e-3  # unit: m, focal length of wispr_inner
    A_aperture = 42e-6  # unit: m^2, aperture of wispr_inner
    A_pixel = 10e-6 ** 2  # unit: m^2, area of one APS pixel
    qe_transmission = 0.24
    AU = 1.496e11  # unit: m
    wavelength = np.linspace(480, 770, 30) * 1e-9  # 480 - 770 nm, unit of it: m
    etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
    PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
    psp_distance = np.sqrt(PSP_pos[0] ** 2 + PSP_pos[1] ** 2 + PSP_pos[2] ** 2) * 1e3
    # unit in spice is km, while we require it to be m.
    r_to_SUN = np.sqrt(d_to_obs ** 2 + psp_distance ** 2 - 2 * d_to_obs * psp_distance * np.cos(elongation))
    sin_scattering_theta = psp_distance * np.sin(elongation) / r_to_SUN
    albedo = 0.25
    photon_num = 0
    for fun_i in range(30):
        alpha = 2 * np.pi * radius / wavelength[fun_i]
        bessel = spe.jv(1, alpha * sin_scattering_theta)
        bessel_length = np.sqrt(bessel.real ** 2 + bessel.imag ** 2)
        sigma = radius ** 2 * bessel_length ** 2 / np.abs(sin_scattering_theta) ** 2 + albedo * radius ** 2 / 4
        the_irradience = solar_spectrum(wavelength[fun_i])
        intensity_W = sigma * the_irradience * R_s**2 / r_to_SUN**2 * A_aperture / d_to_obs**2
        photon_num = photon_num + intensity_W * 10e-6 / max_speed / h / c_0 * wavelength[fun_i]
    total_electron_num = photon_num * qe_transmission
    # intensity_DN = total_electron_num / 2.716 * A_pixel / max(A_pixel, np.pi * (radius * f / d_to_obs)**2)
    intensity_DN = total_electron_num / 2.716
    intensity_MSB = intensity_DN / total_exp * 3.93e-14
    # the factor from DN/s to MSB is 3.93e-14. (refers to Hess et.al. 2021)
    return intensity_MSB


def par_in_camera(num_par, fun_state,  single_exp, total_exp, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
    :param num_par
    :param fun_state:
    :param single_exp:
    :param total_exp:
    :param timestr:
    :param time_format:
    :return:
    """
    f = 28e-3  # unit: m
    a_pixel = 10e-6  # unit: m
    fun_step = the_step
    trace_3d = np.zeros((num_par, fun_step+1, 3), dtype=float)
    trace_2d = np.zeros((num_par, fun_step+1, 2), dtype=float)
    orientation_2d = np.zeros((num_par, 2), dtype=float)
    # orientation_2d[i] means a unit vector that is vertical to the (i+1)_th particle trace in camera frame (x-y, 2d).
    width_2d = np.zeros((num_par, fun_step+1), dtype=float)
    fun_elongation = np.zeros((num_par, fun_step+1), dtype=float)
    brightness_2d = np.zeros((num_par, fun_step+1), dtype=float)
    for fun_j in range(num_par):
        for fun_i in range(fun_step+1):
            trace_3d[fun_j, fun_i, :] = fun_state[fun_j, 1:4, 0] + fun_i / fun_step * (fun_state[fun_j, 1:4, 1] - fun_state[fun_j, 1:4, 0])
            trace_2d[fun_j, fun_i, 0] = trace_3d[fun_j, fun_i, 0] * f / trace_3d[fun_j, fun_i, 2] / a_pixel
            trace_2d[fun_j, fun_i, 1] = trace_3d[fun_j, fun_i, 1] * f / trace_3d[fun_j, fun_i, 2] / a_pixel
            width_2d[fun_j, fun_i] = 2 * max(np.sqrt(42e-6) * f / trace_3d[fun_j, fun_i, 2] / a_pixel, 1)
            temp_dis = np.sqrt(trace_3d[fun_j, fun_i, 2]**2+trace_3d[fun_j, fun_i, 1]**2+trace_3d[fun_j, fun_i, 0]**2)
            temp_speed = np.maximum(np.abs(fun_state[fun_j, 1, 1] - fun_state[fun_j, 1, 0]),
                                    np.abs(fun_state[fun_j, 1, 1] - fun_state[fun_j, 1, 0])
                                    ) / single_exp * f / trace_3d[fun_j, fun_i, 2]
            fun_elongation[fun_j, fun_i] = cal_elongation(trace_3d[fun_j, fun_i, :], timestr, time_format=time_format)
            brightness_2d[fun_j, fun_i] = scattering_intensity(fun_elongation[fun_j, fun_i], fun_state[fun_j, 0, 0],
                                                               temp_dis, temp_speed,
                                                               total_exp, timestr, time_format)
        orientation_2d[fun_j, 1] = trace_2d[fun_j, -1, 0] - trace_2d[fun_j, 0, 0]
        orientation_2d[fun_j, 0] = -(trace_2d[fun_j, -1, 1] - trace_2d[fun_j, 0, 1])
        temp_len = np.sqrt(orientation_2d[fun_j, 0]**2 + orientation_2d[fun_j, 1]**2)
        orientation_2d[fun_j, :] = orientation_2d[fun_j, :] / temp_len
    # transform the unit from meter to pixel
    update_info = [num_par, fun_step, trace_2d, width_2d, orientation_2d, brightness_2d]
    return update_info, trace_3d


def observing_brightness_diffusion(num_streak, path='D://Microsoft Download/Formal Files/data file/FITS/WISPR-I_ENC07'
                                                    '_L3_FITS/20210112/psp_L3_wispr_20210112T030017_V1_1221.fits',
                                   output_file_path='D://Desktop/output.txt'):
    """
    :param num_streak: The number of streaks retrieved
    :param path: The file path of  FITS(input)
    :param output_file_path: The file path of output txt file.
    :return: NULL.
    """
    step_num = the_step
    scale = 5
    info = tcf.get_point_slope(num_streak, path)
    brightness_array = np.zeros((info[4], step_num+1))
    fun_fig = plt.figure(figsize=(9, 9))
    fun_ax = fun_fig.add_subplot(111)
    step_array = np.linspace(0, 1, step_num+1, endpoint=True)
    for fun_i in range(info[4]):
        temp_step_len = (info[2][fun_i, 1, 0] - info[2][fun_i, 0, 0]) / step_num
        for fun_j in range(step_num+1):
            temp_coor_x = int(info[2][fun_i, 0, 0] + fun_j * temp_step_len)
            temp_coor_y = int(info[3][fun_i, 0]*(info[2][fun_i, 0, 0] + fun_j * temp_step_len) + info[3][fun_i, 1])
            # brightness_array[fun_i, fun_j] = np.max(info[1][temp_coor_y-5:temp_coor_y+5, temp_coor_x])
            brightness_array[fun_i, fun_j] = info[1][temp_coor_y, temp_coor_x]
            # np.median(info[1][temp_y_min:temp_y_max, temp_coor_x])
        brightness_array[fun_i] = brightness_array[fun_i] / brightness_array[fun_i, 0]
    brightness_array_1d = brightness_array.reshape(-1, 1)
    step_array_1d = np.zeros_like(brightness_array_1d)
    brightness_std = brightness_array.std(axis=0)
    for fun_i in range(info[4]):
        step_array_1d[fun_i*(step_num+1):(fun_i+1)*(step_num+1), 0] = step_array
    xy = np.vstack([step_array_1d[:, 0], brightness_array_1d[:, 0]])
    z = gaussian_kde(xy)(xy)
    fun_ax.scatter(step_array_1d[:, 0], brightness_array_1d[:, 0], c=z, s=15, cmap='Spectral_r')
    fun_ax.set_xlim(0, 1)
    fun_ax.set_ylim(0, 3)
    fun_ax.set_xlabel('Normalized Distance from the Left End of the Streak', fontsize='15')
    fun_ax.set_ylabel('B/B_0', fontsize='15')
    fun_ax.tick_params(labelsize=15)
    plt.show()
    file = open(output_file_path, 'a+')
    file.write('STEP_ARRAY:\n')
    for fun_i in range(step_array_1d.size-1):
        file.write(str(step_array_1d[fun_i, 0])+', ')
    file.write(str(step_array_1d[-1, 0])+'\n\n')
    file.write('BRIGHTNESS_ARRAY:\n')
    for fun_i in range(brightness_array_1d.size-1):
        file.write(str(brightness_array_1d[fun_i, 0])+', ')
    file.write(str(brightness_array_1d[-1, 0]))
    file.close()


def read_output_file(file_path='D://Desktop/output.txt'):
    """
    The file is the txt file output in function above: 'observing_brightness_diffusion()'
    :param file_path:
    :return:
    """
    the_file = open(file_path, 'r')
    all_str = the_file.read().splitlines()
    the_file.close()
    step_array_1d = np.array(all_str[1].split(', '), dtype=float)
    brightness_array_1d = np.array(all_str[4].split(', '), dtype=float)
    # delete_index = np.bitwise_or(brightness_array_1d > 3, brightness_array_1d < 0)
    delete_index = np.array(np.where(np.bitwise_or(brightness_array_1d > 3, brightness_array_1d < 0)), dtype=int)
    delete_index = delete_index.flatten()
    insert_index = copy.deepcopy(delete_index)
    brightness_array_1d_simp = np.delete(brightness_array_1d, delete_index)
    step_array_1d_simp = np.delete(step_array_1d, delete_index)
    for fun_i in range(len(delete_index)):
        insert_index[fun_i] = delete_index[fun_i] - fun_i

    brightness_array_1d_nan = np.insert(brightness_array_1d_simp, insert_index, np.nan)
    step_array_1d_nan = np.insert(step_array_1d_simp, insert_index, np.nan)
    step_array_nan = step_array_1d_nan.reshape(-1, the_step+1)
    brightness_array_nan = brightness_array_1d_nan.reshape(-1, the_step+1)

    # brightness_array_1d_simp = np.log(brightness_array_1d_simp)
    xy = np.vstack([step_array_1d_simp[:], brightness_array_1d_simp[:]])
    density_1d_simp = gaussian_kde(xy)(xy)
    density_1d = np.insert(density_1d_simp, insert_index, np.nan)
    density = density_1d.reshape(-1, the_step+1)
    step = np.linspace(0, the_step, num=the_step+1, dtype=int)
    max_dens_pos = np.nanargmax(density, axis=0)
    standard_devi = np.nanstd(density, axis=0)

    fun_fig = plt.figure(figsize=(9, 9))
    fun_ax = fun_fig.add_subplot(1, 1, 1)
    fun_ax.set_xlim(0, 1)
    fun_ax.set_ylim(0, 3)
    fun_ax.scatter(step_array_1d_simp[:], brightness_array_1d_simp[:], c=density_1d_simp, s=120, cmap='Spectral_r')
    fun_ax.plot(step_array_1d[0:the_step+1], brightness_array_nan[max_dens_pos, step], linewidth=2, color='black')
    fun_ax.fill_between(step_array_1d[0:the_step+1], brightness_array_nan[max_dens_pos, step] - standard_devi,
                        brightness_array_nan[max_dens_pos, step] + standard_devi, alpha=0.4,
                        color='grey')
    # fun_ax.set_yscale('log')
    fun_ax.set_xlabel('Normalized Distance from the Left End of the Streak', fontsize='15')
    fun_ax.set_ylabel('B/B_0', fontsize='15')
    fun_ax.tick_params(labelsize=15)
    fun_fig.suptitle('Created by Examine_08.py', fontsize=15)
    plt.show()


def main_function():
    my_path = 'D://Microsoft Download/Formal Files/data file/FITS/WISPR-I_ENC07_L3_FITS/20210116/' \
              'psp_L3_wispr_20210116T203018_V1_1211.fits'
    data, header = read_file(my_path, 'fits')[0]
    my_par_num = 6
    # observing_brightness_diffusion(my_par_num, my_path)
    read_output_file(file_path='D://Microsoft Download/Formal Files/work/WISPR Dust Storm-Like Event/'
                               'Brightness Evolution/ON_SC_20210116T203018.txt')


main_function()


