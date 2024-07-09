"""
This script is for simulation of streaks in WISPR.
This 3d space is located at spacecraft(SC) frame.
"""
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


h = 6.626e-34  # Planck constant
c_0 = 3e8  # the velocity of light in vacuum
k_B = 1.38e-23  # Boltzmann constant
R_s = 6.955e8  # the average radius of SUN.
T_s = 5775  # the temperature of SUN is approximately 5775 K.
psp_3d_model_path = 'D://Microsoft Download/Formal Files/data file/3d_model/ParkerSolarProbe/stl/ParkerSolarProbe.stl'
WISPR_pos = np.array([0.865, -0.249, -0.300], dtype=float)  # the position of WISPR onboard PSP in spacecraft frame.
transfer_factor = 2.25 / 0.219372 / 2
generate_flag = 3  # 1 means using the generating function 'generate_par_1', 2 means 'generate_par_2'


def generate_par_1(num_par, range_distance, range_radius, velocity, fun_exp_time):
    """
    This function generate the state of particles in WISPR_Inner frame. These particles are of the same velocity.
    :param num_par: (int)The number of particles generated.
    :param range_distance: (1*2 float ndarray) The range of distance from particle to WISPR(or specifically, the value
    of z coordinate of particles in WISPR_Inner frame). [0] means min, while [1] means max. (unit: m)
    :param range_radius: (1*2 float ndarray) The range of radii of particles, [0] means min, while [1] means max.
     (unit: m)
    :param velocity: (1*4 float ndarray) the common velocity vector of particles in spacecraft-stationary frame
    (WISPR-Inner frame).
    [0] means the magnitude of it. [1-3] means the orientation of it. (unit: m/s)
    :param fun_exp_time: (float) the single exposure time of WISPR figure.
    :return: fun_state: (an n*4*2 ndarray) the description of it is shown below.
             velocity[0]: the magnitude of velocity.
    """
    x_max = range_distance[1] * 960 * 10e-6 / 28e-3
    y_max = range_distance[1] * 1024 * 10e-6 / 28e-3
    z_range = copy.deepcopy(range_distance)
    fun_radius = np.random.random(num_par) * (range_radius[1] - range_radius[0]) + range_radius[0]
    coor_x_beg = -x_max + 2 * np.random.random(num_par) * x_max
    coor_y_beg = -y_max + 2 * np.random.random(num_par) * y_max
    coor_z_beg = np.random.random(num_par) * (z_range[1] - z_range[0]) + z_range[0]
    fun_state = np.zeros((num_par, 4, 2), dtype=float)
    # 4 means radius/x/y/z, 2 means the start/end position of particle. (e.g. fun_state[i, 3, 0] is the start y
    # coordinate of the (i+1)_th particle )(the coordinate is in WISPR_Inner frame and the origin is at WISPR-Inner)
    fun_state[:, 0, 0] = fun_state[:, 0, 1] = fun_radius
    fun_state[:, 1, 0] = coor_x_beg
    fun_state[:, 2, 0] = coor_y_beg
    fun_state[:, 3, 0] = coor_z_beg
    fun_state[:, 1, 1] = coor_x_beg + velocity[0] * velocity[1] * fun_exp_time
    fun_state[:, 2, 1] = coor_y_beg + velocity[0] * velocity[2] * fun_exp_time
    fun_state[:, 3, 1] = coor_z_beg + velocity[0] * velocity[3] * fun_exp_time
    return fun_state, velocity[0]


def generate_par_2(num_par, cone_origin, cone_axis, cone_angle, range_radius, range_speed, fun_exp_time,
                   timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
    This function generate the state of particles in WISPR_Inner frame. These particles are ejected from a single point.
    :param num_par: (int) The number of particles generated.
    :param cone_origin: (1*3 float ndarrays) The coordinate of the cone's origin(i.e. the origin of these particles).
    :param cone_axis: (1*3 float ndarrays) The unit vector of the cone's axis.
    :param cone_angle: (int, unit: degree) The angle of the cone.
    :param range_radius:(1*2 float ndarrays) The minimum and maximum of radii of these particles.
    :param range_speed: (1*2 float ndarrays) The minimum and maximum of magnitudes of these particles' velocities.
    :param fun_exp_time:(float) the single exposure time of WISPR figure.
    :param timestr:
    :param time_format:
    :return: fun_state: same as that in function 'generate_par_1()'
    """
    etime = spice.datetime2et(datetime.strptime(timestr, time_format))
    vel_vec = np.zeros([num_par, 3], dtype=float)
    vel_mag = np.random.random(num_par) * (range_speed[1] - range_speed[0]) + range_speed[0]
    fun_radius = np.random.random(num_par) * (range_radius[1] - range_radius[0]) + range_radius[0]
    base_unit = np.zeros([2, 3], dtype=float)
    base_unit[0, 0] = 1
    base_unit[0, 1] = 0
    base_unit[0, 2] = - cone_axis[0] * base_unit[0, 0] / cone_axis[2]
    base_unit[0, :] = base_unit[0, :] / np.sqrt(base_unit[0, 0]**2 + base_unit[0, 1]**2 + base_unit[0, 2]**2)
    base_unit[1, :] = np.cross(cone_axis, base_unit[0])

    r_cone = np.random.random(num_par) * np.tan(cone_angle / 180 * np.pi)
    phi_deg = np.random.random(num_par) * 180
    for fun_i in range(num_par):
        vel_vec[fun_i, :] = cone_axis[:] + r_cone[fun_i] * base_unit[0, :] * np.cos(phi_deg[fun_i]*np.pi/180) + \
                            r_cone[fun_i] * base_unit[1, :] * np.sin(phi_deg[fun_i]*np.pi/180)
        vel_vec[fun_i, :] = vel_vec[fun_i, :] / np.sqrt(vel_vec[fun_i, 0]**2 + vel_vec[fun_i, 1]**2 + vel_vec[fun_i, 2]**2)

    fun_state = np.zeros((num_par, 4, 2), dtype=float)
    # 4 means radius/x/y/z, 2 means the start/end position of particle. (e.g. fun_state[i, 3, 0] is the start y
    # coordinate of the (i+1)_th particle )
    fun_state[:, 0, 0] = fun_state[:, 0, 1] = fun_radius
    fun_state[:, 1, 0] = cone_origin[0]
    fun_state[:, 2, 0] = cone_origin[1]
    fun_state[:, 3, 0] = cone_origin[2]
    fun_state[:, 1, 1] = cone_origin[0] + vel_mag[:] * vel_vec[:, 0] * fun_exp_time
    fun_state[:, 2, 1] = cone_origin[1] + vel_mag[:] * vel_vec[:, 1] * fun_exp_time
    fun_state[:, 3, 1] = cone_origin[2] + vel_mag[:] * vel_vec[:, 2] * fun_exp_time

    fun_state[:, 1, 0] = fun_state[:, 1, 0] - WISPR_pos[0]
    fun_state[:, 2, 0] = fun_state[:, 2, 0] - WISPR_pos[1]
    fun_state[:, 3, 0] = fun_state[:, 3, 0] - WISPR_pos[2]
    fun_state[:, 1, 1] = fun_state[:, 1, 1] - WISPR_pos[0]
    fun_state[:, 2, 1] = fun_state[:, 2, 1] - WISPR_pos[1]
    fun_state[:, 3, 1] = fun_state[:, 3, 1] - WISPR_pos[2]

    for fun_i in range(num_par):
        temp_state, _ = spice.spkcpt([fun_state[fun_i, 1, 0], fun_state[fun_i, 2, 0], fun_state[fun_i, 3, 0]],
                                     'SPP', 'SPP_SPACECRAFT', etime, 'SPP_WISPR_INNER', 'OBSERVER', 'NONE', 'SPP')
        fun_state[fun_i, 1:4, 0] = temp_state[0:3]
        temp_state, _ = spice.spkcpt([fun_state[fun_i, 1, 1], fun_state[fun_i, 2, 1], fun_state[fun_i, 3, 1]],
                                     'SPP', 'SPP_SPACECRAFT', etime, 'SPP_WISPR_INNER', 'OBSERVER', 'NONE', 'SPP')
        fun_state[fun_i, 1:4, 1] = temp_state[0:3]
    return fun_state, vel_vec, vel_mag


def generate_par_3(method_flag):
    """
    Different from the two generating function above, particles that this function generates are fixed, instead of random.
    :param method_flag: 1 means generating parallel-motion particles, 2 means generating divergence-motion particles.
    :return: fun_state: same as that in function 'generate_par_1()'
    """
    num_par = 10
    fun_state = np.zeros((num_par, 4, 2), dtype=float)
    # fun_radius = np.random.random(num_par) * (1e-5 - 1e-7) + 1e-7
    fun_radius = np.array([1e-6]*10, dtype=float)
    fun_state[:, 0, 0] = fun_state[:, 0, 1] = fun_radius
    if method_flag is 1:
        fun_state[:, 1, 0] = 2*np.array([-5, -5, -5, -5, -5,
                                         -5, -5, -5, -5, -5],
                                        dtype=float)
        fun_state[:, 2, 0] = 2*np.array([-0.332, -1.21, 0.61, -1.2, 0.21,
                                         -1.44, -0.66, 0.34, 1.26, 0.55],
                                        dtype=float)
        fun_state[:, 3, 0] = 2*np.array([10, 10, 10, 10, 10,
                                         10, 10, 10, 10, 10],
                                        dtype=float)
        fun_velocity = np.array([0.462774, -0.037287, -0.88569115], dtype=float) * 5
        fun_state[:, 1, 1] = fun_state[:, 1, 0] + fun_velocity[0] * 1
        fun_state[:, 2, 1] = fun_state[:, 2, 0] + fun_velocity[1] * 1
        fun_state[:, 3, 1] = fun_state[:, 3, 0] + fun_velocity[2] * 1
    else:
        fun_state[:, 1, 0] = np.array([-0.77914151]*10, dtype=float)
        fun_state[:, 2, 0] = np.array([0.05773113]*10, dtype=float)
        fun_state[:, 3, 0] = np.array([1.46555316]*10, dtype=float)
        fun_velocity = np.array([[0.8, -0.2, np.sqrt(1-0.8**2-0.2**2)],
                                 [0.75, -0.35, -np.sqrt(1-0.75**2-0.35**2)],
                                 [0.7, 0.3, -np.sqrt(1-0.7**2-0.3**2)],
                                 [0.75, -0.36, np.sqrt(1-0.75**2-0.36**2)],
                                 [0.79, 0.30, np.sqrt(1-0.79**2-0.3**2)],
                                 [0.72, 0.42, np.sqrt(1-0.72**2-0.42**2)],
                                 [0.92, 0.12, np.sqrt(1 - 0.92 ** 2 - 0.12 ** 2)],
                                 [0.82, 0.22, -np.sqrt(1 - 0.82 ** 2 - 0.22 ** 2)],
                                 [1, -0.0, np.sqrt(1 - 1 ** 2 - 0.0 ** 2)],
                                 [0.82, 0.42, np.sqrt(1 - 0.82 ** 2 - 0.42 ** 2)]], dtype=float) * 3
        fun_state[:, 1, 1] = fun_state[:, 1, 0] + fun_velocity[:, 0] * 1
        fun_state[:, 2, 1] = fun_state[:, 2, 0] + fun_velocity[:, 1] * 1
        fun_state[:, 3, 1] = fun_state[:, 3, 0] + fun_velocity[:, 2] * 1
    return fun_state


def generate_par_4(velocity, method_flag):
    """
    This function generates one particle with its original position and velocity changing over time.
    :param velocity:
    :param method_flag:
    :return:
    """
    fun_state = np.zeros((1, 4, 2), dtype=float)
    if method_flag is 2:
        fun_state[0, 1, 0] = -0.77914151
        fun_state[0, 2, 0] = 0.05773113
        fun_state[0, 3, 0] = 1.46555316
        fun_state[0, 0, 0] = fun_state[0, 0, 1] = 1e-6
        fun_state[0, 1, 1] = velocity[0] * 1
        fun_state[0, 2, 1] = velocity[1] * 1
        fun_state[0, 3, 1] = velocity[2] * 1
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


def scattering_intensity(elongation, radius, d_to_obs, single_exp, total_exp,
                         streak_len, obs_time, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
        This method to calculating the scattering brightness refers to Mie scattering theory.
        :param elongation: the elongation (rad).
        :param radius: the radius of the particle (unit: m)
        :param d_to_obs: the distance from particle to observer (unit: m)
        :param single_exp: the time of single exposure of a WISPR figure. (unit: s)
        :param total_exp: the summed time of each exposure of  a WISPR figure. (unit: s)
        :param streak_len: the maximum of x and y length of the streak (unit: pixel)
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
        photon_num = photon_num + intensity_W * single_exp / streak_len / h / c_0 * wavelength[fun_i]
    total_electron_num = photon_num * qe_transmission
    # intensity_DN = total_electron_num / 2.716 * A_pixel / max(A_pixel, np.pi * (radius * f / d_to_obs)**2)
    intensity_DN = total_electron_num / 2.716
    intensity_MSB = intensity_DN / total_exp * 3.93e-14
    # the factor from DN/s to MSB is 3.93e-14. (refers to Hess et.al. 2021)
    return intensity_MSB


def par_in_camera(num_par, fun_state,  single_exp, total_exp, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
    :param num_par
    :param fun_state: The output array in 'generate_par_*()' function
    :param single_exp:
    :param total_exp:
    :param timestr:
    :param time_format:
    :return:
    """
    f = 28e-3  # unit: m
    a_pixel = 10e-6  # unit: m
    fun_step = 500
    trace_3d = np.zeros((num_par, fun_step+1, 3), dtype=float)
    trace_2d = np.zeros((num_par, fun_step+1, 2), dtype=float)
    orientation_2d = np.zeros((num_par, 2), dtype=float)
    # orientation_2d[i] means a unit vector that is vertical to the (i+1)_th particle trace in camera frame (x-y, 2d).
    width_2d = np.zeros((num_par, fun_step+1), dtype=float)
    fun_elongation = np.zeros((num_par, fun_step+1), dtype=float)
    brightness_2d = np.zeros((num_par, fun_step+1), dtype=float)
    for fun_j in range(num_par):
        max_len = max(np.fabs((fun_state[fun_j, 1, 1] / fun_state[fun_j, 3, 1] - fun_state[fun_j, 1, 0] / fun_state[fun_j, 3, 0]) * f / a_pixel),
                      np.fabs((fun_state[fun_j, 2, 1] / fun_state[fun_j, 3, 1] - fun_state[fun_j, 2, 0] / fun_state[fun_j, 3, 0]) * f / a_pixel)
                      )
        for fun_i in range(fun_step+1):
            trace_3d[fun_j, fun_i, :] = fun_state[fun_j, 1:4, 0] + fun_i / fun_step * (fun_state[fun_j, 1:4, 1] - fun_state[fun_j, 1:4, 0])
            trace_2d[fun_j, fun_i, 0] = trace_3d[fun_j, fun_i, 0] * f / trace_3d[fun_j, fun_i, 2] / a_pixel
            trace_2d[fun_j, fun_i, 1] = trace_3d[fun_j, fun_i, 1] * f / trace_3d[fun_j, fun_i, 2] / a_pixel
            width_2d[fun_j, fun_i] = 2 * max(np.sqrt(42e-6) * f / trace_3d[fun_j, fun_i, 2] / a_pixel, 1)
            temp_dis = np.sqrt(trace_3d[fun_j, fun_i, 2]**2+trace_3d[fun_j, fun_i, 1]**2+trace_3d[fun_j, fun_i, 0]**2)
            fun_elongation[fun_j, fun_i] = cal_elongation(trace_3d[fun_j, fun_i, :], timestr, time_format=time_format)
            brightness_2d[fun_j, fun_i] = scattering_intensity(fun_elongation[fun_j, fun_i], fun_state[fun_j, 0, 0],
                                                               temp_dis,
                                                               single_exp, total_exp, max_len, timestr, time_format)
            brightness_2d[fun_j, fun_i] = brightness_2d[fun_j, fun_i] * np.abs(trace_3d[fun_j, fun_i, 2])
            print('The present step percent is: %.3f %.3e %.1f %.3e' % (fun_i/(fun_step+1), temp_dis,
                                                                        fun_elongation[fun_j, fun_i] * 180 / np.pi,
                                                                        brightness_2d[fun_j, fun_i]))
        orientation_2d[fun_j, 1] = trace_2d[fun_j, -1, 0] - trace_2d[fun_j, 0, 0]
        orientation_2d[fun_j, 0] = -(trace_2d[fun_j, -1, 1] - trace_2d[fun_j, 0, 1])
        temp_len = np.sqrt(orientation_2d[fun_j, 0]**2 + orientation_2d[fun_j, 1]**2)
        orientation_2d[fun_j, :] = orientation_2d[fun_j, :] / temp_len
    # transform the unit from meter to pixel
    update_info = [num_par, fun_step, trace_2d, width_2d, orientation_2d, brightness_2d]
    return update_info, trace_3d


def update_fig(update_info, raw_data):
    """
    :param update_info:
    :param raw_data:
    :return: the final result( 2048 * 1920 pixel array)(similar to raw_data).
    """
    fig_data = copy.deepcopy(raw_data)
    fig_data[:, :] = 1e-15
    for fun_j in range(update_info[0]):
        for fun_i in range(update_info[1]):
            update_info[2][fun_j, fun_i, 0] = update_info[2][fun_j, fun_i, 0] + 960
            update_info[2][fun_j, fun_i, 1] = update_info[2][fun_j, fun_i, 1] + 1024
    temp_pos = np.zeros(2, dtype=int)
    for fun_j in range(update_info[0]):
        for fun_i in range(update_info[1]+1):
            temp_pos[0] = update_info[2][fun_j, fun_i, 0]
            temp_pos[1] = update_info[2][fun_j, fun_i, 1]
            pos_min = np.zeros(2, dtype=float)
            pos_max = np.zeros(2, dtype=float)
            if temp_pos[0] >= 1920 or temp_pos[0] <= 0 or temp_pos[1] >= 2048 or temp_pos[1] <= 0:
                continue
            elif update_info[3][fun_j, fun_i] > 1:
                pos_min[0] = temp_pos[0] - update_info[3][fun_j, fun_i] * update_info[4][fun_j, 0]
                pos_min[1] = temp_pos[1] - update_info[3][fun_j, fun_i] * update_info[4][fun_j, 1]
                pos_max[0] = temp_pos[0] + update_info[3][fun_j, fun_i] * update_info[4][fun_j, 0]
                pos_max[1] = temp_pos[1] + update_info[3][fun_j, fun_i] * update_info[4][fun_j, 1]
                pos_all_1 = np.linspace(int(pos_min[0]), int(pos_max[0]), np.abs(int(pos_max[0]) - int(pos_min[0]))+1
                                        , endpoint=True, dtype=int)
                pos_all_2 = np.linspace(int(pos_min[1]), int(pos_max[1]), np.abs(int(pos_max[1]) - int(pos_min[1]))+1
                                        , endpoint=True, dtype=int)
                for fun_k in range(min(len(pos_all_1), len(pos_all_2))):
                    if pos_all_1[fun_k] >= 1920 or pos_all_1[fun_k] <= 0 or pos_all_2[fun_k] >= 2048 or \
                            pos_all_2[fun_k] <= 0:
                        continue
                    elif fig_data[pos_all_2[fun_k], pos_all_1[fun_k]] == 1e-15 and update_info[5][fun_j, fun_i] >= 1e-15:
                        fig_data[pos_all_2[fun_k], pos_all_1[fun_k]] = update_info[5][fun_j, fun_i]
                    else:
                        fig_data[pos_all_2[fun_k], pos_all_1[fun_k]] = max(fig_data[pos_all_2[fun_k], pos_all_1[fun_k]],
                                                                           update_info[5][fun_j, fun_i])

            elif fig_data[int(temp_pos[1])][int(temp_pos[0])] == 1e-15 and update_info[5][fun_j, fun_i] >= 1e-15:
                fig_data[int(temp_pos[1])][int(temp_pos[0])] = update_info[5][fun_j, fun_i]
            else:
                fig_data[int(temp_pos[1])][int(temp_pos[0])] = max(fig_data[int(temp_pos[1])][int(temp_pos[0])],
                                                                   update_info[5][fun_j, fun_i])
    return fig_data


def add_psp_3d_model(pos, rot_theta, scale=float(10)):
    """
    :param pos: A vector [x,y,z]. Center position of the spacecraft.
    :param rot_theta: A float (deg). Rotation around the z axis centered at pos.
    :param scale: A float, 10 by default. Scaling of the spacecraft.
    :return: A trace (go.Mesh3d()) for plotly.
    Note that the z axis in stl file is the true -z axis, x axis is the true y axis and y axis is the true -x axis!
    ——Tianhang Chen, 2022.3.18
    """
    mesh = pv.read(psp_3d_model_path)
    # mesh.plot()    showed by pyvista package.
    # scale = 10
    mesh.points = scale * mesh.points / np.max(mesh.points)
    # theta_x = 80
    # theta_z = -90 + rot_theta
    # theta_y = 0
    # axes = pv.Axes(show_actor=True, actor_scale=5.0, line_width=10)
    # axes.origin = (0, 0, 0)
    # rot = mesh.rotate_x(theta_x, point=axes.origin, inplace=False)
    # rot = rot.rotate_z(theta_z, point=axes.origin, inplace=False)

    # # Visualize by Pyvista
    # p = pv.Plotter()
    # p.add_actor(axes.actor)
    # p.add_mesh(rot)
    # # p.add_mesh(mesh)
    # p.show()

    vertices = mesh.points
    triangles = mesh.faces.reshape(-1, 4)
    trace = go.Mesh3d(x=vertices[:, 1] + pos[1], y=-(vertices[:, 0] + pos[0]), z=-(vertices[:, 2] + pos[2]),
                      opacity=1,            # the inner configuration of PSP is not significant, thus opacity set 1;
                      color='silver',       # the color changes from gold to silver,
                                            # in order to emphasize the dust traces and FOV   ——Tianhang Chen, 2022.3.18
                      i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                      showscale=False,
                      )
    return trace


def plot_FOV_and_FrameAxes(timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
    :param timestr: (str) the time of the figure
    :param time_format: (str) the format of the time string, such as 'yyyy-MM-DDThh:mm:ss'.
    :return: fun_fov_traces and fun_axes_traces are both the 'trace' type introduced in plotly(https://plotly.com/python)
    """
    epoch_time = spice.datetime2et(datetime.strptime(timestr, time_format))
    fun_fov_traces = []
    fun_axes_traces = []
    wispr_inner_parameter = spice.getfov(-96100, 4)
    for i_edge in range(4):
        edge_inner1, _ = spice.spkcpt(wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                      'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
        edge_motion = np.zeros([3, 1000], dtype=float)
        if generate_flag is 1:
            for cadence in range(1000):
                edge_motion[:, cadence] = (edge_inner1[0:3] * 3e2) * cadence / 6000
        else:
            for cadence in range(1000):
                edge_motion[:, cadence] = (edge_inner1[0:3] * 40) * cadence / 6000
        edge_motion[0, :] = edge_motion[0, :] + WISPR_pos[0]
        edge_motion[1, :] = edge_motion[1, :] + WISPR_pos[1]
        edge_motion[2, :] = edge_motion[2, :] + WISPR_pos[2]
        fun_fov_traces.append(go.Scatter3d(
                                             x=edge_motion[0], y=edge_motion[1], z=edge_motion[2], mode='lines',
                                             opacity=0.6, line=dict(color='green', width=5, dash='dash'),
                                             legendgroup='FOV',
                                             legendgrouptitle=dict(text='FOV of WISPR_Inner')
                                             )
                              )
    # ax_3.plot([0, the_center_point_3d[0]], [0, the_center_point_3d[1]], [0, the_center_point_3d[2]], c='orange'
    #           , label='vanishing point position')
    # the top(bottom) of streaks retrieved that converge at the y<0(y>0)
    # vanishing point in 3d frame.
    x_wispr_inner = np.array([2, 0, 0], dtype=float)
    x_sc, _ = spice.spkcpt(x_wispr_inner, 'SPP', 'SPP_WISPR_INNER', epoch_time, 'SPP_SPACECRAFT',
                           'OBSERVER', 'NONE', 'SPP')
    x_sc = np.array(x_sc, dtype=float)
    x_sc = x_sc[0:3] + WISPR_pos
    y_wispr_inner = np.array([0, 1, 0], dtype=float)
    y_sc, _ = spice.spkcpt(y_wispr_inner, 'SPP', 'SPP_WISPR_INNER', epoch_time, 'SPP_SPACECRAFT',
                           'OBSERVER', 'NONE', 'SPP')
    y_sc = np.array(y_sc, dtype=float)
    y_sc = y_sc[0:3] + WISPR_pos
    z_wispr_inner = np.array([0, 0, 4], dtype=float)
    z_sc, _ = spice.spkcpt(z_wispr_inner, 'SPP', 'SPP_WISPR_INNER', epoch_time, 'SPP_SPACECRAFT',
                           'OBSERVER', 'NONE', 'SPP')
    z_sc = np.array(z_sc, dtype=float)
    z_sc = z_sc[0:3] + WISPR_pos
    # axix_sc means the WISPR frame axis in spacecraft frame.
    fun_axes_traces.append(go.Scatter3d(
                                        x=np.array([WISPR_pos[0], x_sc[0]], dtype=float),
                                        y=np.array([WISPR_pos[1], x_sc[1]], dtype=float),
                                        z=np.array([WISPR_pos[2], x_sc[2]], dtype=float), mode='lines',
                                        opacity=0.6, line=dict(color='black', width=5), legendgroup='frame axis',
                                        legendgrouptitle=dict(text='WISPR_frame axis, longest '
                                                                   'for z, shortest for y')
                                        )
                           )
    fun_axes_traces.append(go.Scatter3d(
                                        x=np.array([WISPR_pos[0], y_sc[0]], dtype=float),
                                        y=np.array([WISPR_pos[1], y_sc[1]], dtype=float),
                                        z=np.array([WISPR_pos[2], y_sc[2]], dtype=float), mode='lines',
                                        opacity=0.6, line=dict(color='black', width=5), legendgroup='frame axis',
                                        legendgrouptitle=dict(text='WISPR_frame axis, longest '
                                                                   'for z, shortest for y')
                                        )
                           )
    fun_axes_traces.append(go.Scatter3d(
                                        x=np.array([WISPR_pos[0], z_sc[0]], dtype=float),
                                        y=np.array([WISPR_pos[1], z_sc[1]], dtype=float),
                                        z=np.array([WISPR_pos[2], z_sc[2]], dtype=float), mode='lines',
                                        opacity=0.6, line=dict(color='black', width=5), legendgroup='frame axis',
                                        legendgrouptitle=dict(text='WISPR_frame axis, longest '
                                                                   'for z, shortest for y')
                                        )
                           )
    return fun_fov_traces, fun_axes_traces


def plot_par(fun_state, timestr, time_format='%Y-%m-%dT%H:%M:%S.%f'):
    """
    Note that this function takes it into account that WISPR camera is not located at origin of PSP in SC frame.
    :param fun_state: The fun_state object produced by generate_par_1().
    :param timestr: (str) the time of the figure
    :param time_format: (str) the format of the time string, such as 'yyyy-MM-DDThh:mm:ss'.
    :return: plotly_trace are both the 'trace' type introduced in package 'plotly'(https://plotly.com/python).
    """
    target_pos = coor_transform(fun_state, timestr, time_format=time_format)
    target_pos[:, 1:4, 0] = target_pos[:, 1:4, 0] + WISPR_pos[:]
    target_pos[:, 1:4, 1] = target_pos[:, 1:4, 1] + WISPR_pos[:]
    plotly_trace = []
    temp_pos = np.zeros((3, 1000), dtype=float)
    for fun_i in range(len(target_pos[:, 0, 0])):
        for fun_j in range(1000):
            temp_pos[0, fun_j] = target_pos[fun_i, 1, 0] + fun_j/1000 * (target_pos[fun_i, 1, 1]
                                                                         - target_pos[fun_i, 1, 0])
            temp_pos[1, fun_j] = target_pos[fun_i, 2, 0] + fun_j / 1000 * (target_pos[fun_i, 2, 1]
                                                                           - target_pos[fun_i, 2, 0])
            temp_pos[2, fun_j] = target_pos[fun_i, 3, 0] + fun_j / 1000 * (target_pos[fun_i, 3, 1]
                                                                           - target_pos[fun_i, 3, 0])
        plotly_trace.append(
                            go.Scatter3d(
                                        x=temp_pos[0], y=temp_pos[1], z=temp_pos[2], mode='lines',
                                        opacity=0.6, line=dict(width=5),
                                        legendgroup='par',
                                        legendgrouptitle=dict(text='particles'),
                                        name=str(fun_i+1)+': raidus = ' +str(format(fun_state[fun_i, 0, 0], '.2e'))+'m',
                                        text=str(fun_i+1)
                                        )
        )
    return plotly_trace


def is_in_fov(fun_update):
    total_num = copy.deepcopy(fun_update[0])
    for fun_j in range(total_num):
        if fun_update[2][fun_j, 0, 0] < -960 or fun_update[2][fun_j, 0, 1] < -1024:
            np.delete(fun_update[2], np.s_[fun_j], axis=0)
            np.delete(fun_update[3], np.s_[fun_j], axis=0)
            np.delete(fun_update[4], np.s_[fun_j], axis=0)
            fun_update[0] = fun_update[0] - 1
    return fun_update


def brightness_diffusion(update_info):
    # update_info = is_in_fov(old_info)
    norm_step_array = np.zeros_like(update_info[-1])
    for fun_j in range(update_info[0]):
        norm_step_array[fun_j, :] = np.sqrt((update_info[2][fun_j, :, 0] - update_info[2][fun_j, 0, 0])**2 +
                                            (update_info[2][fun_j, :, 1] - update_info[2][fun_j, 0, 1])**2)
    for fun_j in range(update_info[0]):
        norm_step_array[fun_j, :] = norm_step_array[fun_j, :] / norm_step_array[fun_j, -1]
    norm_brightness = np.zeros_like(update_info[5])
    fun_fig = plt.figure(figsize=(9, 9))
    fun_ax = fun_fig.add_subplot(111)
    for fun_j in range(update_info[0]):
        norm_brightness[fun_j] = update_info[5][fun_j] / update_info[5][fun_j, 0]
        # fun_ax.plot(norm_step_array, norm_brightness[fun_j], '--', color='tab:blue', alpha=0.5)
    norm_brightness_1d = copy.deepcopy(norm_brightness.flatten())
    norm_step_array_1d = copy.deepcopy(norm_step_array.flatten())
    # mean_brightness = norm_brightness.mean(axis=0)
    # norm_brightness_log = np.log(norm_brightness)
    # brightness_log_std = norm_brightness.std(axis=0)
    xy = np.vstack([norm_step_array_1d, norm_brightness_1d])
    z = gaussian_kde(xy)(xy)
    brightness_std = norm_brightness.std(axis=0)
    # for fun_j in range(update_info[0]-1):
    #     fun_ax.scatter(norm_step_array, norm_brightness[fun_j], c=brightness_std, s=15, cmap='Spectral_r')
    fun_ax.scatter(norm_step_array_1d, norm_brightness_1d, c=z, s=15, cmap='Spectral_r')
    fun_ax.set_xlim(0, 1)
    fun_ax.set_xlabel('Normalized Distance from the Left End of the Streak', fontsize='15')
    fun_ax.set_ylabel('B/B_0', fontsize='15')
    fun_ax.tick_params(labelsize=15)
    plt.show()


def main_function():
    my_path = 'D://Microsoft Download/Formal Files/data file/FITS/WISPR-I_ENC07_L3_FITS/20210112/psp_L3_wispr_' \
              '20210112T213015_V1_1221.fits'
    data, header = read_file(my_path)[0]
    my_par_num = 5
    if generate_flag is 1:
        my_velocity = np.array([10, 0.40, 0.1968, -0.8951], dtype=float)
        my_state, my_speed = generate_par_1(my_par_num, np.array([1, 10], dtype=float), np.array([1e-7, 4e-5], dtype=float),
                                            my_velocity, header['XPOSURE']/header['NSUMEXP'])
    elif generate_flag is 2:
        my_origin = np.array([1.00, -0.5, 1.353914], dtype=float)
        my_axis = np.array([0.705395, 0.4, -0.585165], dtype=float)
        my_radius = np.array([1e-7, 1e-5], dtype=float)
        my_speed_range = np.array([0.1, 4], dtype=float)
        my_state, my_vel_vec, my_vel_mag = generate_par_2(my_par_num, my_origin, my_axis, 60, my_radius,
                                                          my_speed_range, header['XPOSURE']/header['NSUMEXP'],
                                                          header['DATE-BEG'])
    else:
        my_state = generate_par_3(1)
    my_update, my_trace = par_in_camera(my_par_num, my_state, header['XPOSURE']/header['NSUMEXP'],
                                        header['XPOSURE'], header['DATE-BEG'])
    brightness_diffusion(my_update)
    # my_data = np.zeros((2048, 1920), dtype=float)
    # my_data = update_fig(my_update, my_data)
    # a_map = sunpy.map.Map(data, header)
    # # my_colormap = copy.deepcopy(a_map.cmap)
    # my_fig = plt.figure(figsize=(9, 9))
    # ax = plt.subplot(111)
    # # output_data = pd.DataFrame(my_data)
    # # writer = pd.ExcelWriter('D://Desktop/data.xlsx')
    # # output_data.to_excel(writer, 'page_1', float_format='%.3e')
    # # writer.save()
    # norm_SL = colors.SymLogNorm(linthresh=0.001 * 1e-10, linscale=0.1 * 1e-10, vmin=-0.0038 * 1e-10, vmax=0.14 * 1e-10)
    # plt.imshow(my_data, cmap='viridis', norm=norm_SL)
    # ax.set_xlabel('x-axis [pixel]', fontsize=15)
    # ax.set_ylabel('y-axis [pixel]', fontsize=15)
    # ax.set_title(header['DATE-BEG'], fontsize=15)
    # ax.tick_params(labelsize=15)
    # my_mappabel = matplotlib.cm.ScalarMappable(cmap='viridis', norm=norm_SL)
    # plt.colorbar(label='[MSB]')
    #
    plotly_trace_1 = add_psp_3d_model(np.zeros([3, 1], dtype=float), rot_theta=0, scale=transfer_factor)
    plotly_trace_2, plotly_trace_3 = plot_FOV_and_FrameAxes(header['DATE-BEG'])
    plotly_trace_4 = plot_par(my_state, header['DATE-BEG'])
    plotly_fig = go.Figure()
    plotly_fig.add_trace(plotly_trace_1)
    for i in range(len(plotly_trace_2)):
        plotly_fig.add_trace(plotly_trace_2[i])
    for i in range(len(plotly_trace_4)):
        plotly_fig.add_trace(plotly_trace_4[i])
    plotly_fig.update_layout(title_text="Spacecraft Frame" + ' ' + header['DATE-BEG'],
                             title_font_size=30)
    plotly_fig.update_layout(scene=dict(
                                        xaxis_title='', yaxis_title='', zaxis_title='',
                                        xaxis=dict(showaxeslabels=False, showbackground=False,
                                        showgrid=False, showline=False, showticklabels=False),
                                        yaxis=dict(showaxeslabels=False, showbackground=False,
                                                   showgrid=False, showline=False, showticklabels=False),
                                        zaxis=dict(showaxeslabels=False, showbackground=False,
                                                   showgrid=False, showline=False, showticklabels=False)
                                        ))
    plotly_fig.show()
    plt.show()


main_function()
