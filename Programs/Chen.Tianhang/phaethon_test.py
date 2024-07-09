import spiceypy as spice
import numpy as np
import scipy.special as spe
import furnsh_all_kernels
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

phaethon_id = '2003200'
phaethon_radius_scale = 6.25 / 3 * 1e3  # unit: m
phaethon_geo_albedo = 0.1066
f = 19.8e-3
h = 6.626e-34  # Planck constant
c_0 = 3e8  # the velocity of light in vacuum
k_B = 1.38e-23  # Boltzmann constant
R_s = 6.955e8  # the average radius of SUN.
AU = 1.496e8  # unit: km


def get_para(time_str, exp_period, the_datetime, time_format='%Y-%m-%dT%H:%M:%S', flag=True):
    if flag:
        epoch_time = spice.datetime2et(datetime.strptime(time_str, time_format))
    else:
        epoch_time = spice.datetime2et(the_datetime)
    time_plus = datetime.strptime(time_str, time_format) + timedelta(seconds=exp_period)
    epoch_time_plus = spice.datetime2et(time_plus)
    phaethon_pos_hci = spice.spkpos(phaethon_id, epoch_time, 'SPP_HCI', 'NONE', 'SUN')
    phaethon_pos_hci_plus = spice.spkpos(phaethon_id, epoch_time_plus, 'SPP_HCI', 'NONE', 'SUN')
    phaethon_vel_hci = np.array((phaethon_pos_hci_plus[0] - phaethon_pos_hci[0]) / exp_period)
    psp_pos_hci = spice.spkpos('spp', epoch_time, 'SPP_HCI', 'NONE', 'SUN')
    psp_pos_hci_plus = spice.spkpos('spp', epoch_time_plus, 'SPP_HCI', 'NONE', 'SUN')
    psp_vel_hci = np.array((psp_pos_hci_plus[0] - psp_pos_hci[0]) / exp_period)
    # unit of length above: kilometer, unit of time above: second
    rela_pos_hci = np.array((phaethon_pos_hci[0] - psp_pos_hci[0]) * 1e3, dtype=float)
    rela_vel_hci = (phaethon_vel_hci - psp_vel_hci) * 1e3
    print(np.sqrt(np.sum(rela_pos_hci**2))/phaethon_radius_scale)
    # unit of length changes to meter.
    the_elongation = np.arccos(np.sum(rela_pos_hci * (0 - psp_pos_hci[0]))
                               / np.sqrt(np.sum(rela_pos_hci**2)) / np.sqrt(np.sum(psp_pos_hci[0]**2))) * 180 / np.pi
    rela_pos_wispr_o, _ = spice.spkcpt(rela_pos_hci[0:3],
                                       'SUN', 'SPP_HCI', epoch_time, 'SPP_WISPR_OUTER',
                                       'OBSERVER', 'NONE', 'SUN')
    rela_pos_vector = rela_pos_wispr_o[0:3]
    rela_vel_wispr_o, _ = spice.spkcpt(rela_vel_hci[0:3],
                                       'SUN', 'SPP_HCI', epoch_time, 'SPP_WISPR_OUTER',
                                       'OBSERVER', 'NONE', 'SUN')
    rela_vel_vector = rela_vel_wispr_o[0:3]

    return psp_pos_hci, phaethon_pos_hci, the_elongation


def spectrum(wavelength):
    """
    :param wavelength: unit: m
    :return: the irradience of sun
    """
    irradience_0 = 2 * h * c_0**2 / wavelength**5 / (np.exp(h*c_0/wavelength/k_B/5775)-1) * 25e-9 * np.pi
    # the factor 'pi' comes from Lambert's cosine law.
    return irradience_0


def scattering_intensity_6(elongation, width, d_to_obs, velocity, obs_time, time_format='%Y-%m-%dT%H:%M:%S'):
    """
        This function refers to Mie scattering theory and uses PyMieScatt package.
        :param elongation: the elongation (unit: degree)
        :param width: the width of the streak (unit: pixel)
        :param d_to_obs: the distance from particle to observer (unit: m)
        :param velocity: [1*3 ndarray]the velocity of particle (unit: m/s) in camera frame(i.e. WISPR-Inner frame)
        :param obs_time: the observing time of the map
        :param time_format: time string's format
        :return: the scattering intensity (unit: MSB)
        """
    A_aperture = 51e-6  # unit: m^2
    A_pixel = 10e-6 ** 2  # unit: m^2
    deg_to_rad = np.pi / 180
    qe_transmission = 0.24
    wavelength = np.linspace(475, 725, 11) * 1e-9  # 480 - 770 nm, unit of it: m
    elongation_rad = elongation * deg_to_rad
    etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
    PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
    psp_distance = np.sqrt(PSP_pos[0] ** 2 + PSP_pos[1] ** 2 + PSP_pos[2] ** 2) * 1e3
    # unit in spice is km, while we require it to be m.
    r_to_SUN = np.sqrt(d_to_obs ** 2 + psp_distance ** 2 - 2 * d_to_obs * psp_distance * np.cos(elongation_rad))
    sin_scattering_theta = psp_distance * np.sin(elongation_rad) / r_to_SUN
    albedo = phaethon_geo_albedo
    photon_num = 0
    radius = d_to_obs / f * width * 20e-6
    for fun_i in range(wavelength.size):
        alpha = 2 * np.pi * radius / wavelength[fun_i]
        bessel = spe.jv(1, alpha * sin_scattering_theta)
        bessel_length = np.sqrt(bessel.real ** 2 + bessel.imag ** 2)
        sigma = radius ** 2 * bessel_length ** 2 / np.abs(sin_scattering_theta) ** 2 + albedo * radius ** 2 / 4
    # s_1, s_2 = ps.MieS1S2(m, alpha, np.sqrt(1 - sin_scattering_theta ** 2))
    # i_1 = s_1.real ** 2 + s_1.imag ** 2
    # i_2 = s_2.real ** 2 + s_2.imag ** 2
    # sigma = wavelength ** 2 / 8 / np.pi ** 2 * (i_1 + i_2)
        the_irradiance = spectrum(wavelength[fun_i])
        max_speed = np.maximum(np.abs(velocity[0]), np.abs(velocity[1]))
        intensity_W = sigma * the_irradiance * R_s**2 / r_to_SUN**2 * A_aperture / d_to_obs**2
        # photon_num = photon_num + intensity_W * 10e-6 * d_to_obs / f / velocity / h / c_0 * wavelength[fun_i]
        photon_num = photon_num + intensity_W * 10e-6 * d_to_obs / f / max_speed / h / c_0 * wavelength[fun_i]
    total_electron_num = photon_num * qe_transmission
    intensity_DN = total_electron_num / 2.716
    intensity_MSB = intensity_DN / 48.64 * 3.93e-1

    return intensity_MSB


the_time = '2021-01-18T00:04:00'
date_time_0 = datetime.strptime(the_time, '%Y-%m-%dT%H:%M:%S')
interval = 1
fig_1 = plt.figure(figsize=(9, 9))
ax = fig_1.add_subplot(111, projection='3d')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')
# ax.set_xlim(-1024, 1024)
# ax.set_ylim(-960, 960)
ax.scatter(0, 0, 0, marker='o', color='orange', label='Sun')
for fun_i in range(0, 3600*24*5, 3600*2):
    date_time = date_time_0 + timedelta(seconds=fun_i)
    psp_pos, phaethon_pos, elo = get_para(the_time, interval, date_time, flag=False)
    if fun_i is not 0:
        ax.scatter(psp_pos[0][0]/AU, psp_pos[0][1]/AU, psp_pos[0][2]/AU, color='black')
        ax.scatter(phaethon_pos[0][0]/AU, phaethon_pos[0][1]/AU, phaethon_pos[0][2]/AU, color='red')
    else:
        fov_outer = spice.getfov(-96120, 4)
        for i_edge in range(4):
            edge_outer1, _ = spice.spkcpt(fov_outer[4][i_edge], 'SPP', 'SPP_WISPR_OUTER', spice.datetime2et(date_time),
                                          'SPP_HCI', 'OBSERVER', 'NONE', 'SPP')
            edge_motion = edge_outer1[0:3] * 5e7 + psp_pos[0]
            if i_edge is not 0:
                ax.plot([psp_pos[0][0]/AU, edge_motion[0]/AU], [psp_pos[0][1]/AU, edge_motion[1]/AU],
                        [psp_pos[0][2]/AU, edge_motion[2]/AU], color='green')
            else:
                ax.plot([psp_pos[0][0]/AU, edge_motion[0]/AU], [psp_pos[0][1]/AU, edge_motion[1]/AU],
                        [psp_pos[0][2]/AU, edge_motion[2]/AU], color='green', label='FOV of WISPR_Outer')
        ax.scatter(psp_pos[0][0]/AU, psp_pos[0][1]/AU, psp_pos[0][2]/AU, color='black', label='PSP')
        ax.scatter(phaethon_pos[0][0]/AU, phaethon_pos[0][1]/AU, phaethon_pos[0][2]/AU, color='red', label='Phaethon')
    # pos, vel, elo = get_para(the_time, interval, date_time, flag=False)
    # print(pos[2])
    # x = pos[0] / pos[2] * f / 10e-6
    # y = pos[1] / pos[2] * f / 10e-6
    # ax.scatter(x, y, color='r')
plt.legend()
plt.show()
# # the_width = phaethon_radius_scale / pos[2] * f / 10e-6
# # brightness = scattering_intensity_6(elo, the_width, np.sqrt(np.sum(pos**2)), vel, the_time)
# print(the_width, brightness)



