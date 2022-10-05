"""
This script is for scattering intensity/brightness calculating.
"""
import copy

import scipy.special as spe
import numpy as np
import spiceypy as spice
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
import PyMieScatt as ps
import copy
import furnsh_all_kernels

h = 6.626e-34  # Planck constant
c_0 = 3e8  # the velocity of light in vacuum
k_B = 1.38e-23  # Boltzmann constant
R_s = 6.955e8  # the average radius of SUN.
AU = 1.496e8  # unit: m


def scattering_intensity_1(theta, radius, distance):
    """
    :param theta: scattering angle, not elongation. UNIT: rad
    :param radius: the size of single dust particle. UNIT: m
    :param distance: the distance from SUN to the particle. UNIT: AU
    :return: the scattering intensity.
    The algorithm refers to the formula from Mann(1992).
    """

    albedo = 0.25
    wavelength = 490 * 1e-9  # 490 - 740 nm
    alpha = 2 * np.pi * radius / wavelength
    bessel = spe.jv(1, alpha * np.sin(theta))
    bessel_length = np.sqrt(bessel.real**2 + bessel.imag**2)
    sigma = radius**2 * bessel_length**2 / np.abs(np.sin(theta))**2 + albedo * radius**2 / 4
    irradiance_0 = 1  # the unit of it is F_O(the irradiance of sun at 1AU)
    irradiance_F_0 = (1 / distance)**2 * irradiance_0 * sigma / np.pi / radius**2  # the unit is F_0 / sr ??????????????????
    ########################################
    # NO!!!!!!!!!!!!!!
    # The unit of irradiance is not the F_0 / sr but the F_0 * m^2 /sr !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # fuck!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ########################################
    irradiance_MSB = irradiance_F_0 * 4.50 / 6.61 * 1e-4  # (the unit transfers to MSB)
    return irradiance_MSB


def scattering_intensity_2(elongation, radius, d_to_obs, obs_time, time_format='%Y%m%dT%H%M%S'):
    """
    some changes to the parameters.
    :param elongation: elongation (unit: degree)
    :param radius: the diameter of the particle (unit: m)
    :param d_to_obs: the distance from particle to observer (unit: km)
    :param obs_time: the time when observing the particles
    :param time_format: the format of the 'obs_time' input
    :return: the scattering intensity (unit: Mean Solar Brightness)
    """
    f = 28e-6   # unit: km
    A_pixel = 0.01e-3**2
    deg_to_rad = np.pi / 180
    albedo = 0.25
    wavelength = 490 * 1e-9  # 490 - 740 nm
    msb_to_w = 2.3e7    # ( 1 MSB = 2.3*10^7 W /(m^2 * sr))
    solar_constant = 1.362e3   # W/m^2
    alpha = 2 * np.pi * radius / wavelength
    elongation_rad = elongation * deg_to_rad
    etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
    PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
    psp_distance = np.sqrt(PSP_pos[0]**2 + PSP_pos[1]**2 + PSP_pos[2]**2)
    r_to_SUN = np.sqrt(d_to_obs**2 + psp_distance**2 - 2 * d_to_obs * psp_distance * np.cos(elongation_rad))
    bessel = spe.jv(1, alpha * np.sin(elongation_rad) * psp_distance / r_to_SUN)
    bessel_length = np.sqrt(bessel.real ** 2 + bessel.imag ** 2)
    sigma = radius**2 * bessel_length**2 / np.abs(np.sin(elongation_rad) * psp_distance / r_to_SUN)**2 \
        + albedo * radius**2 / 4
    S_dust = np.pi * (f * radius / d_to_obs)**2
    intensity_W = sigma * solar_constant * (1 * AU / r_to_SUN)**2 / np.pi / radius**2 * S_dust / A_pixel
    intensity_MSB = intensity_W / msb_to_w
    return intensity_MSB


def scattering_intensity_3(m, elongation, radius, d_to_obs, obs_time, time_format='%Y%m%dT%H%M%S'):
    """
    This function refers to Mie scattering theory and uses PyMieScatt package.
    :param m: the complex refractive index of the particle
    :param elongation: the elongation
    :param radius: the radius of the particle (unit: m)
    :param d_to_obs: the distance from particle to observer (unit: km)
    :param obs_time: the observing time of the map
    :param time_format: time string's format
    :return: the scattering intensity (unit: MSB)
    """
    f = 28e-6  # unit: km
    A_pixel = 0.01e-3 ** 2
    deg_to_rad = np.pi / 180
    AU = 1.496e8
    wavelength = 490*1e-9  # 490 - 740 nm
    alpha = 2 * np.pi * radius / wavelength
    elongation_rad = elongation * deg_to_rad
    etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
    PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
    psp_distance = np.sqrt(PSP_pos[0] ** 2 + PSP_pos[1] ** 2 + PSP_pos[2] ** 2)
    r_to_SUN = np.sqrt(d_to_obs ** 2 + psp_distance ** 2 - 2 * d_to_obs * psp_distance * np.cos(elongation_rad))
    sin_scattering_theta = psp_distance * np.sin(elongation_rad) / r_to_SUN
    s_1, s_2 = ps.MieS1S2(m, alpha, np.sqrt(1 - sin_scattering_theta**2))
    i_1 = s_1.real**2 + s_1.imag**2
    i_2 = s_2.real**2 + s_2.imag**2
    sigma = wavelength**2 / 8 / np.pi**2 * (i_1 + i_2)
    S_dust = np.pi * (f * radius / d_to_obs) ** 2
    F_0 = 1
    intensity_F_0 = sigma * F_0 * (1 * AU / r_to_SUN) ** 2 / np.pi / radius ** 2 * S_dust / A_pixel
    intensity_MSB = intensity_F_0 * 4.50 / 6.61 * 1e-4
    return intensity_MSB


def scattering_intensity_4(m, elongation, radius, d_to_obs, obs_time, time_format='%Y%m%dT%H:%M:%S'):
    """
    This function refers to Mie scattering theory and uses PyMieScatt package.
    :param m: the complex refractive index of the particle
    :param elongation: the elongation
    :param radius: the radius of the particle (unit: m)
    :param d_to_obs: the distance from particle to observer (unit: m)
    :param obs_time: the observing time of the map
    :param time_format: time string's format
    :return: the scattering intensity (unit: MSB)
    """
    f = 28e-3  # unit: m
    A_pixel = 10e-6 ** 2  # unit: m^2
    deg_to_rad = np.pi / 180
    wavelength = 490*1e-9  # 490 - 740 nm, unit of it: m
    alpha = 2 * np.pi * radius / wavelength
    elongation_rad = elongation * deg_to_rad
    etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
    PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
    psp_distance = np.sqrt(PSP_pos[0] ** 2 + PSP_pos[1] ** 2 + PSP_pos[2] ** 2) * 1e3
    # unit in spice is km, while we require it to be m.
    r_to_SUN = np.sqrt(d_to_obs ** 2 + psp_distance ** 2 - 2 * d_to_obs * psp_distance * np.cos(elongation_rad))
    sin_scattering_theta = psp_distance * np.sin(elongation_rad) / r_to_SUN
    albedo = 0.25
    bessel = spe.jv(1, alpha * sin_scattering_theta)
    bessel_length = np.sqrt(bessel.real ** 2 + bessel.imag ** 2)
    sigma = radius ** 2 * bessel_length ** 2 / np.abs(sin_scattering_theta) ** 2 + albedo * radius ** 2 / 4
    # s_1, s_2 = ps.MieS1S2(m, alpha, np.sqrt(1 - sin_scattering_theta ** 2))
    # i_1 = s_1.real ** 2 + s_1.imag ** 2
    # i_2 = s_2.real ** 2 + s_2.imag ** 2
    # sigma = wavelength ** 2 / 8 / np.pi ** 2 * (i_1 + i_2)
    F_0 = 1
    intensity_F_0 = sigma * F_0 * (1 * AU / r_to_SUN) ** 2 * f**2 / d_to_obs**2 / A_pixel
    intensity_MSB = intensity_F_0 * 4.50 / 6.61 * 1e-4
    return intensity_MSB


def spectrum(wavelength):
    """
    :param wavelength: unit: m
    :return: the irradience of sun
    """
    irradience_0 = 2 * h * c_0**2 / wavelength**5 / (np.exp(h*c_0/wavelength/k_B/5775)-1) * 10e-9 * np.pi
    # the factor 'pi' comes from Lambert's cosine law.
    return irradience_0


def scattering_intensity_5(elongation, radius, d_to_obs, exp_time, streak_len, obs_time, time_format='%Y%m%dT%H:%M:%S'):
    """
        This function refers to Mie scattering theory and uses PyMieScatt package.
        :param elongation: the elongation
        :param radius: the radius of the particle (unit: m)
        :param d_to_obs: the distance from particle to observer (unit: m)
        :param exp_time: the single exposure time of this figure. (unit: s)
        :param streak_len: the maximum of X and Y lengths of the chosen streak (unit: pixel)
        :param obs_time: the observing time of the map
        :param time_format: time string's format
        :return: the scattering intensity (unit: MSB)
        """
    f = 28e-3  # unit: m
    A_aperture = 42e-6  # unit: m^2
    A_pixel = 10e-6 ** 2  # unit: m^2
    deg_to_rad = np.pi / 180
    qe_transmission = 0.24
    wavelength = np.linspace(480, 770, 30, endpoint=True) * 1e-9  # 480 - 770 nm, unit of it: m
    elongation_rad = elongation * deg_to_rad
    etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
    PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
    psp_distance = np.sqrt(PSP_pos[0] ** 2 + PSP_pos[1] ** 2 + PSP_pos[2] ** 2) * 1e3
    # unit in spice is km, while we require it to be m.
    r_to_SUN = np.sqrt(d_to_obs ** 2 + psp_distance ** 2 - 2 * d_to_obs * psp_distance * np.cos(elongation_rad))
    sin_scattering_theta = psp_distance * np.sin(elongation_rad) / r_to_SUN
    albedo = 0.25
    photon_num = 0
    for fun_i in range(30):
        alpha = 2 * np.pi * radius / wavelength[fun_i]
        bessel = spe.jv(1, alpha * sin_scattering_theta)
        bessel_length = np.sqrt(bessel.real ** 2 + bessel.imag ** 2)
        sigma = radius ** 2 * bessel_length ** 2 / np.abs(sin_scattering_theta) ** 2 + albedo * radius ** 2 / 4
    # s_1, s_2 = ps.MieS1S2(m, alpha, np.sqrt(1 - sin_scattering_theta ** 2))
    # i_1 = s_1.real ** 2 + s_1.imag ** 2
    # i_2 = s_2.real ** 2 + s_2.imag ** 2
    # sigma = wavelength ** 2 / 8 / np.pi ** 2 * (i_1 + i_2)
        the_irradience = spectrum(wavelength[fun_i])
        intensity_W = sigma * the_irradience * R_s**2 / r_to_SUN**2 * A_aperture / d_to_obs**2
        # photon_num = photon_num + intensity_W * 10e-6 * d_to_obs / f / velocity / h / c_0 * wavelength[fun_i]
        photon_num = photon_num + intensity_W * exp_time / streak_len / h / c_0 * wavelength[fun_i]
    total_electron_num = photon_num * qe_transmission
    intensity_DN = total_electron_num / 2.716
    intensity_MSB = intensity_DN / 48.64 * 3.93e-14
    return intensity_MSB


def scattering_intensity_6(elongation, width, d_to_obs, velocity, obs_time, time_format='%Y%m%dT%H:%M:%S'):
    """
        This function refers to Mie scattering theory and uses PyMieScatt package.
        :param elongation: the elongation (unit: degree)
        :param width: the width of the streak (unit: pixel)
        :param d_to_obs: the distance from particle to observer (unit: m)
        :param velocity: the velocity of particle (unit: m/s)
        :param obs_time: the observing time of the map
        :param time_format: time string's format
        :return: the scattering intensity (unit: MSB)
        """
    f = 28e-3  # unit: m
    A_aperture = 42e-6  # unit: m^2
    A_pixel = 10e-6 ** 2  # unit: m^2
    deg_to_rad = np.pi / 180
    qe_transmission = 0.24
    wavelength = np.linspace(480, 770, 30) * 1e-9  # 480 - 770 nm, unit of it: m
    elongation_rad = elongation * deg_to_rad
    etime = spice.datetime2et(datetime.strptime(obs_time, time_format))
    PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
    psp_distance = np.sqrt(PSP_pos[0] ** 2 + PSP_pos[1] ** 2 + PSP_pos[2] ** 2) * 1e3
    # unit in spice is km, while we require it to be m.
    r_to_SUN = np.sqrt(d_to_obs ** 2 + psp_distance ** 2 - 2 * d_to_obs * psp_distance * np.cos(elongation_rad))
    sin_scattering_theta = psp_distance * np.sin(elongation_rad) / r_to_SUN
    albedo = 0.25
    photon_num = 0
    radius = d_to_obs / f * width * 20e-6
    for fun_i in range(30):
        alpha = 2 * np.pi * radius / wavelength[fun_i]
        bessel = spe.jv(1, alpha * sin_scattering_theta)
        bessel_length = np.sqrt(bessel.real ** 2 + bessel.imag ** 2)
        sigma = radius ** 2 * bessel_length ** 2 / np.abs(sin_scattering_theta) ** 2 + albedo * radius ** 2 / 4
    # s_1, s_2 = ps.MieS1S2(m, alpha, np.sqrt(1 - sin_scattering_theta ** 2))
    # i_1 = s_1.real ** 2 + s_1.imag ** 2
    # i_2 = s_2.real ** 2 + s_2.imag ** 2
    # sigma = wavelength ** 2 / 8 / np.pi ** 2 * (i_1 + i_2)
        the_irradiance = spectrum(wavelength[fun_i])
        intensity_W = sigma * the_irradiance * R_s**2 / r_to_SUN**2 * A_aperture / d_to_obs**2
        # photon_num = photon_num + intensity_W * 10e-6 * d_to_obs / f / velocity / h / c_0 * wavelength[fun_i]
        photon_num = photon_num + intensity_W * 10e-6 * d_to_obs / f / velocity / h / c_0 * wavelength[fun_i]
    total_electron_num = photon_num * qe_transmission
    intensity_DN = total_electron_num / 2.716
    intensity_MSB = intensity_DN / 48.64 * 3.93e-14
    return intensity_MSB


def scattering_intensity_7(scattering_angle, d_to_obs, d_to_sun, n_density, bulk_radius, exp_time):
    r_e = 2.81e-15  # unit: m
    f = 28e-3  # unit: m
    A_aperture = 42e-6  # unit: m^2
    A_pixel = 10e-6 ** 2  # unit: m^2
    diff_cross_section = r_e**2 / 2 * (1 + np.cos(scattering_angle/180*np.pi)**2) \
        * 4/3 * np.pi * bulk_radius**3 * n_density
    wavelength = np.linspace(480, 770, 30) * 1e-9  # 480 - 770 nm, unit of it: m
    qe_transmission = 0.24
    total_electron_num = 0
    for fun_i in range(30):
        the_irradiance = spectrum(wavelength[fun_i])
        intensity_W = diff_cross_section * the_irradiance * R_s**2 / d_to_sun**2 * A_aperture / d_to_obs**2
        total_electron_num = total_electron_num + intensity_W * exp_time / h / c_0 * wavelength[fun_i] * qe_transmission
    intensity_DN = total_electron_num / 2.716
    intensity_MSB = intensity_DN / 48.64 * 3.93e-14
    return intensity_MSB


def main_function():
    a_elongation = 5
    a_radius = 1e-4
    a_distance = 10
    # result = scattering_intensity_4(complex(1000, -0.1), a_elongation, a_radius, a_distance, '20210102T13:00:01')
    # print(result)
    radius = np.linspace(1e-7, 1e-2, 2000) * 1e6  # unit: micron
    distance = np.linspace(0.1, 20, 2000)  # unit: m
    x, y = np.meshgrid(distance, radius)
    # result = scattering_intensity_5(5, a_radius, a_distance, 10, '20210102T13:00:01')
    # print(result)
    result = scattering_intensity_5(5, y/1e6, x, 3, 500, '20210102T13:00:01')
    figure = plt.figure(figsize=(9, 9))
    ax_1 = figure.add_subplot(111)
    pcm = ax_1.contourf(x, y, result, [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8],
                        norm=colors.LogNorm(), cmap='rainbow', alpha=0.8)
    cb = figure.colorbar(pcm, ax=ax_1)
    ax_1.set_title('Observing Brightness')
    cb.set_label('[MSB]')
    ax_1.set_ylim(1e-1, 1e4)
    ax_1.set_yscale('log')
    ax_1.set_xlabel('distance from dust to PSP  [$m$]')
    ax_1.set_ylabel('radius of dust  [$\mu m$]')
    wished_radius = 10e-6 * distance / 28e-3 * 1e6  # unit:micron
    ax_1.plot(distance, wished_radius, color='black', lw=2, label='radius of the particle that fills one pixel')
    plt.legend()
    plt.show()

    # cb = figure.colorbar(pcm, ax=ax_1)
    # ax_1.set_xlabel('distance from dust to PSP  [$km$]')
    # ax_1.set_ylabel('radius of dust  [$\mu m$]')
    # ax_1 = figure.add_subplot(1, 1, 1)
    #
    # # intensity_2 = scattering_intensity_2(a_elongation, a_radius, a_distance, '20210112T030017')
    # # intensity_3 = scattering_intensity_3(complex(1.2, 0), a_elongation, a_radius, a_distance, '20210112T030017')
    # # print(intensity_2, intensity_3)
    #
    # intensity_1 = scattering_intensity_2(a_elongation, y / 1e6, x, '20210112T030017')
    #
    # # intensity_2 = np.zeros([100, 100], dtype=float)
    # # for my_dis in range(100):
    # #     for my_radius in range(100):
    # #         intensity_2[my_radius][my_dis] = scattering_intensity_3(1.2, a_elongation, radius[my_radius] / 1e6,
    # #         distance[my_dis],'20210112T030017')
    #
    # figure = plt.figure(figsize=(9, 9))
    # ax_1 = figure.add_subplot(1, 1, 1)
    #
    # # ax_2 = figure.add_subplot(1, 1, 1)
    #
    # pcm = ax_1.contourf(x, y, intensity_1, [1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8],
    #                     norm=colors.LogNorm(), cmap='rainbow')
    # cb = figure.colorbar(pcm, ax=ax_1)
    # ax_1.set_xlabel('distance from dust to PSP  [$km$]')
    # ax_1.set_ylabel('radius of dust  [$\mu m$]')
    #
    # # pcm = ax_2.pcolor(x, y, intensity_2, norm=colors.LogNorm(vmin=np.percentile(intensity_2, 1),
    # #                                                       vmax=np.percentile(intensity_2, 99)), cmap='rainbow')
    # # cb = figure.colorbar(pcm, ax=ax_2)
    # # ax_2.set_xlabel('distance from dust to PSP  [$km$]')
    # # ax_2.set_ylabel('radius of dust  [$\mu m$]')
    #
    # plt.show()

# main_function()




