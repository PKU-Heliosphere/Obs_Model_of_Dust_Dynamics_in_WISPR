"""
This script is for scattering intensity/brightness calculating.
"""
import scipy.special as spe
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import special as sp

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
    bessel = sp.jv(1, alpha * np.sin(theta))
    bessel_length = np.abs(bessel.real**2 + bessel.imag**2)
    # bessel_length = np.abs(sp.jv(1, alpha * np.sin(theta)))

    # sigma = radius ** 2 * abs(sp.jv(1, alpha * np.sin(np.deg2rad(scatter_angle)))) ** 2 / \
    #         abs(np.sin(np.deg2rad(scatter_angle))) ** 2 + albedo * a ** 2 / 4  # m^2

    sigma = radius**2 * bessel_length**2 / np.abs(np.sin(theta))**2 + albedo * radius**2 / 4
    # print(sigma,end=' ')
    irradiance_0 = 1  # the unit of it is F_O(the irradiance of sun at 1AU)
    from_arcsec_to_deg = 1/3600
    resolution = 203 * from_arcsec_to_deg # unit: arcsec -> deg
    distace_dust_to_SC = 8e0  # unit:m
    Radius_effec = distace_dust_to_SC * np.tan(resolution/2 /180*np.pi)
    S_effective = (Radius_effec) ** 2

    S_dust =  np.pi * radius**2
    print(S_effective, S_dust)
    if  S_dust < S_effective:
        S_effective = S_effective
    if S_dust >= S_effective:
        S_effective = S_dust
    irradiance_F_0 = (1 / distance)**2 * irradiance_0 * sigma * S_dust/ S_effective**2 # the unit is F_0 / sr ??????????????????
    ########################################
    # NO!!!!!!!!!!!!!!
    # The unit of irradiance is not the F_0 / sr but the F_0 * m^2 /sr !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # fuck!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ########################################
    irradiance_MSB = irradiance_F_0 * 4.50e-16 / 6.61e-12  # (the unit transfers to MSB)
    return irradiance_MSB
the_intensity_arr = []
a_theta_arr = range(30,120)
for a_theta in a_theta_arr:
    a_theta = a_theta / 180 * np.pi
    a_radius = 1e-4 # unit: m
    a_distance = 0.25
    the_intensity = scattering_intensity_1(a_theta, a_radius, a_distance)
    the_intensity_arr = np.append(the_intensity_arr,the_intensity)
# print(np.min(the_intensity_arr))
plt.plot(a_theta_arr,the_intensity_arr)
plt.yscale('log')
plt.xlabel('a_theta')
plt.ylabel('MSB')
plt.show()