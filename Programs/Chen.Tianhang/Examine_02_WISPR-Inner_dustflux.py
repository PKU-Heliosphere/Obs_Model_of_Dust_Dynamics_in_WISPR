"""
This code is for particle count simulation(calculate the number of dust particles that could possibly be detected by
 WISPR-Inner).
 The model of dust refers to Mann, et al. 2004 & Mann, et al. 2021
"""
import numpy as np
import scipy
import spiceypy as spice
import scattering_intensity as cal_sca
from datetime import datetime
import warnings


def cal_flux(method=1):
    count_flux = 0.006  # unit: s^-1
    velocity_r = 9e4  # unit: m/s
    conversion_factor = 3  # unit: m^2
    area_ccd = 2048 * 1920 * 1e-10  # the size of each pixel is 10 μm.
    f = 28e-3  # the focal distance of WISPR_Inner is 28 mm.
    exp_time = 3 * 5
    obs_time = '2018/11/9T17:33:00'
    radius_particle = 120e-9  # the average radius of simulated particles is 120 nm.
    background_intensity = 5e-14  # the intensity of background is 0.05 pMSB.
    # the time of single WISPR_Inner exposure is approximately 3 s. Each fig output is the sum of 5 exposures.
    distance_min = 0
    distance_max = 1
    the_intensity = cal_sca.scattering_intensity_2(90, radius_particle, distance_max / 1e3,
                                                   obs_time, time_format='%Y/%m/%dT%H:%M:%S')

    while the_intensity > background_intensity:
        distance_max = distance_max + 0.1
        the_intensity = cal_sca.scattering_intensity_2(90, radius_particle, distance_max / 1e3,
                                                       obs_time, time_format='%Y/%m/%dT%H:%M:%S')

    if method == 1:
        num_density = count_flux / conversion_factor / velocity_r
        print(num_density)
        fov_volume = 1/3 * (distance_max - distance_min) * (area_ccd * distance_min**2 / f ** 2 +
                                                            area_ccd * distance_max**2 / f ** 2 +
                                                            area_ccd * distance_min * distance_max / f ** 2)
        total_num = num_density * fov_volume
    elif method == 2:
        particle_flux = count_flux / conversion_factor
        fov_area = 1/2 * (distance_max - distance_min) * (2048e-5 * distance_min / f + 2048e-5 * distance_max / f)
        total_num = particle_flux * fov_area * exp_time
    else:
        warnings.warn('The method does not exist!', UserWarning)
        return -1

    return total_num


result_1 = cal_flux(method=1)
result_2 = cal_flux(method=2)
print(result_1, result_2)
# a i shi te lu
