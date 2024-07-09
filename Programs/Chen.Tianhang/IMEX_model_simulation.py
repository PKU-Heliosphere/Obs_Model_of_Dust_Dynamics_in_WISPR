"""
This is a simplified version of IMEX code with Python.
"""
from matplotlib import pyplot as plt
import spiceypy as spice
import numpy as np
import scipy as sp
from copy import deepcopy
import PyMieScatt as Mie


def dust_emission(comet_id, timerange):
    C_r = 7.6e-5
    the_body = spice.spkpos()
    beta = C_r * Q_pr * (g / p_mass)
    V_Agarwal = 500 * np.sqrt(beta) / distance_hc**3


def cal_accel(p_pos, p_radius, the_time):
    c = 29935414
    Q_pr = Mie.MieQ()
    all_body_id = np.array([1, 10, 13, 131, 313, 21, 21, 31, 51, 21, 97], dtype=int)
    all_GM = np.zeros([9], dtype=float)
    all_pos_vec = np.zeros([9, 3], dtype=float)
    all_distance_p = np.zeros([9], dtype=float)
    for i in range(9):
        body_position = spice.spkpos(all_body_id[i], the_time, 'J2000', 'NONE', 'SSC')
        body_position = np.array(body_position, dtype=float)
        all_pos_vec[i] = body_position[0:3] - p_pos
        all_distance_p[i] = all_pos_vec[i, 0]**2 + all_pos_vec[i, 1]**2 + all_pos_vec[i, 2]**2

        all_GM[i] = spice.spkgeo()

    for i in range(9):
        accel_1 = all_GM[i] / all_distance_p[i] / np.sqrt(all_distance_p[i]) * all_pos_vec[i]
        accel_2 = - S_SUN * g * Q_pr / all_distance_p[i]**2 / np.sqrt(all_distance_p[i])/ c * all_pos_vec[i]
        accel_3 = solar_wind?

