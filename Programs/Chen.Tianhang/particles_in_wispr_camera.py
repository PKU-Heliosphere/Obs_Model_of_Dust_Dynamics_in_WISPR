import copy
import numpy as np
import numpy.linalg as lin
import spiceypy as spice
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def timestr2et(times, format='%Y%m%dT%H:%M:%S'):
    my_time = datetime.strptime(times, format)
    my_et = spice.datetime2et(my_time)
    return my_et
#############################################################################################
# Note that all the units of the object value in the functions below are AU, not km or m
# Now we get the exposure time of WISPR-I :01:09.761 - 00:19.072 = 50.689 s
#############################################################################################


def get_unit_vector(pos_1, pos_2):
    minus = pos_2 - pos_1
    unit_vec = minus / np.sqrt(minus[0]**2 + minus[1]**2 + minus[2]**2)
    return unit_vec


def get_plain_vector(normal_vec, radius):
    """
    :param normal_vec: the normal vector(1*3 numpy_array) of the plain and the length of it is 1.
    :param radius: the length of the plain vector
    :return: two 1*3 numpy_arrays showing the orthogonal vector group of the plain
    """
    x = - normal_vec[2]
    y = - normal_vec[2]
    z = normal_vec[0] + normal_vec[1]
    plain_vector_1 = np.array([x, y, z], dtype=float)
    plain_vector_1 = plain_vector_1 / np.sqrt(plain_vector_1[0]**2 + plain_vector_1[1]**2 + plain_vector_1[2]**2)
    plain_vector_2 = copy.deepcopy(plain_vector_1)
    plain_vector_2[0] = plain_vector_1[1] * normal_vec[2] - plain_vector_1[2] * normal_vec[1]
    plain_vector_2[1] = plain_vector_1[2] * normal_vec[0] - plain_vector_1[0] * normal_vec[2]
    plain_vector_2[2] = plain_vector_1[0] * normal_vec[1] - plain_vector_1[1] * normal_vec[0]
    plain_vector_2 = plain_vector_2 / np.sqrt(plain_vector_2[0]**2 + plain_vector_2[1]**2 + plain_vector_2[2]**2)
    plain_vector_1 = plain_vector_1 * radius
    plain_vector_2 = plain_vector_2 * radius
    return plain_vector_1, plain_vector_2


def get_cylinder_point(origin_point, plain_vec1, plain_vec2, num_of_point):
    """
    :param origin_point: the origin(1*3 numpy_array) of the plain.
    :param plain_vec1: plain_vec1 and plain_vec2 are the vector provided by get_plain_vector()
    :param plain_vec2: ...
    :param num_of_point: the number of your point , int (marks n)
    :return: n*3 numpy.arrays presenting the n points in the edge of the given cylinder.
    """
    theta = 2 * np.pi / num_of_point
    points = [None] * num_of_point
    for i in range(len(points)):
        points[i] = [float(0)] * 3
    points = np.array(points)
    for i in range(num_of_point):
        points[i] = origin_point + plain_vec1 * np.cos(i * theta) + plain_vec2 * np.sin(i * theta)
    return points


def random_velocity(num_of_point):
    """
    :param num_of_point:
    :return: the 1*n velocity
    """
    return


def get_line(start_point, unit_vector, length, step):
    """
    :param start_point: the start of the line(1*3 numpy.array)
    :param unit_vector: the unit vector of the line(1*3 numpy.array)
    :param length: the length of the line(a float number)
    :param step: the point cadence of the line(a int number, usually > 100)
    :return: a 3 * step numpy.array indicating the point group on the line.
    """
    line_point = [None] * 3
    for i in range(3):
        line_point[i] = [float(0)] * step
    line_point = np.array(line_point)
    for i in range(step):
        line_point[0, i] = start_point[0] + unit_vector[0] * length / step * i
        line_point[1, i] = start_point[1] + unit_vector[1] * length / step * i
        line_point[2, i] = start_point[2] + unit_vector[2] * length / step * i
    return line_point


spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_v300.tf')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_2018_224_2025_243_RO5_00_nocontact.alp.bc')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_001.tf')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_dyn_v201.tf')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_wispr_v002.ti')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_sclk_0865.tsc')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/naif0012.tls')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/earthstns_itrf93_201023.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20201016_20210101_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/pck00010.tpc')
# importing necessary kernels
###############################
# no time information, so I use the tls file.
# At least one SPK file needs to be loaded by SPKLEF before beginning a search, so I use the bsp file
# Insufficient ephemeris data has been loaded to compute the position of -96 (SPP) relative to 10 (SUN) ,
# so I use the ephemeris files bsp.
###############################

AU = 1.496e8
et1 = timestr2et('20201016T20:00:00')
et2 = timestr2et('20201229T20:00:00')
number = 4000
times = [(et2 - et1)/number * i + et1 for i in range(number)]
PSP_pos, light_time = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
PSP_pos = PSP_pos.T
PSP_pos = PSP_pos/AU
# getting PSP position

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')
ax.set_xlim(-0.5, -0.2)
ax.set_ylim(-0.6, -0.2)
ax.scatter(0, 0, 0, c='#FF3333', label='Sun', marker='o')
ax.text(PSP_pos[0, 2000], PSP_pos[1, 2000], PSP_pos[2, 2000], 'PSP')
ax.text(0, 0, 0, 'Sun')
ax.plot(PSP_pos[0], PSP_pos[1], PSP_pos[2])
plt.title('SPP-HCI frame')
# plotting the position of PSP

wispr_inner_parameter = spice.getfov(-96100, 4)
#######################################################
# the FOV edge of wispr is -19.2-19.2 (longitude) and -19.4-19.4(latitude), not the real helioprojective frame!!
#######################################################
WISPR_pos = PSP_pos.T[0]
for i_edge in range(4):
    edge_inner1, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', et1,
                                 'SPP_HCI', 'OBSERVER', 'NONE', 'SUN')
    edge_inner1 = edge_inner1 / AU
    edge_motion = np.zeros([3, 1000], dtype=float)
    for cadence in range(1000):
        edge_motion[:, cadence] = (edge_inner1[0:3] - WISPR_pos[0:3]) * cadence / 6000 + WISPR_pos[0:3]
    if i_edge is 3:
        ax.plot(edge_motion[0], edge_motion[1], edge_motion[2], c='green', label='FOV of WISPR_INNER')
    else:
        ax.plot(edge_motion[0], edge_motion[1], edge_motion[2], c='green')
ax.legend()
# plotting the FOV of WISPR-I

WISPR_I_z_axis = [0 * AU, 0 * AU,  0.3 * AU]
rotation_matrix = np.array([[0.82518038663773430, -0.17192917141267056, 0.53806848033103727],
                            [0.14930159834816223, 0.98506308873062820, 0.08578895005278017],
                            [-0.54478102228242442, 0.00954312516039215, 0.83852404051588747]], dtype=float)
WISPR_I_z_axis = np.array(WISPR_I_z_axis, dtype=float)
z_in_SPP, _ = spice.spkcpt(WISPR_I_z_axis, 'SPP', 'SPP_WISPR_INNER', et1, 'SPP_HCI', 'OBSERVER', 'NONE', 'SUN')
z_in_SPP = z_in_SPP / AU
##################################
# This is just a test for frames' transformation.
# z_vec = z_in_SPP[0:3] - WISPR_pos
# z_vec_length = np.sqrt(z_vec[0]**2 + z_vec[1]**2 + z_vec[2]**2)
# print(z_vec_length)
##################################
ax.plot([WISPR_pos[0], z_in_SPP[0]], [WISPR_pos[1], z_in_SPP[1]], [WISPR_pos[2], z_in_SPP[2]], c='k')
# plotting the z-axis(boresight) of the WISPR-I

assuming_pos = [-0.31 * AU, -0.345 * AU, 0.02 * AU]
assuming_pos = np.array(assuming_pos)
assuming_pos2 = [-0.39 * AU, -0.400 * AU, 0.025 * AU]
assuming_pos2 = np.array(assuming_pos2)
pre_pos = 2 * assuming_pos - assuming_pos2

my_step = 1000
vector1 = get_unit_vector(assuming_pos/AU, assuming_pos2/AU)
p_vec1, p_vec2 = get_plain_vector(vector1, 0.012)
all_points = get_cylinder_point(assuming_pos/AU, p_vec1, p_vec2, 10)
line_points = np.zeros([10, 3, my_step], dtype=float)
for i in range(10):
    line_points[i] = get_line(all_points[i], vector1, 0.06, my_step)
    ax.plot(line_points[i, 0], line_points[i, 1], line_points[i, 2])
    str_1 = str(i)
    ax.text(all_points[i, 0], all_points[i, 1], all_points[i, 2], str_1)
# assuming a motion of 10 particles
# plotting the motion of the particle in 3D figure(World frame)

vector1p = get_unit_vector(assuming_pos/AU, pre_pos/AU)
p_vec1p, p_vec2p = get_plain_vector(vector1p, 0.012)
all_points_p = get_cylinder_point(assuming_pos/AU, p_vec1p, p_vec2p, 10)
line_points_p = np.zeros([10, 3, my_step], dtype=float)
for i in range(10):
    line_points_p[i] = get_line(all_points_p[i], vector1p, 0.06, my_step)
# the elongation of the line.

ax.scatter(PSP_pos[0, 0], PSP_pos[1, 0], PSP_pos[2, 0], c='r', marker='^')
ax.text(PSP_pos[0, 0], PSP_pos[1, 0], PSP_pos[2, 0], 'observing pos')
# plotting the observing position of PSP

inner_line = np.zeros([10, 2, my_step])
temp_3D = np.zeros([3], dtype=float)
for i in range(10):
    for j in range(my_step):
        temp_3D[0] = line_points[i, 0, j] * AU
        temp_3D[1] = line_points[i, 1, j] * AU
        temp_3D[2] = line_points[i, 2, j] * AU
        temp, one_way_motion = spice.spkcpt(temp_3D, 'SUN', 'SPP_HCI', times[0],
                                            'SPP_WISPR_INNER', 'OBSERVER', 'NONE', 'SPP')
        temp_rad = spice.reclat(temp[[2, 0, 1]])
        inner_line[i, 0, j] = np.rad2deg(temp_rad[1])
        inner_line[i, 1, j] = np.rad2deg(temp_rad[2])
inner_line_p = np.zeros([10, 2, my_step])
for i in range(10):
    for j in range(my_step):
        temp_3D[0] = line_points_p[i, 0, j] * AU
        temp_3D[1] = line_points_p[i, 1, j] * AU
        temp_3D[2] = line_points_p[i, 2, j] * AU
        temp, one_way_motion = spice.spkcpt(temp_3D, 'SUN', 'SPP_HCI', times[0],
                                            'SPP_WISPR_INNER', 'OBSERVER', 'NONE', 'SPP')
        temp_rad = spice.reclat(temp[[2, 0, 1]])
        inner_line_p[i, 0, j] = np.rad2deg(temp_rad[1])
        inner_line_p[i, 1, j] = np.rad2deg(temp_rad[2])
# coordinate transformation

for i in range(10):
    inner_line_p[i, 0, 999] = (inner_line_p[i, 0, 999] - inner_line_p[i, 0, 0]) * 3 + inner_line_p[i, 0, 0]
    inner_line_p[i, 1, 999] = (inner_line_p[i, 1, 999] - inner_line_p[i, 1, 0]) * 3 + inner_line_p[i, 1, 0]
fig_2 = plt.figure(figsize=(7, 7))
ax_2 = fig_2.add_subplot(111)
for i in range(10):
    ax_2.plot(inner_line[i, 0], -inner_line[i, 1])
    ax_2.plot([inner_line_p[i, 0, 0], inner_line_p[i, 0, 999]],
              [-inner_line_p[i, 1, 0], -inner_line_p[i, 1, 999]],
              '--r', linewidth=0.5)
    str_2 = str(i)
    ax_2.text(inner_line[i, 0, 500], -inner_line[i, 1, 500], str_2)
ax_2.set_xlim(-38.4*3/4, 38.4*3/4)
ax_2.set_ylim(-38.7992*3/4, 38.7992*3/4)
rec = plt.Rectangle((-19.2, -19.3996), 38.4, 38.7992, fill=False, edgecolor='green', linestyle='--')
ax_2.add_patch(rec)
ax_2.text(-19.2, -16, 'FOV')
ax_2.set_xlabel('Longitude (Solar-X) [deg]')
ax_2.set_ylabel('Latitude (Solar-Y) [deg]')
plt.title('2020/10/16 T20:00:00')
plt.show()
# plotting the motion in the WISPR-I camera frame.
