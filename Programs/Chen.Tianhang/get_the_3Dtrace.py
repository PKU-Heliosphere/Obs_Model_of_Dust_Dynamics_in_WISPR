"""
This is a script for trace retrieving from a L3 WISPR-INNER map.
"""
import spiceypy as spice
from sunpy.net import attrs as a
import sunpy.map
import sunpy.io.fits
from sunpy.net import Fido
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.time import TimeDelta
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
import copy
import os
from datetime import datetime
import sympy
import furnsh_all_kernels


def get_vanishing_point(data_point, need_test=False):
    """
    This function refers to the Gradient Descent Algorithm.
    :param data_point: n * 2 array
    :param need_test: if you want to examine the descent speed and check if it converges.
    :return: the center point of all point input. 1 * 2 array
    """
    fun_x = sympy.symbols('x')
    fun_y = sympy.symbols('y')
    expense = 0
    vanish_point = np.zeros([2], dtype=float)

    for i_inner in range(data_point[:, 0].size - 1):
        expense = expense + sympy.sqrt((fun_x - data_point[i_inner, 0])**2 + (fun_y - data_point[i_inner, 1])**2)
        vanish_point = vanish_point + data_point[i_inner]
    vanish_point = vanish_point / data_point[:, 0].size
    descent_rate = 1
    d_x = sympy.diff(expense, fun_x)
    d_y = sympy.diff(expense, fun_y)
    for descent_count in range(300):
        vanish_point[0] = vanish_point[0] - descent_rate * d_x.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})
        vanish_point[1] = vanish_point[1] - descent_rate * d_y.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})
        if need_test is True:
            if descent_count is 0:
                expense_fun = np.array([expense.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})], dtype=float)
            else:
                expense_fun = np.insert(expense_fun, descent_count, values=np.array(
                    [expense.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})]), axis=0)
    if need_test is True:
        i_fun = np.linspace(1, 300, 300)
        p_1 = plt.figure()
        test_ax = p_1.add_subplot(111)
        test_ax.plot(i_fun, expense_fun)
        test_ax.set_xlabel('descent count')
        test_ax.set_ylabel('cost'
                           ' function')
    return vanish_point

AU = 1.496e8
etime = spice.datetime2et(datetime.strptime('20210112T030017', '%Y%m%dT%H%M%S'))
z_unit_vec_HCI = [0, 0, 1*AU]
z_unit_vec_HCI = np.array(z_unit_vec_HCI, dtype=float)
y_unit_vec_HCI = [0, 1*AU, 0]
y_unit_vec_HCI = np.array(y_unit_vec_HCI, dtype=float)
x_unit_vec_HCI = [1*AU, 0, 0]
x_unit_vec_HCI = np.array(x_unit_vec_HCI, dtype=float)
PSP_pos, _ = spice.spkpos('SPP', etime, 'SPP_HCI', 'NONE', 'SUN')
z_WISPR_I, _ = spice.spkcpt(z_unit_vec_HCI, 'SPP', 'SPP_WISPR_INNER', etime,
                                     'SPP_HCI', 'OBSERVER', 'NONE', 'SUN')
y_WISPR_I, _ = spice.spkcpt(y_unit_vec_HCI, 'SPP', 'SPP_WISPR_INNER', etime,
                                     'SPP_HCI', 'OBSERVER', 'NONE', 'SUN')
x_WISPR_I, _ = spice.spkcpt(x_unit_vec_HCI, 'SPP', 'SPP_WISPR_INNER', etime,
                                     'SPP_HCI', 'OBSERVER', 'NONE', 'SUN')
x_unit_vec_WISPR_I = (x_WISPR_I[0:3] - PSP_pos) / AU
y_unit_vec_WISPR_I = (y_WISPR_I[0:3] - PSP_pos) / AU
z_unit_vec_WISPR_I = (z_WISPR_I[0:3] - PSP_pos) / AU
rotation_matrix = np.zeros([3, 3], dtype='float')
rotation_matrix[:, 0] = x_unit_vec_WISPR_I[:]
rotation_matrix[:, 1] = y_unit_vec_WISPR_I[:]
rotation_matrix[:, 2] = z_unit_vec_WISPR_I[:]
my_matrix = spice.sxform('SPP_HCI', 'SPP_WISPR_INNER', etime)     # 另一种求转换矩阵的方法，用已有函数sxform()，
# 且my_matrix是6*6矩阵，也可求出一阶微分的变换，而且是正演的变换。
# 第1步：得到 HCI 到 WISPR_INNER 坐标的转换(旋转矩阵，平移向量)

my_path = 'D://Microsoft Download/Formal Files/data file/FITS/WISPR-I_ENC07_L3_FITS/20210112/psp_L3_wispr_2021011\
2T113015_V1_1221.fits'
data, header = sunpy.io.fits.read(my_path)[0]
header['BUNIT'] = 'MSB'
a_map = sunpy.map.Map(data, header)
my_colormap = copy.deepcopy(a_map.cmap)
true_data = copy.deepcopy(data)
i = 0
while i < 1024:
    j = 0
    while j < 960:
        true_data[i, j] = data[1023-i, j]
        j = j + 1
    i = i + 1

my_fig = plt.figure(figsize=(9, 9))
ax = plt.subplot(111)
astropy_norm = simple_norm(true_data, stretch='log', log_a=500)
norm_SL = colors.SymLogNorm(linthresh=0.001 * 1e-10, linscale=0.1 * 1e-10, vmin=-0.0038 * 1e-10, vmax=0.14 * 1e-10)
plt.imshow(true_data, cmap=my_colormap, norm=norm_SL)
ax.set_xlabel('x-axis [pixel]')
ax.set_ylabel('y-axis [pixel]')
ax.set_title('2021-01-12 03:00:17 UT')
my_mappabel = matplotlib.cm.ScalarMappable(cmap=my_colormap, norm=norm_SL)
plt.colorbar(label='[MSB]')
point_set = np.zeros([5, 2, 2], dtype=float)
streak_num = 5
for i in range(streak_num):
    [x_1, x_2] = plt.ginput(2, timeout=-1)
    point_set[i, 0, :] = np.array(x_1, dtype=int) + 1
    point_set[i, 1, :] = np.array(x_2, dtype=int) + 1  # +1 是因为python数组序号是从0开始，像素系坐标应当比该数组大1
    # Note that point_set[i, 0, 0]( x-axis ) represents the second axis of the FITS data(axis-1) while point_set[i, 0,
    # 1] represents the first axis of the FITS data(axis-0).
plt.show()
# 第2步，画出WISPR-INNER的图，得到尘埃的原始数据点，下一步：通过数据点得到5条曲线并求出交点

slope_and_intercept = np.ones([5, 2], dtype=float)
for i in range(streak_num):
    slope_and_intercept[i, 0] = (point_set[i, 1, 1] - point_set[i, 0, 1]) / (point_set[i, 1, 0] - point_set[i, 0, 0])
    # the second index '0' means the slope
    slope_and_intercept[i, 1] = point_set[i, 1, 1] - slope_and_intercept[i, 0] * point_set[i, 1, 0]
    # the second index '1' means the intercept
interceptions = np.zeros([1, 2], dtype=float)

i = count = 0
while i < streak_num:
    j = i + 1
    while j < streak_num:
        temp_interp_x = (slope_and_intercept[j, 1] - slope_and_intercept[i, 1]) / (slope_and_intercept[i, 0] -
                                                                                   slope_and_intercept[j, 0])
        temp_interp_y = slope_and_intercept[i, 0] * temp_interp_x + slope_and_intercept[i, 1]
        interceptions = np.insert(interceptions, count, values=np.array([temp_interp_x, temp_interp_y], dtype=float),
                                  axis=0)
        count = count + 1
        j = j + 1
    i = i + 1

the_center_point = get_vanishing_point(interceptions)
second_figure = plt.figure()
ax = second_figure.add_subplot(111)
plot_x = np.linspace(-600, 1100)
for i in range(streak_num):
    plot_y = slope_and_intercept[i, 0] * plot_x + slope_and_intercept[i, 1]
    ax.plot(plot_x, plot_y)
for i in range(interceptions[:, 0].size - 1):
    ax.scatter(interceptions[i, 0], interceptions[i, 1])
ax.scatter(the_center_point[0], the_center_point[1], marker='x')
plt.show()
# 第3步，从原始数据点中得到中心点(灭点)，下一步：通过已经推出的公式反演得到直线方向

x_0_pixel = 960
y_0_pixel = 1024
f = 28e-3 / AU  # 相机焦距为28mm
alpha = 0.01e-3 * 2  # 单个CCD像素的尺寸(由于fits给出的数据点规模小了一倍，故乘以2)
phi = np.arctan((the_center_point[1] - y_0_pixel) / (the_center_point[0] - x_0_pixel))
theta = np.arctan( np.cos(phi) / alpha / (the_center_point[0] - x_0_pixel) * f)   # unit of phi and theta: rad
phi_deg = phi * 180 / np.pi
theta_deg = theta * 180 / np.pi
orientation_vec_WISPR_I = np.zeros([3, 1], dtype=float)
orientation_vec_WISPR_I[0] = np.cos(theta) * np.cos(phi)
orientation_vec_WISPR_I[1] = np.cos(theta) * np.sin(phi)
orientation_vec_WISPR_I[2] = np.sin(theta)
# 第4步，得到WISPR-INNER坐标系下的直线方向矢量，下一步：转换到HCI坐标系下

orientation_vec_HCI = np.dot(rotation_matrix.T, orientation_vec_WISPR_I)
print(orientation_vec_HCI)
# 第5步，得到尘埃粒子轨迹在HCI系下的方向矢量，结果可以用正演程序来检验一下

etime_2 = spice.datetime2et(datetime.strptime('20210216T030017', '%Y%m%dT%H%M%S'))
number = 4000
times = [(etime_2 - etime)/number * i + etime for i in range(number)]
PSP_pos_time, light_time = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
PSP_pos_time = PSP_pos_time.T
PSP_pos_time = PSP_pos_time/AU
fig_3 = plt.figure(figsize=(9, 9))
ax_3 = fig_3.add_subplot(111, projection='3d')
ax_3.set_xlabel('x (AU)')
ax_3.set_ylabel('y (AU)')
ax_3.set_zlabel('z (AU)')
ax_3.set_xlim(0, 0.3)
ax_3.set_ylim(-0.21, 0)
ax_3.scatter(0, 0, 0, c='#FF3333', label='Sun', marker='o')
ax_3.text(PSP_pos_time[0, 2000], PSP_pos_time[1, 2000], PSP_pos_time[2, 2000], 'PSP')
ax_3.text(0, 0, 0, 'Sun')
ax_3.plot(PSP_pos_time[0], PSP_pos_time[1], PSP_pos_time[2])
plt.title('SPP-HCI frame')
wispr_inner_parameter = spice.getfov(-96100, 4)
WISPR_pos = PSP_pos / AU
for i_edge in range(4):
    edge_inner1, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', etime,
                                 'SPP_HCI', 'OBSERVER', 'NONE', 'SUN')
    edge_inner1 = edge_inner1 / AU
    edge_motion = np.zeros([3, 1000], dtype=float)
    for cadence in range(1000):
        edge_motion[:, cadence] = (edge_inner1[0:3] - WISPR_pos[0:3]) * cadence / 6000 + WISPR_pos[0:3]
    if i_edge is 3:
        ax_3.plot(edge_motion[0], edge_motion[1], edge_motion[2], c='green', label='FOV of WISPR_INNER')
    else:
        ax_3.plot(edge_motion[0], edge_motion[1], edge_motion[2], c='green')
ax_3.legend()
x_0_3d = 0.15
y_0_3d = -0.05
z_0_3d = 0
ax_3.plot([x_0_3d, x_0_3d + orientation_vec_HCI[0] * 0.1], [y_0_3d, y_0_3d + orientation_vec_HCI[1] * 0.1],
          [z_0_3d, z_0_3d + orientation_vec_HCI[2] * 0.1])
plt.show()
# 第6步，通过正演定性判断结论的正确性
