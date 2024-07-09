"""
This script is a test for the case that the origin of particles' cone locates behind S/C camera but observed by it.
The boresight of the camera is just z axis.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

focal_length = 28e-3  # unit: m
pix_size = 20e-6  # unit: m
x_max = 1024  # unit: pixel
y_max = 960  # unit: pixel


def gen_par(num_par, cone_origin, cone_axis, cone_angle, speed, fun_exp_time):
    """
    This function generate the state of particles in WISPR_Inner frame. These particles are ejected from a single point.
    :param num_par: (int) The number of particles generated.
    :param cone_origin: (1*3 float ndarrays) The coordinate of the cone's origin(i.e. the origin of these particles).
    :param cone_axis: (1*3 float ndarrays) The unit vector of the cone's axis.
    :param cone_angle: (int, unit: degree) The angle of the cone.
    :param speed: (float) The magnitude of these particles' velocities.
    :param fun_exp_time:(float) the single exposure time of WISPR figure.
    :return: fun_state:
    """
    vel_vec = np.zeros([num_par, 3], dtype=float)
    vel_mag = np.array([speed]*num_par, dtype=float)
    base_unit = np.zeros([2, 3], dtype=float)
    base_unit[0, 0] = 1
    base_unit[0, 1] = 0
    base_unit[0, 2] = - cone_axis[0] * base_unit[0, 0] / cone_axis[2]
    base_unit[0, :] = base_unit[0, :] / np.sqrt(base_unit[0, 0]**2 + base_unit[0, 1]**2 + base_unit[0, 2]**2)
    base_unit[1, :] = np.cross(cone_axis, base_unit[0])

    r_cone = np.random.random(num_par) * np.tan(cone_angle / 180 * np.pi)
    phi_deg = np.random.random(num_par) * 360
    for fun_i in range(num_par):
        vel_vec[fun_i, :] = cone_axis[:] + r_cone[fun_i] * base_unit[0, :] * np.cos(phi_deg[fun_i]*np.pi/180) + \
                            r_cone[fun_i] * base_unit[1, :] * np.sin(phi_deg[fun_i]*np.pi/180)
        vel_vec[fun_i, :] = vel_vec[fun_i, :] / np.sqrt(vel_vec[fun_i, 0]**2 + vel_vec[fun_i, 1]**2 + vel_vec[fun_i, 2]**2)

    fun_state = np.zeros((num_par, 3, 2), dtype=float)
    # 4 means radius/x/y/z, 2 means the start/end position of particle. (e.g. fun_state[i, 3, 0] is the start y
    # coordinate of the (i+1)_th particle )
    fun_state[:, 0, 0] = cone_origin[0]
    fun_state[:, 1, 0] = cone_origin[1]
    fun_state[:, 2, 0] = cone_origin[2]
    fun_state[:, 0, 1] = cone_origin[0] + vel_mag[:] * vel_vec[:, 0] * fun_exp_time
    fun_state[:, 1, 1] = cone_origin[1] + vel_mag[:] * vel_vec[:, 1] * fun_exp_time
    fun_state[:, 2, 1] = cone_origin[2] + vel_mag[:] * vel_vec[:, 2] * fun_exp_time
    return fun_state


def par_in_camera(num_par, fun_state):
    """
    :param num_par: the number of particles generated.
    :param fun_state: The output data in function gen_par().
    :return: output_state: similar to fun_state, but in 2d camera frame. (unit: m)
    """
    if fun_state[0, 2, 0] <= focal_length:
        for fun_i in range(num_par):
            k = (fun_state[fun_i, 2, 1] - 0.1) / (fun_state[fun_i, 2, 1] - fun_state[fun_i, 2, 0])
            fun_state[fun_i, 0, 0] = fun_state[fun_i, 0, 1] - (fun_state[fun_i, 0, 1] - fun_state[fun_i, 0, 0]) * k
            fun_state[fun_i, 1, 0] = fun_state[fun_i, 1, 1] - (fun_state[fun_i, 1, 1] - fun_state[fun_i, 1, 0]) * k
            fun_state[fun_i, 2, 0] = 0.1

    output_state = np.zeros((num_par, 2, 2), dtype=float)
    output_state[:, 0, 0] = fun_state[:, 0, 0] * focal_length / fun_state[:, 2, 0]
    output_state[:, 0, 1] = fun_state[:, 0, 1] * focal_length / fun_state[:, 2, 1]
    output_state[:, 1, 0] = -fun_state[:, 1, 0] * focal_length / fun_state[:, 2, 0]
    output_state[:, 1, 1] = -fun_state[:, 1, 1] * focal_length / fun_state[:, 2, 1]

    return output_state


def plot_par(num_par, state_2d, state_3d):
    """
    :param num_par: ...
    :param state_2d:  the output data in function par_in_camera().
    :param state_3d:  the output data in function gen_par().
    :return: None.
    """
    fig_1 = plt.figure(figsize=(9, 9))
    fig_2 = plt.figure()
    ax_1 = fig_1.add_subplot(111, projection='3d')
    ax_2 = fig_2.add_subplot(111)
    for fun_i in range(num_par):
        ax_1.plot([state_3d[fun_i, 0, 0], state_3d[fun_i, 0, 1]], [state_3d[fun_i, 1, 0], state_3d[fun_i, 1, 1]],
                  [state_3d[fun_i, 2, 0], state_3d[fun_i, 2, 1]])
        ax_2.plot([state_2d[fun_i, 0, 0], state_2d[fun_i, 0, 1]], [state_2d[fun_i, 1, 0], state_2d[fun_i, 1, 1]])
    ax_2.set_xlim(-x_max * pix_size / 2, x_max * pix_size / 2)
    ax_2.set_ylim(-y_max * pix_size / 2, y_max * pix_size / 2)
    ax_1.set_xlim(-10, 10)
    ax_1.set_ylim(-10, 10)
    ax_1.set_title('The Particles in 3D Camera frame')
    ax_2.set_title('The Particles in 2D Camera frame')
    ax_1.scatter(0, 0, 0,  c='#FF3333', marker='o')
    ax_1.text(0, 0, 0, 'Camera Origin')
    ax_1.plot([0, -x_max * pix_size / 2 * 100], [0, -y_max * pix_size / 2 * 100], [0, focal_length * 100], '--g')
    ax_1.plot([0, x_max * pix_size / 2 * 100], [0, -y_max * pix_size / 2 * 100], [0, focal_length * 100], '--g')
    ax_1.plot([0, -x_max * pix_size / 2 * 100], [0, y_max * pix_size / 2 * 100], [0, focal_length * 100], '--g')
    ax_1.plot([0, x_max * pix_size / 2 * 100], [0, y_max * pix_size / 2 * 100], [0, focal_length * 100], '--g',
              label='FOV')
    ax_1.set_xlim(-2.5, 2.5)
    ax_1.set_ylim(-1.5, 1.5)
    ax_1.set_zlim(0, 4)
    ax_1.set_xlabel('x')
    ax_1.set_ylabel('y')
    ax_1.set_zlabel('z')
    ax_2.set_xlabel('x')
    ax_2.set_ylabel('y')
    plt.show()


def main_function():
    my_number = 100
    my_origin = np.array([-2, 0, -0.5], dtype=float)
    my_axis = np.array([9/10, 0, np.sqrt(1-(9/10)**2)], dtype=float)
    my_angle = 30
    my_speed_range = 20
    my_exp_time = 0.5
    state_1 = gen_par(my_number, my_origin, my_axis, my_angle, my_speed_range, my_exp_time)
    state_2 = par_in_camera(my_number, state_1)
    print(state_1, state_2)
    plot_par(my_number, state_2, state_1)


main_function()
