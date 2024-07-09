"""
This script tries to explain why the image origin of cone will reverse if the object origin locates behind camera.
"""
import numpy as np
import spiceypy as spice
from datetime import datetime
import sklearn.cluster as cluster
import pyvista as pv
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.transforms as trans
import plotly.offline as py
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import copy

import furnsh_all_kernels


psp_3d_model_path = 'D://Microsoft Download/Formal Files/data file/3d_model/ParkerSolarProbe/stl/ParkerSolarProbe.stl'
WISPR_pos = np.array([0.865, -0.249, -0.300], dtype=float)  # the position of WISPR onboard PSP in spacecraft frame.
scale = 2.25 / 0.219372 / 2
my_time = '2021-01-12T12:30:16'
epoch_time = spice.datetime2et(datetime.strptime(my_time, '%Y-%m-%dT%H:%M:%S'))
line_k = (550.299-387.673) / (546.849-221.938)
impact_origin = np.array([1.00, -0.5, 1.353914], dtype=float)


def add_right_cax(ax, pad, width):
    axpos = ax.get_position()
    caxpos = trans.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


def plot_cone_projective(cone_origin, cone_angle):
    """
    This function is performed in WISPR_Inner camera frame.
    :param cone_origin:
    :param cone_angle: unit: deg
    :return:
    """
    beta = (cone_origin[1] - line_k * cone_origin[0]) / cone_origin[2]
    theta_max = np.arcsin(np.sqrt(1/(1+beta**2))) * 180 / np.pi
    phi = np.linspace(0, 360, num=1800, endpoint=False) * np.pi / 180
    theta = np.linspace(-theta_max, theta_max, num=300) * np.pi / 180
    [theta_mesh, phi_mesh] = np.meshgrid(theta, phi)
    axis_c = np.sin(theta_mesh)
    axis_a = (-2*beta*line_k*axis_c + np.sqrt((2*beta*line_k*axis_c)**2 -
                                              4*(1+line_k**2)*(axis_c**2 + beta**2 * axis_c**2 - 1))) / 2/(1+line_k**2)
    axis_b = beta * axis_c + line_k * axis_a
    vec_a = np.zeros_like(axis_a)
    vec_b = np.zeros_like(axis_a)
    vec_c = np.zeros_like(axis_a)
    norm_vec = np.zeros([3], dtype=float)
    trial_vec = np.zeros_like(axis_a)
    base_unit = np.zeros([2, 3], dtype=float)
    for fun_i in range(300):
        norm_vec[0] = -axis_b[0, fun_i]
        norm_vec[1] = axis_a[0, fun_i]
        norm_vec[2] = 0
        norm_vec_len = np.sqrt(norm_vec[0]**2 + norm_vec[1]**2)
        norm_vec[:] = norm_vec[:]/norm_vec_len
        k_matrix = np.array([[0, -norm_vec[2], norm_vec[1]], [norm_vec[2], 0, -norm_vec[0]],
                             [-norm_vec[1], norm_vec[0], 0]], dtype=float)
        cos_alpha = axis_c[0, fun_i]
        rot_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) + np.sqrt(1-cos_alpha**2) * k_matrix + \
                     (1-cos_alpha) * np.matmul(k_matrix, k_matrix)
        base_unit[0, :] = np.matmul(rot_matrix, [1, 0, 0])
        base_unit[1, :] = np.matmul(rot_matrix, [0, 1, 0])
        for fun_j in range(1800):
            vec_a[fun_j, fun_i] = axis_a[fun_j, fun_i] + \
                                  np.tan(cone_angle/180*np.pi) * base_unit[0, 0] * np.cos(phi_mesh[fun_j, fun_i]) + \
                                  np.tan(cone_angle/180*np.pi) * base_unit[1, 0] * np.sin(phi_mesh[fun_j, fun_i])
            vec_b[fun_j, fun_i] = axis_b[fun_j, fun_i] + \
                                  np.tan(cone_angle / 180 * np.pi) * base_unit[0, 1] * np.cos(phi_mesh[fun_j, fun_i]) +\
                                  np.tan(cone_angle / 180 * np.pi) * base_unit[1, 1] * np.sin(phi_mesh[fun_j, fun_i])
            vec_c[fun_j, fun_i] = axis_c[fun_j, fun_i] + \
                                  np.tan(cone_angle / 180 * np.pi) * base_unit[0, 2] * np.cos(phi_mesh[fun_j, fun_i]) +\
                                  np.tan(cone_angle / 180 * np.pi) * base_unit[1, 2] * np.sin(phi_mesh[fun_j, fun_i])
    final_k = (vec_c * cone_origin[1] - vec_b * cone_origin[2]) / (vec_c * cone_origin[0] - vec_a * cone_origin[2])
    final_angle = np.arctan(final_k)*180 / np.pi
    final_cone_angle = final_angle.max(axis=0) - final_angle.min(axis=0)
    fun_fig = plt.figure(figsize=(9, 9))
    fun_ax1 = fun_fig.add_subplot(2, 1, 1)
    fun_ax2 = fun_fig.add_subplot(2, 1, 2)
    p = fun_ax1.pcolormesh(theta_mesh*180/np.pi, phi_mesh*180/np.pi,  final_angle)
    fun_ax2.plot(theta*180/np.pi, final_cone_angle)
    cax = add_right_cax(fun_ax1, pad=0.02, width=0.02)
    cbar = fun_fig.colorbar(p, cax=cax)
    fun_ax1.set_ylabel(r'$ \varphi $ [deg] ', fontsize=15)
    fun_ax1.set_xlabel(r'$ \theta $ [deg] ', fontsize=15)
    cbar.set_label('Inclination Angle [deg]', fontsize=15)
    fun_ax1.tick_params(labelsize=15)
    fun_ax2.set_ylabel(r'$\Delta \alpha$ [deg]', fontsize=15)
    fun_ax1.set_xlabel(r'$ \theta $ [deg] ', fontsize=15)
    fun_ax2.tick_params(labelsize=15)
    plt.show()


plot_cone_projective(impact_origin, 30)
# trace_union = []
# mesh = pv.read(psp_3d_model_path)
# mesh.points = scale * mesh.points / np.max(mesh.points)
#
# vertices = mesh.points
# vertexes = copy.deepcopy(vertices)
# vertexes[:, 0] = vertices[:, 1]
# vertexes[:, 1] = -vertices[:, 0]
# vertexes[:, 2] = -vertices[:, 2]   # The true coordinates of each point in SC frame
# triangles = mesh.faces.reshape(-1, 4)
# # trace = go.Mesh3d(x=vertexes[:, 0], y=vertexes[:, 1], z=vertexes[:, 2],
# #                   opacity=1,
# #                   color='silver',
# #                   i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
# #                   showscale=False,
# #                   )
# # trace_union.append(trace)
# # plot the PSP model.
#
# wispr_inner_parameter = spice.getfov(-96100, 4)
# for i_edge in range(4):
#     edge_motion = np.zeros([3, 1000], dtype=float)
#     for cadence in range(1000):
#         edge_motion[:, cadence] = (wispr_inner_parameter[4][i_edge] * 2e1) * cadence / 6000
#     trace_union.append(go.Scatter3d(
#                                     x=edge_motion[0], y=edge_motion[1], z=edge_motion[2], mode='lines',
#                                     opacity=0.6, line=dict(color='green', width=5),
#                                     showlegend=False,
#                                     legendgroup='FOV',
#                                     legendgrouptitle=dict(text='FOV of WISPR_Inner')
#                                              )
#                        )
#
# point_1_3d = np.array([-960*10e-6, -720*10e-6, 28e-3], dtype='float32')*1.5e2
# point_2_3d = np.array([960*10e-6, -480*10e-6, 28e-3], dtype='float32')*1.5e2
# point_3_3d = np.zeros([3], dtype='float32')
# point_3_3d[0] = -point_1_3d[1] * (point_2_3d[0] - point_1_3d[0]) / (point_2_3d[1] - point_1_3d[1]) + point_1_3d[0]
# point_3_3d[2] = point_1_3d[2]
# trace_union.append(
#                     go.Mesh3d(
#                                     x=[0, point_1_3d[0], point_2_3d[0]],
#                                     y=[0, point_1_3d[1], point_2_3d[1]],
#                                     z=[0, point_1_3d[2], point_2_3d[2]],
#                                     opacity=0.4,
#                                     color='blue',
#                                     i=[0], j=[1], k=[2],
#                                     showscale=False,
#                                 )
#                    )
#
# trace_union.append(
#                     go.Scatter3d(
#                                     x=[point_2_3d[0], point_3_3d[0]],
#                                     y=[point_2_3d[1], point_3_3d[1]],
#                                     z=[point_2_3d[2], point_3_3d[2]],
#                                     mode='lines',
#                                     opacity=0.6, line=dict(color='blue', width=5, dash='dash'),
#                                     showlegend=False,
#                                              )
#                  )
#
# trace_union.append(
#                     go.Scatter3d(
#                                     x=[-0.5*point_3_3d[0], 0.5*point_2_3d[0]],
#                                     y=[-0.5*point_3_3d[1], 0.5*point_2_3d[1]],
#                                     z=[-0.5*point_3_3d[2], 0.5*point_2_3d[2]],
#                                     mode='lines',
#                                     opacity=1, line=dict(width=5, dash='solid'),
#                                     showlegend=False,
#                                              )
#                  )
#
# trace_union.append(
#                     go.Scatter3d(
#                                     x=[0, 5],
#                                     y=[0, 0],
#                                     z=[0, 0],
#                                     mode='lines',
#                                     opacity=0.6, line=dict(color='black', width=5, dash='solid'),
#                                     showlegend=False,
#                                              )
#                  )
# trace_union.append(
#                     go.Scatter3d(
#                                     x=[0, 0],
#                                     y=[0, 1.5],
#                                     z=[0, 0],
#                                     mode='lines',
#                                     opacity=0.6, line=dict(color='black', width=5, dash='solid'),
#                                     showlegend=False,
#                                              )
#                  )
# trace_union.append(
#                     go.Scatter3d(
#                                     x=[0, 0],
#                                     y=[0, 0],
#                                     z=[0, 7],
#                                     mode='lines',
#                                     opacity=0.6, line=dict(color='black', width=5, dash='solid'),
#                                     showlegend=False,
#                                              )
#                  )
#
# trace_union.append(
#                     go.Scatter3d(
#                                     x=[-0.5*point_3_3d[0], point_3_3d[0]],
#                                     y=[-0.5*point_3_3d[1], point_3_3d[1]],
#                                     z=[-0.5*point_3_3d[2], point_3_3d[2]],
#                                     mode='lines',
#                                     opacity=0.6, line=dict(color='blue', width=5, dash='dash'),
#                                     showlegend=False,
#                                              )
#                  )
#
# trace_union.append(
#                     go.Scatter3d(
#                                     x=[-0.5*point_3_3d[0]],
#                                     y=[-0.5*point_3_3d[1]],
#                                     z=[-0.5*point_3_3d[2]],
#                                     mode='markers',
#                                     opacity=1,
#                                     showlegend=False,
#                                              )
#                  )
#
# plotly_fig = go.Figure()
# for fun_trace in trace_union:
#     plotly_fig.add_trace(fun_trace)
# # plotly_fig.update_layout(title_text="Spacecraft Frame" + ' ' + my_time,
# #                          title_font_size=30)
# plotly_fig.show()
#
# fig = plt.figure()
# ax_1 = fig.add_subplot(111)
# ax_1.plot([-960, 720], [960, 480])
# ax_1.set_xlim(-960, 960)
# ax_1.set_ylim(-1024, 1024)
# ax_1.set_xlabel('x [pixel]')
# ax_1.set_ylabel('y [pixel]')
# ax_1.set_xticks([-960, -480, 0, 480, 960])
# ax_1.set_yticks([-1024, -512, 0, 512, 1024])
# ax_1.set_xticklabels(['0', '480', '960', '1440', '1920'])
# ax_1.set_yticklabels(['2048', '1536', '1024', '512', '0'])
# plt.show()





