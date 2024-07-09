"""
This is a script for dividing the impact zone that generates different dust storms detected by WISPR-Inner.
The 06 code is based on the projective transformation theory with the assumption that streaks are from impact-generated
debris, not primary dusts.
This examination is just geometry analysis. Optics is not considered.
By 2022.8.10, Tianhang Chen
"""
import spiceypy as spice
from sunpy.net import attrs as a
import sunpy.map
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
import sklearn.cluster as cluster
import pyvista as pv
from plotly import graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots

import furnsh_all_kernels

psp_3d_model_path = 'D://Microsoft Download/Formal Files/data file/3d_model/ParkerSolarProbe/stl/ParkerSolarProbe.stl'
WISPR_pos = np.array([0.865, -0.249, -0.300], dtype=float)  # the position of WISPR onboard PSP in spacecraft frame.
INPUT_path = 'D://Desktop/Impact_Rate/Orbit07_WISPR_Impact_Region_statistics/all_impact_origin.txt'


def add_psp_3d_model(pos, trace_union, epoch_time, scale=2.25 / 0.219372 / 2):
    """
    :param pos: A vector [x,y,z]. Center position of the spacecraft.
    :param trace_union:
    :param epoch_time:
    :param scale: A float, 10 by default. Scaling of the spacecraft.
    :return: A trace (go.Mesh3d()) for plotly.
    Note that the z axis in stl file is the true -z axis, x axis is the true y axis and y axis is the true -x axis!
    ——Tianhang Chen, 2022.8.10
    """
    mesh = pv.read(psp_3d_model_path)
    mesh.points = scale * mesh.points / np.max(mesh.points)
    vertices = mesh.points
    vertexes = copy.deepcopy(vertices)
    vertexes[:, 0] = vertices[:, 1] + pos[1]
    vertexes[:, 1] = -(vertices[:, 0] + pos[0])
    vertexes[:, 2] = -(vertices[:, 2] + pos[2])   # The true coordinates of each point in SC frame
    triangles = mesh.faces.reshape(-1, 4)

    # trace_union = plot_div(trace_union, WISPR_pos, epoch_time)
    trace = go.Mesh3d(x=vertexes[:, 0], y=vertexes[:, 1], z=vertexes[:, 2],
                      opacity=0.6,
                      color='silver',
                      i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                      showscale=False,
                      )
    trace_union.append(trace)

    # the_index = div_region(vertexes, triangles, WISPR_pos, epoch_time)
    # the_left_tri = the_index['left_triangle']
    # the_right_tri = the_index['right_triangle']
    # the_para_tri = the_index['parallel_triangle']
    # trace_union.append(
    #     go.Mesh3d(x=vertexes[:, 0], y=vertexes[:, 1], z=vertexes[:, 2],
    #               opacity=0.6,
    #               color='gold',
    #               i=triangles[the_left_tri, 1], j=triangles[the_left_tri, 2], k=triangles[the_left_tri, 3],
    #               showscale=False,
    #               )
    # )
    # trace_union.append(
    #     go.Mesh3d(x=vertexes[:, 0], y=vertexes[:, 1], z=vertexes[:, 2],
    #               opacity=0.6,
    #               color='gold',
    #               i=triangles[the_right_tri, 1], j=triangles[the_right_tri, 2], k=triangles[the_right_tri, 3],
    #               showscale=False,
    #               )
    # )
    # trace_union.append(
    #     go.Mesh3d(x=vertexes[:, 0], y=vertexes[:, 1], z=vertexes[:, 2],
    #               opacity=0.6,
    #               color='gold',
    #               i=triangles[the_para_tri, 1], j=triangles[the_para_tri, 2], k=triangles[the_para_tri, 3],
    #               showscale=False,
    #               )
    # )
    return trace_union


def div_region(all_point, triangles, camera_origin, epoch_time):
    """
    :param all_point: n*3 ndarray
    :param triangles: n*3 ndarray
    :param camera_origin: 1*3 ndarray
    :param epoch_time:
    :return:
    Note that the z axis in stl file is the true -z axis, x axis is the true y axis and y axis is the true -x axis!
    ——Tianhang Chen, 2022.8.10
    """
    para_len = 0.1  # unit: m
    boresight_sc, _ = spice.spkcpt(np.array([0, 0, 1], dtype=float), 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                   'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    boresight_sc = np.array(boresight_sc[0:3])
    xaxis_sc, _ = spice.spkcpt(np.array([1, 0, 0], dtype=float), 'SPP', 'SPP_WISPR_INNER', epoch_time,
                               'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    xaxis_sc = np.array(xaxis_sc[0:3])
    num_point = len(triangles[:, 0])
    para_point_index = []
    left_point_index = []
    right_point_index = []
    para_tri_index = []
    left_tri_index = []
    right_tri_index = []
    for fun_i in range(num_point):
        if np.fabs(boresight_sc[0] * (all_point[triangles[fun_i, 1], 0] - camera_origin[0])
                   + boresight_sc[1] * (all_point[triangles[fun_i, 1], 1] - camera_origin[1])
                   + boresight_sc[2] * (all_point[triangles[fun_i, 1], 2] - camera_origin[2])) <= para_len:
            para_tri_index.append(fun_i)
            para_point_index.append(triangles[fun_i, 1])
            para_point_index.append(triangles[fun_i, 2])
            para_point_index.append(triangles[fun_i, 3])
        elif boresight_sc[0] * (all_point[triangles[fun_i, 1], 0] - camera_origin[0]) \
            + boresight_sc[1] * (all_point[triangles[fun_i, 1], 1] - camera_origin[1]) \
                + boresight_sc[2] * (all_point[triangles[fun_i, 1], 2] - camera_origin[2]) > para_len:
            if xaxis_sc[0] * (all_point[triangles[fun_i, 1], 0] - camera_origin[0]) \
                   + xaxis_sc[1] * (all_point[triangles[fun_i, 1], 1] - camera_origin[1]) \
                   + xaxis_sc[2] * (all_point[triangles[fun_i, 1], 2] - camera_origin[2]) <= 0:
                left_tri_index.append(fun_i)
                left_point_index.append(triangles[fun_i, 1])
                left_point_index.append(triangles[fun_i, 2])
                left_point_index.append(triangles[fun_i, 3])
            else:
                right_tri_index.append(fun_i)
                right_point_index.append(triangles[fun_i, 1])
                right_point_index.append(triangles[fun_i, 2])
                right_point_index.append(triangles[fun_i, 3])
        elif xaxis_sc[0] * (all_point[triangles[fun_i, 1], 0] - camera_origin[0]) \
                + xaxis_sc[1] * (all_point[triangles[fun_i, 2], 1] - camera_origin[1]) \
                + xaxis_sc[2] * (all_point[triangles[fun_i, 3], 2] - camera_origin[2]) <= 0:
            right_tri_index.append(fun_i)
            right_point_index.append(triangles[fun_i, 1])
            right_point_index.append(triangles[fun_i, 2])
            right_point_index.append(triangles[fun_i, 3])
        else:
            left_tri_index.append(fun_i)
            left_point_index.append(triangles[fun_i, 0])
            left_point_index.append(triangles[fun_i, 1])
            left_point_index.append(triangles[fun_i, 2])
    all_index = {'left_point': left_point_index, 'left_triangle': left_tri_index,
                 'right_point': right_point_index, 'right_triangle': right_tri_index,
                 'parallel_point': para_point_index, 'parallel_triangle': para_tri_index}
    return all_index


def plot_div(trace_union, camera_origin, epoch_time):
    """
    :param trace_union:
    :param camera_origin: 1*3 ndarray
    :param epoch_time:
    :return:
    Note that the z axis in stl file is the true -z axis, x axis is the true y axis and y axis is the true -x axis!
    ——Tianhang Chen, 2022.8.10
    """
    para_len = 0.2  # unit: m
    boresight_sc, _ = spice.spkcpt(np.array([0, 0, 1], dtype=float), 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                   'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    boresight_sc = np.array(boresight_sc[0:3])
    xaxis_sc, _ = spice.spkcpt(np.array([1, 0, 0], dtype=float), 'SPP', 'SPP_WISPR_INNER', epoch_time,
                               'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    xaxis_sc = np.array(xaxis_sc[0:3])
    x = np.linspace(-3, 3)
    y = np.linspace(-3, 3)
    x_range, y_range = np.meshgrid(x, y)
    z_left = (para_len - boresight_sc[0] * (x_range - camera_origin[0]) - boresight_sc[1]
              * (y_range - camera_origin[1])) / boresight_sc[2] + camera_origin[2]
    z_right = (- para_len - boresight_sc[0] * (x_range - camera_origin[0]) - boresight_sc[1]
               * (y_range - camera_origin[1])) / boresight_sc[2] + camera_origin[2]
    x_plane = (- xaxis_sc[0] * (x_range - camera_origin[0]) - xaxis_sc[1]
               * (y_range - camera_origin[1])) / xaxis_sc[2] + camera_origin[2]
    trace_union.append(go.Surface(x=x, y=y, z=z_left, opacity=0.5, colorbar=dict(bgcolor='rgba(1,0,0,0)'), showscale=False))
    trace_union.append(go.Surface(x=x, y=y, z=z_right, opacity=0.5, colorbar=dict(bgcolor='rgba(1,0,0,0)'), showscale=False))
    trace_union.append(go.Surface(x=x, y=y, z=x_plane, opacity=0.5, colorbar=dict(bgcolor='rgba(0,0,0,0)'), showscale=False))
    return trace_union


def get_fov_wispr(trace_union, epoch_time):
    """
    :param trace_union:
    :param epoch_time: the epoch(SPICE) of observing time.
    :return: fun_fov_traces and fun_axes_traces are both the 'trace' type introduced in plotly(https://plotly.com/python)
    """
    wispr_inner_parameter = spice.getfov(-96100, 4)
    for i_edge in range(4):
        edge_inner1, _ = spice.spkcpt(wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                      'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
        edge_motion = np.zeros([3, 1000], dtype=float)
        for cadence in range(1000):
            edge_motion[:, cadence] = (edge_inner1[0:3] * 2e1) * cadence / 6000
        edge_motion[0, :] = edge_motion[0, :] + WISPR_pos[0]
        edge_motion[1, :] = edge_motion[1, :] + WISPR_pos[1]
        edge_motion[2, :] = edge_motion[2, :] + WISPR_pos[2]
        trace_union.append(go.Scatter3d(
                                             x=edge_motion[0], y=edge_motion[1], z=edge_motion[2], mode='lines',
                                             opacity=0.6, line=dict(color='green', width=5),
                                             legendgroup='FOV',
                                             legendgrouptitle=dict(text='FOV of WISPR_Inner')
                                             )
                           )
    # ax_3.plot([0, the_center_point_3d[0]], [0, the_center_point_3d[1]], [0, the_center_point_3d[2]], c='orange'
    #           , label='vanishing point position')
    # the top(bottom) of streaks retrieved that converge at the y<0(y>0)
    # vanishing point in 3d frame.
    return trace_union


def plot_all(trace_union, time_str):
    plotly_fig = go.Figure()
    for fun_trace in trace_union:
        plotly_fig.add_trace(fun_trace)
    plotly_fig.update_layout(title_text="Spacecraft Frame" + ' ' + time_str,
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


def read_impact_position(trace_union, input_file_path):
    """
    :param trace_union:
    :param input_file_path:
    :return:
    """
    fun_file = open(input_file_path, 'r')
    impact_pos_str = fun_file.read().splitlines()
    num_pos = len(impact_pos_str)
    impact_pos = np.zeros([num_pos, 3], dtype=float)
    fun_file.close()
    for fun_i in range(num_pos):
        temp_str = impact_pos_str[fun_i].split(', ')
        if temp_str[0][0] == 'w':
            continue
        impact_pos[fun_i, 0] = float(temp_str[0].strip('['))
        impact_pos[fun_i, 1] = float(temp_str[1])
        impact_pos[fun_i, 2] = float(temp_str[2])
        temp_radius = float(temp_str[4].strip(']'))
        temp_circle_x = np.linspace(impact_pos[fun_i, 0] - temp_radius, impact_pos[fun_i, 0] + temp_radius, 100)
        temp_circle_y_po = impact_pos[fun_i, 1] + np.sqrt(np.abs(temp_radius**2 - (temp_circle_x -
                                                                                   impact_pos[fun_i, 0])**2))
        temp_circle_y_ne = impact_pos[fun_i, 1] - np.sqrt(np.abs(temp_radius**2 - (temp_circle_x -
                                                                                   impact_pos[fun_i, 0])**2))
        temp_circle_z = [impact_pos[fun_i, 2]]*100
        temp_date = temp_str[3].split('T')
        trace_union.append(
            go.Scatter3d(x=[impact_pos[fun_i, 0]], y=[impact_pos[fun_i, 1]], z=[impact_pos[fun_i, 2]],
                         opacity=1,
                         mode='markers',
                         name=temp_date[0] + '\n' + temp_date[1],
                         marker=dict(size=4, color='red', symbol='x'),
                         legendgroup=str(fun_i),
                         showlegend=False,
                         hoverlabel=dict(namelength=-1)
                         )
        )
        trace_union.append(
            go.Scatter3d(x=temp_circle_x, y=temp_circle_y_po, z=temp_circle_z,
                         opacity=1,
                         mode='lines',
                         name=temp_date[0] + '\n' + temp_date[1],
                         marker=dict(size=4),
                         legendgroup=str(fun_i),
                         showlegend=False,
                         line=dict(width=5, dash='dashdot', color='blue'),
                         hoverlabel=dict(namelength=-1)
                         )
        )
        trace_union.append(
            go.Scatter3d(x=temp_circle_x, y=temp_circle_y_ne, z=temp_circle_z,
                         opacity=1,
                         mode='lines',
                         name=temp_date[0] + '\n' + temp_date[1],
                         marker=dict(size=4),
                         legendgroup=str(fun_i),
                         showlegend=False,
                         line=dict(width=5, dash='dashdot', color='blue'),
                         hoverlabel=dict(namelength=-1)
                         )
        )

    fun_file.close()
    return trace_union


def main_function():
    my_time = '20210112T123016'
    etime = spice.datetime2et(datetime.strptime(my_time, '%Y%m%dT%H%M%S'))
    my_trace = []
    my_trace = add_psp_3d_model(np.array([0, 0, 0]), my_trace, etime)
    my_trace = get_fov_wispr(my_trace, etime)
    my_trace = read_impact_position(my_trace, INPUT_path)
    plot_all(my_trace, my_time)


main_function()


