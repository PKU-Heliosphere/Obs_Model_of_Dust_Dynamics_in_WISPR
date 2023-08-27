import pyvista as pv
import numpy as np
import scipy as sp
import furnsh_all_kernels
import spiceypy as spice
import plotly.graph_objects as go
import plotly.offline as offline
from datetime import datetime
import sunpy.io
import sunpy.map
import copy
from utils import *


AU = 1.496e8  # unit: km
fits_file_path = 'D://Microsoft Download/Formal Files/data file/FITS/WISPR-I_ENC07_L3_FITS/20210112/psp_L3_wispr_' \
                 '20210112T033017_V1_1221.fits'


def matplotlib_to_plotly(cmap, pl_entries, A=1.):
    """
    This code is from HCP. But some modifications are applied.
    In order to emphasize streaks in WISPR figure,the colormap is not a linear one but with a log one.
    :param cmap: colormap in matplotlib.
    :param pl_entries: numbers of entries.
    :param A: the normalized colormap with the form of f(x) = x^A, where f(0) = 0, f(1) = 1. (A, B>0)
    :return:
    """
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    # C = -A
    # C = 0
    # B = 1 + 1/A
    for k in range(pl_entries):
        the_color = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([(k*h)**A, 'rgb'+str((the_color[0], the_color[1], the_color[2]))])
    return pl_colorscale


def dust_density(hc_distance, latitude):
    """
    :param hc_distance: heliocentric distance (unit: AU)
    :param latitude: ecliptic latitude (unit: degree)
    :return: number density of dust cloud at the given position. (unit: m^-3)
    """
    n0 = 1  # normalization constant.
    r_in = 0.0233
    r_out = 0.0884
    n = n0 * (1 + (4.5*np.sin(latitude / 180 * np.pi))**2) ** (-0.65) * hc_distance ** (-1.3) * (hc_distance > r_out)+ \
        n0 * (1 + (4.5 * np.sin(latitude / 180 * np.pi)) ** 2) ** (-0.65) * hc_distance ** (-1.3) * (hc_distance - r_in)\
                / (r_out - r_in) * (hc_distance <= r_out) * (hc_distance >= r_in) + 0 * (hc_distance < r_in)
    return n


def plot_dustcloud(lowerlim, upperlim, grid_num):
    """
    :param lowerlim: 1*3 ndarray. index 0: x, index 1: y, index 2: z. [unit: AU]
    :param upperlim: similar to lowerlim. [unit: AU]
    :param grid_num: similar to lowerlim. [unit: 1]
    :return:
    """
    data, header = sunpy.io.read_file(fits_file_path)[0]
    header['BUNIT'] = 'MSB'
    a_etime = spice.datetime2et(datetime.strptime(header['DATE-BEG'], '%Y-%m-%dT%H:%M:%S.%f'))
    vec_sc, _ = spice.spkcpt([0.40, 0.1968, -0.8951], 'SPP', 'SPP_WISPR_INNER', a_etime,
                             'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    a_map = sunpy.map.Map(data, header)
    my_colormap = copy.deepcopy(a_map.cmap)
    wispr_colorscale = matplotlib_to_plotly(my_colormap, 255, A=0.8)
    x_mesh_ini, y_mesh_ini, z_mesh_ini = np.meshgrid(np.linspace(lowerlim[0], upperlim[0], grid_num[0]),
                                                     np.linspace(lowerlim[1], upperlim[1], grid_num[1]),
                                                     np.linspace(lowerlim[2], upperlim[2], grid_num[2]))
    dimensions_ini = x_mesh_ini.shape
    x_mesh = np.zeros([grid_num[0], grid_num[1], grid_num[2]], dtype=np.float64)
    y_mesh = np.zeros_like(x_mesh)
    z_mesh = np.zeros_like(x_mesh)
    for index_3 in range(dimensions_ini[2]):
        x_mesh[:, :, index_3] = x_mesh_ini[:, :, index_3].T
        y_mesh[:, :, index_3] = y_mesh_ini[:, :, index_3].T
        z_mesh[:, :, index_3] = z_mesh_ini[:, :, index_3].T
    r_mesh = np.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    lat_mesh = np.arcsin(z_mesh/r_mesh) / np.pi * 180
    num_density = dust_density(r_mesh, lat_mesh)
    # print(num_density)
    dimensions = x_mesh.shape
    var_name = 'Number Density of Dust [normalized by the value at 1 AU]'
    vol_trace = go.Volume(x=x_mesh.flatten(), y=y_mesh.flatten(), z=z_mesh.flatten(),
                          value=num_density.flatten(),
                          opacity=0.01,  # needs to be small to see through all surfaces
                          surface=dict(count=200),  # needs to be a large number for good volume rendering
                          colorscale=wispr_colorscale,
                          cmin=0, cmax=1
                          )
    return vol_trace


def util_input():
    Planet_dict = {'Planet_epoch': [datetime(2020, 11, 1, 0, 0), datetime(2020, 11, 1, 6, 0),
                                    datetime(2020, 11, 1, 12, 0), datetime(2020, 11, 1, 18, 0),
                                    datetime(2020, 11, 2, 0, 0), datetime(2020, 11, 2, 6, 0),
                                    datetime(2020, 11, 2, 12, 0), datetime(2020, 11, 2, 18, 0),
                                    datetime(2020, 11, 3, 0, 0), datetime(2020, 11, 3, 6, 0),
                                    datetime(2020, 11, 3, 12, 0), datetime(2020, 11, 3, 18, 0),
                                    datetime(2020, 11, 4, 0, 0), datetime(2020, 11, 4, 6, 0),
                                    datetime(2020, 11, 4, 12, 0), datetime(2020, 11, 4, 18, 0),
                                    datetime(2020, 11, 5, 0, 0), datetime(2020, 11, 5, 6, 0),
                                    datetime(2020, 11, 5, 12, 0), datetime(2020, 11, 5, 18, 0),
                                    datetime(2020, 11, 6, 0, 0), datetime(2020, 11, 6, 6, 0),
                                    datetime(2020, 11, 6, 12, 0), datetime(2020, 11, 6, 18, 0),
                                    datetime(2020, 11, 7, 0, 0), datetime(2020, 11, 7, 6, 0),
                                    datetime(2020, 11, 7, 12, 0), datetime(2020, 11, 7, 18, 0),
                                    datetime(2020, 11, 8, 0, 0), datetime(2020, 11, 8, 6, 0),
                                    datetime(2020, 11, 8, 12, 0), datetime(2020, 11, 8, 18, 0),
                                    datetime(2020, 11, 9, 0, 0), datetime(2020, 11, 9, 6, 0),
                                    datetime(2020, 11, 9, 12, 0), datetime(2020, 11, 9, 18, 0),
                                    datetime(2020, 11, 10, 0, 0), datetime(2020, 11, 10, 6, 0),
                                    datetime(2020, 11, 10, 12, 0), datetime(2020, 11, 10, 18, 0),
                                    datetime(2020, 11, 11, 0, 0), datetime(2020, 11, 11, 6, 0),
                                    datetime(2020, 11, 11, 12, 0), datetime(2020, 11, 11, 18, 0)],
                   'Planet_list': ['MERCURY', 'VENUS'],
                   'Planet_model_dt': datetime(2020, 11, 11, 18, 0)}
    SC_dict = {'SC_str': 'SPP', 'orbit_epoch': [datetime(2020, 11, 1, 0, 0), datetime(2020, 11, 1, 6, 0),
                                    datetime(2020, 11, 1, 12, 0), datetime(2020, 11, 1, 18, 0),
                                    datetime(2020, 11, 2, 0, 0), datetime(2020, 11, 2, 6, 0),
                                    datetime(2020, 11, 2, 12, 0), datetime(2020, 11, 2, 18, 0),
                                    datetime(2020, 11, 3, 0, 0), datetime(2020, 11, 3, 6, 0),
                                    datetime(2020, 11, 3, 12, 0), datetime(2020, 11, 3, 18, 0),
                                    datetime(2020, 11, 4, 0, 0), datetime(2020, 11, 4, 6, 0),
                                    datetime(2020, 11, 4, 12, 0), datetime(2020, 11, 4, 18, 0),
                                    datetime(2020, 11, 5, 0, 0), datetime(2020, 11, 5, 6, 0),
                                    datetime(2020, 11, 5, 12, 0), datetime(2020, 11, 5, 18, 0),
                                    datetime(2020, 11, 6, 0, 0), datetime(2020, 11, 6, 6, 0),
                                    datetime(2020, 11, 6, 12, 0), datetime(2020, 11, 6, 18, 0),
                                    datetime(2020, 11, 7, 0, 0), datetime(2020, 11, 7, 6, 0),
                                    datetime(2020, 11, 7, 12, 0), datetime(2020, 11, 7, 18, 0),
                                    datetime(2020, 11, 8, 0, 0), datetime(2020, 11, 8, 6, 0),
                                    datetime(2020, 11, 8, 12, 0), datetime(2020, 11, 8, 18, 0),
                                    datetime(2020, 11, 9, 0, 0), datetime(2020, 11, 9, 6, 0),
                                    datetime(2020, 11, 9, 12, 0), datetime(2020, 11, 9, 18, 0),
                                    datetime(2020, 11, 10, 0, 0), datetime(2020, 11, 10, 6, 0),
                                    datetime(2020, 11, 10, 12, 0), datetime(2020, 11, 10, 18, 0),
                                    datetime(2020, 11, 11, 0, 0), datetime(2020, 11, 11, 6, 0),
                                    datetime(2020, 11, 11, 12, 0), datetime(2020, 11, 11, 18, 0)],
               'model_position_dt': datetime(2020, 11, 11, 18, 0)}
    Sun_dict = {'surface_data': 'AIA_304', 'scale': 2, 'crid': 2233}
    return Sun_dict, Planet_dict, SC_dict


if __name__ == '__main__':
    min_bc = np.array([-0.6, -0.6, -0.3], dtype=np.float64)
    max_bc = np.array([0.6, 0.6, 0.3], dtype=np.float64)
    step_num = np.array([61, 61, 21], dtype=np.int32)
    fig = go.Figure()
    sun, planet, sc = util_input()
    # sun_trace = get_sun_trace(sun)
    planet_traces = get_planet_traces(planet)
    sc_traces = get_SC_traces(sc)
    fig.add_trace(plot_dustcloud(min_bc, max_bc, step_num))
    # fig.add_trace(sun_trace)
    for sc_trace in sc_traces:
        fig.add_trace(sc_trace)
    for planet_trace in planet_traces:
        fig.add_trace(planet_trace)
    fig.update_layout(scene=dict(
        xaxis_title='', yaxis_title='', zaxis_title='',
        xaxis=dict(showaxeslabels=False, showbackground=False,
                   showgrid=False, showline=False, showticklabels=False),
        yaxis=dict(showaxeslabels=False, showbackground=False,
                   showgrid=False, showline=False, showticklabels=False),
        zaxis=dict(showaxeslabels=False, showbackground=False,
                   showgrid=False, showline=False, showticklabels=False)),
        paper_bgcolor='black'
    )
    fig.show()


