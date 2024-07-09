import os
import requests

import numpy as np
from datetime import datetime, timedelta
import pyvista

import sunpy.map
import sunpy.coordinates.sun as sun

import plotly.graph_objects as go
import spiceypy as spice

from ps_read_hdf_3d import ps_read_hdf_3d


Rs = 696300.
AU = 1.496e8
unit_lst = {'l': 696300,  # km
            'vr': 481.3711, 'vt': 481.3711, 'vp': 481.3711,  # km/s
            'ne': 1e14, 'rho': 1e8, 'p': 3.875717e-2, 'T': 2.807067e7,
            'br': 2.2068908e5, 'bt': 2.2068908e5, 'bp': 2.2068908e5,  # nT
            'j': 2.267e4}


def plot_planet_model(planet_str, dt):
    et = spice.datetime2et(dt)
    planet_pos, _ = spice.spkpos(planet_str, et, 'IAU_SUN', 'NONE', 'SUN')
    planet_pos = np.array(planet_pos).T / AU

    import matplotlib.pyplot as plt
    img = plt.imread('Planet_model/' + planet_str + '.jpeg')
    from PIL import Image
    eight_bit_img = Image.fromarray(img).convert('P', palette='WEB', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape(-1, 3)
    colorscale = [[i / 255.0, 'rgb({},{},{})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    lon = np.linspace(0, 2 * np.pi, img.shape[1])
    lat = np.linspace(np.pi / 2, -np.pi / 2, img.shape[0])
    x0, y0, z0 = get_sphere(8, lon, lat)
    print('Plotting ' + planet_str)
    trace = go.Surface(x=x0 + planet_pos[0], y=y0 + planet_pos[1], z=z0 + planet_pos[2],
                       surfacecolor=np.array(eight_bit_img), cmin=0, cmax=255, colorscale=colorscale, showscale=False)
    return trace


def get_sphere(R0, lon, lat, for_PSI=False):
    if for_PSI:
        lat = np.pi / 2 - lat
    llon, llat = np.meshgrid(lon, lat)
    x0 = R0 * Rs / AU * np.cos(llon) * np.cos(llat)
    y0 = R0 * Rs / AU * np.sin(llon) * np.cos(llat)
    z0 = R0 * Rs / AU * np.sin(llat)
    return x0, y0, z0

def plot_HPS(crid, isos_value = 3e5, opacity = 0.8, color = 'azure'):
    data = ps_read_hdf_3d(crid, 'corona', 'rho002', periodicDim=3)
    r = np.array(data['scales1'])  # 201 in Rs, distance from sun
    t = np.array(data['scales2'])  # 150 in rad, latitude
    p = np.array(data['scales3'])  # 256 in rad, Carrington longitude
    rho = np.array(data['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
    rho = rho * 1e8  # cm^-3
    tv, pv, rv = np.meshgrid(t, p, r, indexing='xy')

    xv = rv * np.cos(pv) * np.sin(tv)
    yv = rv * np.sin(pv) * np.sin(tv)
    zv = rv * np.cos(tv)

    mesh = pyvista.StructuredGrid(xv, yv, zv)
    mesh.point_data['values'] = rho.ravel(order='F')  # also the active scalars
    isos = mesh.contour(isosurfaces=1, rng=[isos_value,isos_value])

    vertices = isos.points
    triangles = isos.faces.reshape(-1, 4)
    trace = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                          opacity=opacity,
                          i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                          color=color, )
    return trace



def plot_HCS(crid, region, surface_dict):
    data = ps_read_hdf_3d(crid, region, 'br002', periodicDim=3)
    r = np.array(data['scales1'])  # 201 in Rs, distance from sun
    t = np.array(data['scales2'])  # 150 in rad, latitude
    p = np.array(data['scales3'])  # 256 in rad, Carrington longitude
    br = np.array(data['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
    br = br * 2.205e5  # nT

    tv, pv, rv = np.meshgrid(t, p, r, indexing='xy')

    xv = rv * np.cos(pv) * np.sin(tv)
    yv = rv * np.sin(pv) * np.sin(tv)
    zv = rv * np.cos(tv)

    mesh = pyvista.StructuredGrid(xv, yv, zv)
    mesh.point_data['values'] = br.ravel(order='F')  # also the active scalars
    isos = mesh.contour(isosurfaces=1, rng=[0, 0])

    vertices = isos.points
    triangles = isos.faces.reshape(-1, 4)

    if surface_dict['param'] != None:
        param = surface_dict['param']
        param_points = vertices[:, 0] * 0.

        data = ps_read_hdf_3d(crid, region, param + '002', periodicDim=3)
        r_param = np.array(data['scales1'])
        t_param = np.array(data['scales2'])
        p_param = np.array(data['scales3'])
        param_data = np.array(data['datas'])
        param_data = param_data * unit_lst[param]

        for i in range(len(vertices)):
            point = np.array(vertices[i])
            r_point, p_point, t_point = xyz2rtp_in_Carrington(point, for_psi=True)

            r_ind = np.argmin(abs(r_point - r_param))
            p_ind = np.argmin(abs(p_point - p_param))
            t_ind = np.argmin(abs(t_point - t_param))

            param_points[i] = param_data[p_ind, t_ind, r_ind]  # * (r_p) ** 2

        intensity = np.array(param_points).reshape(-1, 1)
        if surface_dict['clim'] == None:
            surface_dict['clim'] = [np.nanmin(intensity), np.nanmax(intensity)]

        trace = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                          opacity=surface_dict['opacity'],
                          colorscale=surface_dict['cmap'],
                          cmin=surface_dict['clim'][0], cmax=surface_dict['clim'][1],
                          i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                          intensity=intensity,
                          showscale=surface_dict['showscale'], )
    else:
        trace = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                          opacity=surface_dict['opacity'],
                          i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                          color=surface_dict['color'], )
    return trace


def xyz2rtp_in_Carrington(xyz_carrington, for_psi=False):
    """
    Convert (x,y,z) to (r,p,t) in Carrington Coordination System.
        (x,y,z) follows the definition of SPP_HG in SPICE kernel.
        (r,lon,lat) is (x,y,z) converted to heliographic lon/lat, where lon \in [0,2pi], lat \in [-pi/2,pi/2] .
    :param xyz_carrington:
    :return:
    """
    r_carrington = np.linalg.norm(xyz_carrington[0:3], 2)

    lon_carrington = np.arcsin(xyz_carrington[1] / np.sqrt(xyz_carrington[0] ** 2 + xyz_carrington[1] ** 2))
    if np.shape(xyz_carrington[0]) != ():
        lon_carrington[xyz_carrington[0] < 0] = np.pi - lon_carrington[xyz_carrington[0] < 0]
    else:
        if xyz_carrington[0] < 0:
            lon_carrington = np.pi - lon_carrington
    # if lon_carrington < 0:
    lon_carrington = lon_carrington % (2*np.pi)

    lat_carrington = np.pi / 2 - np.arccos(xyz_carrington[2] / r_carrington)
    if for_psi:
        lat_carrington = np.pi / 2 - lat_carrington
    return r_carrington, lon_carrington, lat_carrington


def plot_object_orbit(obj_name, dt_epoch, type='line'):
    et_epoch = spice.datetime2et(dt_epoch)
    pos, _ = spice.spkpos(obj_name, et_epoch, 'IAU_SUN', 'NONE', 'SUN')
    pos = np.array(pos.T / AU)
    pos = {'x': pos[0], 'y': pos[1], 'z': pos[2]}

    trace = []
    if type == 'line':
        trace = go.Scatter3d(x=pos['x'], y=pos['y'], z=pos['z'],
                             mode='lines',
                             line=dict(color='skyblue', width=5))
    elif type == 'marker':
        trace = go.Scatter3d(x=pos['x'], y=pos['y'], z=pos['z'],
                             mode='markers',
                             marker=dict(size=1))
    return trace


def get_HCS_traces(HCS_dict):
    HCS_trace_list = []
    crid = HCS_dict['crid']
    surface_dict = HCS_dict['surface']

    if HCS_dict['plot_sc']:
        HCS_trace_list.append(plot_HCS(crid, 'corona', surface_dict))
    if HCS_dict['plot_ih']:
        HCS_trace_list.append(plot_HCS(crid, 'helio', surface_dict))

    return HCS_trace_list

def get_HPS_traces(HPS_dict):
    HPS_trace_list = []
    crid = HPS_dict['crid']
    isos_value = HPS_dict['isos_value']
    opacity = HPS_dict['opacity']
    color = HPS_dict['color']
    HPS_trace_list.append(plot_HPS(crid,isos_value=isos_value,opacity=opacity,color=color))
    return HPS_trace_list

def plot_Spacecraft_model(SC_str, dt, scale=0.1):
    et = spice.datetime2et(dt)
    SC_pos, _ = spice.spkpos(SC_str, et, 'IAU_SUN', 'NONE', 'SUN')
    SC_pos = np.array(SC_pos).T / AU
    trans_arr = np.eye(3)
    if SC_str == 'SOLO':
        trans_arr = spice.sxform('SOLO_SRF', 'IAU_SUN', et)
        trans_arr = np.dot(trans_arr[0:3, 0:3], np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    elif SC_str == 'SPP':
        trans_arr = spice.sxform('SPP_SPACECRAFT', 'IAU_SUN', et)
        trans_arr = np.dot(trans_arr[0:3, 0:3], np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

    trans_arr_4 = np.eye(4)
    trans_arr_4[0:3, 0:3] = trans_arr

    mesh = pyvista.read('SC_model/' + SC_str + '.stl')
    mesh.points = scale * mesh.points / np.max(mesh.points)

    axes = pyvista.Axes(show_actor=True, actor_scale=100.0, line_width=10)
    axes.origin = (0, 0, 0)

    trans = mesh.transform(trans_arr_4)

    # p = pyvista.Plotter()
    # p.add_actor(axes.actor, pickable=True)
    # p.add_mesh(mesh)
    # # p.add_mesh(trans)
    # p.show_grid()
    # p.show_axes()
    # p.show()

    vertices = trans.points
    triangles = trans.faces.reshape(-1, 4)
    trace = go.Mesh3d(x=vertices[:, 0] + SC_pos[0], y=vertices[:, 1] + SC_pos[1], z=vertices[:, 2] + SC_pos[2],
                      opacity=1,
                      color='gold',
                      i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                      showscale=False,
                      )
    return trace


def get_SC_traces(SC_dict):
    SC_trace_list = []
    SC_str = SC_dict['SC_str']
    if len(SC_dict['orbit_epoch']) != 0:
        traces_orbit = plot_object_orbit(SC_str, SC_dict['orbit_epoch'], 'line')
        SC_trace_list.append(traces_orbit)

    # if len(SC_dict['orbit_marker_epoch']) != 0:
    #     traces_marker = plot_object_orbit(SC_str, SC_dict['orbit_marker_epoch'], 'marker')
    #     SC_trace_list.append(traces_marker)
    if SC_dict['model_position_dt'] != None:
        trace_model = plot_Spacecraft_model(SC_str, SC_dict['model_position_dt'])
        SC_trace_list.append(trace_model)
    # if len(SC_dict['trace_epoch']) != 0:
    #     traces_trace_list = plot_obj_magtrace(SC_str, SC_dict['trace_epoch'])
    #     for trace in traces_trace_list:
    #         SC_trace_list.append(trace)
    return SC_trace_list


AIA_193_dir = 'AIA_193_map/'


def load_AIA_data(crid):
    aia_dir = AIA_193_dir
    # aia_filename = 'aia193_synmap_cr' + str(crid) + '.fits'
    aia_filename = 'aia193_synmap_cr' + str(crid) + '.fits'
    aia_file = os.path.join(aia_dir, aia_filename)
    if not os.path.exists(aia_file):
        aia_url = 'https://sun.njit.edu/coronal_holes/data/' + aia_filename
        aia_r = requests.get(aia_url)
        with open(aia_file, 'wb') as f:
            f.write(aia_r.content)
            f.close()
    syn_map = sunpy.map.Map(aia_file)
    map_data = syn_map.data
    map_cmap = syn_map.cmap
    map_x = np.linspace(syn_map.bottom_left_coord[0].value, syn_map.top_right_coord[0].value,
                        int(syn_map.dimensions.x.value))
    map_y = np.linspace(syn_map.bottom_left_coord[1].value, syn_map.top_right_coord[1].value,
                        int(syn_map.dimensions.y.value))
    lon = np.deg2rad(map_x)
    lat = np.deg2rad(map_y)
    return lon, lat, map_data, map_cmap


def get_sun_trace(Sun_dict):
    crid = Sun_dict['crid']
    if Sun_dict['surface_data'] == 'br':
        data_sc = ps_read_hdf_3d(crid, 'corona', 'br002', periodicDim=3)
        t_br = np.array(data_sc['scales2'])  # 150 in rad, latitude
        p_br = np.array(data_sc['scales3'])  # 256 in rad, Carrington longitude
        br = np.array(data_sc['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
        br = br * 2.205e5  # nT
        sundata = np.squeeze(br[:, :, 0]).T
        x0, y0, z0 = get_sphere(Sun_dict['scale'], p_br, t_br, for_PSI=True)

        return go.Surface(x=x0, y=y0, z=z0, surfacecolor=sundata, colorscale='rdbu', opacity=1, showscale=False,
                          cmax=3e5,
                          cmin=-3e5)
    elif Sun_dict['surface_data'] == 'AIA_193':
        lon, lat, sundata, map_cmap = load_AIA_data(crid)
        x0, y0, z0 = get_sphere(Sun_dict['scale'], lon, lat)
        return go.Surface(x=x0, y=y0, z=z0, surfacecolor=np.squeeze(sundata), colorscale='turbid_r', opacity=1,
                          showscale=False)
    elif Sun_dict['surface_data'] == 'AIA_304':
        import matplotlib.pyplot as plt
        img = plt.imread('AIA_193_map/' + 'aia_304' + '.jpg')
        from PIL import Image
        eight_bit_img = Image.fromarray(img).convert('P', palette='WEB', dither=None)
        idx_to_color = np.array(eight_bit_img.getpalette()).reshape(-1, 3)
        colorscale = [[i / 255.0, 'rgb({},{},{})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
        lon = np.linspace(0, 2 * np.pi, img.shape[1])
        lat = np.linspace(np.pi / 2, -np.pi / 2, img.shape[0])
        x0, y0, z0 = get_sphere(Sun_dict['scale'], lon, lat)
        print('Plotting ' + 'Sun')
        trace = go.Surface(x=x0, y=y0, z=z0,
                           surfacecolor=np.array(eight_bit_img), cmin=0, cmax=255, colorscale=colorscale,
                           showscale=False)
        return trace

def get_planet_traces(Planet_dict):
    Planet_list = Planet_dict['Planet_list']
    Planet_orbit_epoch = Planet_dict['Planet_epoch']
    Planet_model_dt = Planet_dict['Planet_model_dt']
    Planet_trace_list = []
    for Planet_name in Planet_list:
        if Planet_orbit_epoch != []:
            orbit_trace = plot_object_orbit(Planet_name, Planet_orbit_epoch, type='line')
            Planet_trace_list.append(orbit_trace)
        # if Planet_trace_epoch != []:
        #     magtrace_trace_list = plot_obj_magtrace(Planet_name, Planet_trace_epoch)
        #     for magtrace_trace in magtrace_trace_list:
        #         Planet_trace_list.append(magtrace_trace)
        if Planet_model_dt != None:
            Planet_model_trace = plot_planet_model(Planet_name, Planet_model_dt)
            Planet_trace_list.append(Planet_model_trace)
    return Planet_trace_list


def create_epoch(range_dt, step_td):
    beg_dt = range_dt[0]
    end_dt = range_dt[1]
    return [beg_dt + n * step_td for n in range((end_dt - beg_dt) // step_td)]


if __name__ == '__main__':
    plot = go.Figure()
    plot.add_trace(plot_Spacecraft_model('SPP', datetime(2020, 8, 2), scale=100))
    import plotly

    plotly.offline.plot(plot)
