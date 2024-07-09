import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
from datetime import datetime
import furnsh_kernels  # 需要自己写的代码，用来导入已经下载好的kernel文件
import pandas as pd



AU = 1.49e8  # km

def datetimestr2et(datetime,type='%Y-%m-%d'):
    dt = datetime.strptime(datetime, type)
    # utc = dt.strftime('%b %d, %Y')
    et= spice.datetime2et(dt)
    return et

def plot_frames(obs_time):
    '''Plot PSP frame & WISPR_INNER/OUTER frame/FOV at obs_time'''
    obs_time_str = spice.et2datetime(obs_time).strftime('%Y-%m-%d %H:%M:%S')
    wispr_pos, _ = spice.spkpos('SPP', obs_time, 'SPP_HCI', 'NONE', 'SUN')  # km
    sun_pos, _ = spice.spkpos('SUN', obs_time, 'SPP_HCI', 'NONE', 'SUN')  # km
    obs_pos = wispr_pos.T / AU

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], c='black')
    wispr_inner_parameter = spice.getfov(-96100, 4)
    wispr_outer_parameter = spice.getfov(-96120, 4)

    print('inner_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][3][[2, 0, 1]])[1:3]))
    print('outer_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][3][[2, 0, 1]])[1:3]))

    # Plot PSP FRAME
    x_spp, _ = spice.spkcpt([0.3 * AU, 0, 0], 'SPP', 'SPP_SPACECRAFT', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    x_spp = x_spp / AU
    y_spp, _ = spice.spkcpt([0, 0.6 * AU, 0], 'SPP', 'SPP_SPACECRAFT', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    y_spp = y_spp / AU
    z_spp, _ = spice.spkcpt([0, 0, 0.9 * AU], 'SPP', 'SPP_SPACECRAFT', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    z_spp = z_spp / AU

    ax.plot([obs_pos[0], x_spp[0]], [obs_pos[1], x_spp[1]], [obs_pos[2], x_spp[2]], c='silver', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], y_spp[0]], [obs_pos[1], y_spp[1]], [obs_pos[2], y_spp[2]], c='silver', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], z_spp[0]], [obs_pos[1], z_spp[1]], [obs_pos[2], z_spp[2]], c='silver', linewidth=5.0,
            alpha=0.8, label='PSP_FRAME (longest for z, shortest for x)')

    # Plot INNER FRAME
    x_inner, _ = spice.spkcpt([0.3 * AU, 0, 0], 'SPP', 'SPP_WISPR_INNER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    x_inner = x_inner / AU
    y_inner, _ = spice.spkcpt([0, 0.6 * AU, 0], 'SPP', 'SPP_WISPR_INNER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    y_inner = y_inner / AU
    z_inner, _ = spice.spkcpt([0, 0, 0.9 * AU], 'SPP', 'SPP_WISPR_INNER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    z_inner = z_inner / AU

    ax.plot([obs_pos[0], x_inner[0]], [obs_pos[1], x_inner[1]], [obs_pos[2], x_inner[2]], c='lime', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], y_inner[0]], [obs_pos[1], y_inner[1]], [obs_pos[2], y_inner[2]], c='lime', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], z_inner[0]], [obs_pos[1], z_inner[1]], [obs_pos[2], z_inner[2]], c='lime', linewidth=5.0,
            alpha=0.8, label='WISPR_INNER_FRAME (longest for z, shortest for x)')

    # Plot OUTER FRAME
    x_outer, _ = spice.spkcpt([0.3 * AU, 0, 0], 'SPP', 'SPP_WISPR_OUTER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN');
    x_outer = x_outer / AU
    y_outer, _ = spice.spkcpt([0, 0.6 * AU, 0], 'SPP', 'SPP_WISPR_OUTER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN');
    y_outer = y_outer / AU
    z_outer, _ = spice.spkcpt([0, 0, 0.9 * AU], 'SPP', 'SPP_WISPR_OUTER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN');
    z_outer = z_outer / AU

    ax.plot([obs_pos[0], x_outer[0]], [obs_pos[1], x_outer[1]], [obs_pos[2], x_outer[2]], c='aqua', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], y_outer[0]], [obs_pos[1], y_outer[1]], [obs_pos[2], y_outer[2]], c='aqua', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], z_outer[0]], [obs_pos[1], z_outer[1]], [obs_pos[2], z_outer[2]], c='aqua', linewidth=5.0,
            alpha=0.8, label='WISPR_OUTER_FRAME (longest for z, shortest for x)')

    # Plot INNER/OUTER FOV
    for i_edge in range(4):
        edge_inner, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', obs_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_outer, _ = spice.spkcpt(1 * AU * wispr_outer_parameter[4][i_edge], 'SPP', 'SPP_WISPR_OUTER', obs_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_inner = edge_inner / AU
        edge_outer = edge_outer / AU
        inedge, = ax.plot([obs_pos[0], edge_inner[0]], [obs_pos[1], edge_inner[1]], [obs_pos[2], edge_inner[2]],
                          c='green')
        outedge, = ax.plot([obs_pos[0], edge_outer[0]], [obs_pos[1], edge_outer[1]], [obs_pos[2], edge_outer[2]],
                           c='blue')
    inedge.set_label('WISPR_INNER_FOV [|lon|,|lat|]=' + str(
        abs(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))))
    outedge.set_label('WISPR_OUTER_FOV [|lon|,|lat|]=' + str(
        abs(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))))

    ax.scatter(sun_pos[0], sun_pos[1], sun_pos[2], c='r')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    plt.title('PSP Frames' + '(' + obs_time_str + ')')
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.7, 0.7])
    ax.legend(loc='lower right')
    plt.show()
    return


def find_PSP_in_WISPR(insitu_time_range, wispr_time):
    ''' find future PSP positions during insitu_time_range in WISPR image imaged at wispr_time'''
    wispr_pos, _ = spice.spkpos('SPP', wispr_time, 'SPP_HCI', 'NONE', 'SUN')
    insitu_pos, _ = spice.spkpos('SPP', insitu_time_range, 'SPP_HCI', 'NONE', 'SUN')
    sun_pos, _ = spice.spkpos('SUN', insitu_time_range, 'SPP_HCI', 'NONE', 'SUN')
    obs_pos = wispr_pos.T / AU
    insitu_pos = insitu_pos.T / AU

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], c='black')
    ax.plot(insitu_pos[0], insitu_pos[1], insitu_pos[2])
    wispr_inner_parameter = spice.getfov(-96100, 4)
    wispr_outer_parameter = spice.getfov(-96120, 4)
    print('inner_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][3][[2, 0, 1]])[1:3]))
    print('outer_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][3][[2, 0, 1]])[1:3]))

    for i_edge in range(4):
        edge_inner, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', wispr_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_outer, _ = spice.spkcpt(1 * AU * wispr_outer_parameter[4][i_edge], 'SPP', 'SPP_WISPR_OUTER', wispr_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_inner = edge_inner / AU
        edge_outer = edge_outer / AU
        ax.plot([obs_pos[0], edge_inner[0]], [obs_pos[1], edge_inner[1]], [obs_pos[2], edge_inner[2]], c='green')
        ax.plot([obs_pos[0], edge_outer[0]], [obs_pos[1], edge_outer[1]], [obs_pos[2], edge_outer[2]], c='blue')

    inner_lst = []
    outer_lst = []
    print(spice.et2datetime(wispr_time).strftime('%Y-%m-%d %H:%M:%S'))
    for i in range(len(insitu_time_range)):
        print(i)
        insitu_time_tmp = insitu_time_range[i]
        insitu_utc_tmp = spice.et2datetime(insitu_time_tmp).strftime('%Y-%m-%d %H:%M:%S')

        # get pos of future PSP in the SPP_WISPR_INNER frame centered at SPP at wispr_time
        insitu_innerpos_tmp, _ = spice.spkcpt(insitu_pos[:, i] * AU, 'SUN', 'SPP_HCI', wispr_time, 'SPP_WISPR_INNER',
                                              'OBSERVER', 'NONE', 'SPP')
        insitu_outerpos_tmp, _ = spice.spkcpt(insitu_pos[:, i] * AU, 'SUN', 'SPP_HCI', wispr_time, 'SPP_WISPR_OUTER',
                                              'OBSERVER', 'NONE', 'SPP')

        # convert pos of future PSP to (r,lon,lat)
        insitu_innerpos_lat = spice.reclat(insitu_innerpos_tmp[[2, 0, 1]])
        insitu_outerpos_lat = spice.reclat(insitu_outerpos_tmp[[2, 0, 1]])

        print(np.rad2deg(insitu_innerpos_lat[1:3]))
        print(np.rad2deg(insitu_outerpos_lat[1:3]))

        '''
        <Equivalent to codes below, reserved for future debug.>
        if spice.fovray('SPP_WISPR_INNER',insitu_innerpos_tmp[0:3],'SPP_WISPR_INNER','NONE','SPP',wispr_time):
            print('PSP at '+insitu_utc_tmp+' Found in WISPR INNER FOV')
            inner_lst.append([i,insitu_innerpos_lat[0],insitu_innerpos_lat[1],insitu_innerpos_lat[2]])
            ax.scatter(insitu_pos[0,i],insitu_pos[1,i],insitu_pos[2,i],c='lime',alpha=0.5,s=10)

        if spice.fovray('SPP_WISPR_OUTER',insitu_outerpos_tmp[0:3],'SPP_WISPR_OUTER','NONE','SPP',wispr_time):
            print('PSP at'+insitu_utc_tmp+'Found in WISPR OUTER VOF')
            outer_lst.append([i,insitu_outerpos_lat[0],insitu_outerpos_lat[1],insitu_outerpos_lat[2]])
            ax.scatter(insitu_pos[0,i],insitu_pos[1,i],insitu_pos[2,i],c='aqua',alpha=0.5,s=10)
        '''
        # check whether PSP is in WISPR's FOV
        if spice.fovray('SPP_WISPR_INNER', insitu_pos[:, i] - obs_pos, 'SPP_HCI', 'NONE', 'SPP', wispr_time):
            print('PSP at ' + insitu_utc_tmp + ' Found in WISPR INNER FOV')
            inner_lst.append([insitu_time_tmp, insitu_innerpos_lat[0] / AU, np.rad2deg(insitu_innerpos_lat[1]),
                              np.rad2deg(insitu_innerpos_lat[2])])
            ax.scatter(insitu_pos[0, i], insitu_pos[1, i], insitu_pos[2, i], c='lime', alpha=0.5, s=20)

        if spice.fovray('SPP_WISPR_OUTER', insitu_pos[:, i] - obs_pos, 'SPP_HCI', 'NONE', 'SPP', wispr_time):
            print('PSP at' + insitu_utc_tmp + 'Found in WISPR OUTER VOF')
            outer_lst.append([insitu_time_tmp, insitu_outerpos_lat[0] / AU, np.rad2deg(insitu_outerpos_lat[1]),
                              np.rad2deg(insitu_outerpos_lat[2])])
            ax.scatter(insitu_pos[0, i], insitu_pos[1, i], insitu_pos[2, i], c='aqua', alpha=0.5, s=20)

    inner_lst = np.array(inner_lst)
    outer_lst = np.array(outer_lst)

    ax.scatter(sun_pos[0], sun_pos[1], sun_pos[2], c='r')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    plt.title('PSP (' + spice.et2datetime(insitu_time_range[0]).strftime('%Y-%m-%d') + '-' + spice.et2datetime(
        insitu_time_range[-1]).strftime('%Y-%m-%d') + ')\n WISPR image at ' + spice.et2datetime(wispr_time).strftime(
        '%Y-%m-%d'))
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.close()
    # plt.show()

    return inner_lst, outer_lst


def find_WISPR_for_PSP(insitu_time, wispr_time_range):
    '''For the present PSP, find the past WISPR images.'''
    wispr_pos, _ = spice.spkpos('SPP', wispr_time_range, 'SPP_HCI', 'NONE', 'SUN')
    insitu_pos, _ = spice.spkpos('SPP', insitu_time, 'SPP_HCI', 'NONE', 'SUN')
    sun_pos, _ = spice.spkpos('SUN', insitu_time, 'SPP_HCI', 'NONE', 'SUN')
    obs_pos = wispr_pos.T / AU
    insitu_pos = insitu_pos.T / AU

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(insitu_pos[0], insitu_pos[1], insitu_pos[2], c='black')
    ax.plot(obs_pos[0], obs_pos[1], obs_pos[2])
    wispr_inner_parameter = spice.getfov(-96100, 4)
    wispr_outer_parameter = spice.getfov(-96120, 4)

    print('inner_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][3][[2, 0, 1]])[1:3]))
    print('outer_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][3][[2, 0, 1]])[1:3]))

    inner_lst = []
    outer_lst = []
    for i in range(len(wispr_time_range)):
        print(i)
        wispr_time_tmp = wispr_time_range[i]
        wispr_utc_tmp = spice.et2datetime(wispr_time_tmp).strftime('%Y-%m-%d %H:%M:%S')

        insitu_innerpos_tmp, _ = spice.spkcpt(insitu_pos * AU, 'SUN', 'SPP_HCI', wispr_time_tmp, 'SPP_WISPR_INNER',
                                              'OBSERVER', 'NONE', 'SPP')
        insitu_outerpos_tmp, _ = spice.spkcpt(insitu_pos * AU, 'SUN', 'SPP_HCI', wispr_time_tmp, 'SPP_WISPR_OUTER',
                                              'OBSERVER', 'NONE', 'SPP')

        insitu_innerpos_lat = spice.reclat(insitu_innerpos_tmp[[2, 0, 1]])
        insitu_outerpos_lat = spice.reclat(insitu_outerpos_tmp[[2, 0, 1]])

        print(np.rad2deg(insitu_innerpos_lat[1:3]))
        print(np.rad2deg(insitu_outerpos_lat[1:3]))

        if spice.fovray('SPP_WISPR_INNER', insitu_pos - obs_pos[:, i], 'SPP_HCI', 'NONE', 'SPP', wispr_time_tmp):
            print('PSP Found in WISPR INNER VOF at ' + wispr_utc_tmp)
            inner_lst.append([wispr_time_tmp, insitu_innerpos_lat[0] / AU, np.rad2deg(insitu_innerpos_lat[1]),
                              np.rad2deg(insitu_innerpos_lat[2])])
            ax.scatter(obs_pos[0, i], obs_pos[1, i], obs_pos[2, i], c='lime', alpha=0.5, s=20)
            boresight_inner, _ = spice.spkcpt([0, 0, 0.5 * AU], 'SPP', 'SPP_WISPR_INNER', wispr_time_tmp, 'SPP_HCI',
                                              'CENTER', 'NONE', 'SUN')
            boresight_inner = boresight_inner / AU
            ax.plot([obs_pos[0, i], boresight_inner[0]], [obs_pos[1, i], boresight_inner[1]],
                    [obs_pos[2, i], boresight_inner[2]], c='green')
            ax.plot([obs_pos[0, i], insitu_pos[0]], [obs_pos[1, i], insitu_pos[1]], [obs_pos[2, i], insitu_pos[2]],
                    'k--')

        if spice.fovray('SPP_WISPR_OUTER', insitu_pos - obs_pos[:, i], 'SPP_HCI', 'NONE', 'SPP', wispr_time_tmp):
            print('PSP Found in WISPR OUTER VOF at ' + wispr_utc_tmp)
            outer_lst.append([wispr_time_tmp, insitu_outerpos_lat[0] / AU, np.rad2deg(insitu_outerpos_lat[1]),
                              np.rad2deg(insitu_outerpos_lat[2])])
            ax.scatter(obs_pos[0, i], obs_pos[1, i], obs_pos[2, i], c='aqua', alpha=0.5, s=20)
            boresight_outer, _ = spice.spkcpt([0, 0, 0.5 * AU], 'SPP', 'SPP_WISPR_OUTER', wispr_time_tmp, 'SPP_HCI',
                                              'CENTER', 'NONE', 'SUN')
            boresight_outer = boresight_outer / AU
            ax.plot([obs_pos[0, i], boresight_outer[0]], [obs_pos[1, i], boresight_outer[1]],
                    [obs_pos[2, i], boresight_outer[2]], c='blue')
            ax.plot([obs_pos[0, i], insitu_pos[0]], [obs_pos[1, i], insitu_pos[1]], [obs_pos[2, i], insitu_pos[2]],
                    'k--')

    inner_lst = np.array(inner_lst)
    outer_lst = np.array(outer_lst)

    ax.scatter(sun_pos[0], sun_pos[1], sun_pos[2], c='r')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    plt.title('PSP (' + spice.et2datetime(wispr_time_range[0]).strftime('%Y-%m-%d') + '-' + spice.et2datetime(
        wispr_time_range[-1]).strftime('%Y-%m-%d') + ')\n INSITU observation made at ' + spice.et2datetime(
        insitu_time).strftime('%Y-%m-%d'))
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.show()
    return inner_lst, outer_lst

