import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import special as sp
import sunpy.io
import sunpy.map
import spiceypy as spice
from datetime import datetime

import furnsh_all_kernels

I_0 = 1361  # W/m^2 Solar Irradiance
AU = 1.49e11  # distance from sun to earth
Rs = 6.96e8  # solar radii


def add_right_cax(ax, pad, width):
    """
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    :param ax:
    :param pad:
    :param width:
    :return:
    """
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


def n_r(x):
    """
    :param x: normalized radial distance. x = r / r_0
    :return:
    """
    r_in = 5 * Rs / AU
    r_out = 19 * Rs / AU
    nu = 1.3
    return x ** (-nu) * (x > r_out) + x ** (-nu) * (x - r_in) / (r_out - r_in) * np.logical_and(x <= r_out, x >= r_in)


def obtain_VSF0(a, Nmr0, theta_rad):
    wavelength = 550e-9  # m
    a, theta_rad_mesh = np.meshgrid(a, theta_rad, indexing='ij')
    Nmr0, theta_rad_mesh = np.meshgrid(Nmr0, theta_rad, indexing='ij')
    alpha = 2 * np.pi * a / wavelength
    albedo = 0.25  # bond albedo
    sigma = a ** 2 * abs(sp.jv(1, alpha * np.sin(theta_rad_mesh))) ** 2 / \
            abs(np.sin(theta_rad_mesh)) ** 2 + albedo * a ** 2 / 4  # m^2
    delta_alpha = abs(np.mean(np.diff(alpha, axis=0)))
    # vsf0 = np.nansum(sigma[1:] * (np.diff(Nmr0) / np.diff(alpha)) * delta_alpha)  # m^-1
    vsf0 = np.nansum((sigma[1:, :] + sigma[:-1, :]) / 2 *
                     abs(np.diff(Nmr0, axis=0)),
                     axis=0)  # m^-1
    return vsf0


def get_Brightness_of_F_Corona(fov_angle, SC_pos_carr, show_dist_var=False, ax=None,
                               show_xlabel=True, show_ylabel=True, title_name="", show_legend=False):
    """
    :param fov_angle: deg
    :param SC_pos_carr: meter
    :param show_dist_var:
    :return:
    """
    # Constants & Geometry
    beta_rad = fov_angle[0]  # 经度方向上的方位角
    gamma_rad = fov_angle[1]  # 纬度方向上的方位角

    cos_elongation = (1 + np.tan(beta_rad) ** 2 + np.tan(gamma_rad) ** 2) ** (-1 / 2)
    elongation_rad = np.arccos(cos_elongation)

    x_sc_carr, y_sc_carr, z_sc_carr = SC_pos_carr[0], SC_pos_carr[1], SC_pos_carr[2]
    R_sc = np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2 + z_sc_carr ** 2)

    # theta_rad = np.linspace(elongation_rad, np.pi, 180)
    # l_los = R_sc * np.sin(theta_rad - elongation_rad) / np.sin(theta_rad)
    l_los = np.linspace(0, R_sc*10, 800)
    theta_rad = np.arctan(np.sin(elongation_rad) /
                          (l_los / R_sc
                           - np.cos(elongation_rad)))
    theta_rad[theta_rad < 0] = theta_rad[theta_rad < 0] + np.pi
    radial_dist_los = R_sc * np.sin(elongation_rad) / np.sin(theta_rad)

    carr2fov_arr = np.array(
        [[-x_sc_carr * z_sc_carr / R_sc ** 2, -y_sc_carr * z_sc_carr / R_sc ** 2, 1 - z_sc_carr ** 2 / R_sc ** 2],
         [y_sc_carr / R_sc, -x_sc_carr / R_sc, 0],
         [x_sc_carr / R_sc, y_sc_carr / R_sc, z_sc_carr / R_sc]])
    carr2fov_arr = np.array(
        [[y_sc_carr / R_sc, -x_sc_carr / R_sc, 0],
         [-x_sc_carr * z_sc_carr / R_sc ** 2, -y_sc_carr * z_sc_carr / R_sc ** 2, 1 - z_sc_carr ** 2 / R_sc ** 2],
         [x_sc_carr / R_sc, y_sc_carr / R_sc, z_sc_carr / R_sc]])
    fov2carr_arr = np.array([
        [-y_sc_carr / np.sqrt(x_sc_carr**2 + y_sc_carr**2), x_sc_carr * z_sc_carr / R_sc / np.sqrt(x_sc_carr**2 + y_sc_carr**2), -x_sc_carr / R_sc],
        [x_sc_carr / np.sqrt(x_sc_carr**2 + y_sc_carr**2), y_sc_carr * z_sc_carr / R_sc / np.sqrt(x_sc_carr**2 + y_sc_carr**2), -y_sc_carr / R_sc],
        [0, -np.sqrt(x_sc_carr**2 + y_sc_carr**2) / R_sc, -z_sc_carr / R_sc]])
    if x_sc_carr == 0 and y_sc_carr == 0:
        fov2carr_arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    else:
        fov2carr_arr = np.linalg.inv(carr2fov_arr)
    xp_fov = l_los * np.tan(beta_rad) * cos_elongation
    yp_fov = l_los * np.tan(gamma_rad) * cos_elongation
    zp_fov = - l_los * cos_elongation

    sc_p_carr = np.dot(fov2carr_arr, np.vstack((xp_fov, yp_fov, zp_fov)))
    # sc_p_carr = [-zp_fov, ]
    xp_carr = sc_p_carr[0] + x_sc_carr
    yp_carr = sc_p_carr[1] + y_sc_carr
    zp_carr = sc_p_carr[2] + z_sc_carr

    cos2lat = 1 - zp_carr ** 2 / (xp_carr ** 2 + yp_carr ** 2 + zp_carr ** 2)
    sin2lat = 1 - cos2lat
    # Size & Spatial Distribution of Dust
    a = np.linspace(1e-7, 1e-4, 1000)  # Dust size (m)
    rho = 2.5e6  # g/m^3
    m = (4. / 3) * np.pi * a ** 3 * rho  # g
    c1 = 2.2e3
    c2 = 15.
    c3 = 1.3e-9
    c4 = 1e11
    c5 = 1e27
    c6 = 1.3e-16
    c7 = 1e6
    g1 = 0.306
    g2 = -4.38
    g3 = 2.
    g4 = 4.
    g5 = -0.36
    g6 = 2.
    g7 = -0.85
    Fmr0 = (c1 * m ** g1 + c2) ** g2 \
           + c3 * (m + c4 * m ** g3 + c5 * m ** g4) ** g5 \
           + c6 * (m + c7 * m ** g6) ** g7  # m^-2s^-1
    v0 = 20e3  # m/s
    Nmr0 = 4. * Fmr0 / v0  # m^-3
    # print(np.sum(Nmr0 * 10**1.3 * 1e9 * (a[1] - a[0])) / np.sum(a[-1] - a[0]))
    # print(Fmr0[10], Nmr0[10] * 30 ** 1.3 * 1e9)
    # VSF
    # VSF0 = []
    VSF0 = obtain_VSF0(a, Nmr0, theta_rad)
    # for i in range(0, len(theta_rad)):
    #     VSF0.append(obtain_VSF0(a, Nmr0, theta_rad[i]))  # m^-1
    # plt.axes(yscale = 'log')
    # plt.plot(scatter_angle,VSF0)
    # plt.xlabel('Scatter Angle')
    # plt.ylabel('VSF(r0,theta)')
    # plt.show()
    # LOS Integral
    nu = 1.3
    delta_rad = -np.diff(theta_rad)
    delta_rad = np.append(delta_rad, delta_rad[-1])
    if show_dist_var:
        # I_plot = [I_0 * AU * (1 * np.sin(elongation_rad)) ** (-nu - 1) * \
        #           np.nansum(np.sin(theta_rad[:plot_i]) ** nu * (0.15 + 0.85 * cos2lat[:plot_i] ** 14) *
        #           np.array(VSF0)[:plot_i] * delta_rad) for plot_i in range(theta_rad.size)]
        I_plot = np.array([I_0 * np.nansum((radial_dist_los[:plot_i] / AU) ** (-2) *
                                           n_r(radial_dist_los[:plot_i] / AU) * R_sc * np.sin(elongation_rad) /
                                           np.sin(theta_rad[:plot_i])**2 *
                                           np.exp(-3.8 * np.sqrt(sin2lat[:plot_i]) + 2 * sin2lat[:plot_i]) *
                                           np.array(VSF0)[:plot_i] * delta_rad[:plot_i] * 1e12)
                           for plot_i in range(theta_rad.size)]) / I_0 * 4.50 / 6.61 * 1e-4
        l_los_sta = np.where(I_plot > np.max(I_plot) * 0.15)[0][0]
        l_los_end = np.where(I_plot > np.max(I_plot) * 0.5)[0][0]
        lin_analysis_l = l_los[np.logical_and(I_plot > np.max(I_plot) * 0.15,
                                               I_plot < np.max(I_plot) * 0.5)]
        lin_analysis_I = I_plot[np.logical_and(I_plot > np.max(I_plot) * 0.15,
                                               I_plot < np.max(I_plot) * 0.5)]
        n_analysis = lin_analysis_I.size
        k_fit = (np.sum((lin_analysis_l - np.mean(lin_analysis_l)) * (lin_analysis_I - np.mean(lin_analysis_I))) /
                 np.sum((lin_analysis_l - np.mean(lin_analysis_l))**2))
        b_fit = np.mean(lin_analysis_I) - np.mean(lin_analysis_l) * k_fit
        corr_fit = ((np.sum(lin_analysis_l * lin_analysis_I) -
                    n_analysis * np.mean(lin_analysis_l) * np.mean(lin_analysis_I)) /
                    np.sqrt((np.sum(lin_analysis_l**2) - n_analysis * np.mean(lin_analysis_l)**2) *
                            (np.sum(lin_analysis_I**2) - n_analysis * np.mean(lin_analysis_I)**2)))
        ax.set_xlim(0, l_los[-1] / AU)
        ax.set_ylim(0, I_plot[-1])
        if show_legend:
            ax.scatter(l_los[:-1] / AU, I_plot[:-1], label='Model')
            # ax.plot(l_los / AU, l_los * k_fit + b_fit, '-', color='r', lw=2, label='Linear Fit')
        else:
            ax.scatter(l_los[:-1] / AU, I_plot[:-1])
            # ax.plot(l_los / AU, l_los * k_fit + b_fit, '-', color='r', lw=2)
        # ax.text(lin_analysis_l[int(n_analysis / 2)] / AU + 0.005, lin_analysis_I[int(n_analysis / 2)],
        #         r"$\mathrm{\rho_{xy}}$ = %.5f" % corr_fit, color='red', fontsize=14)
        # ax_yy = ax.twinx()
        # I_plot_diff = np.diff(I_plot) / np.diff(l_los)
        # ax_yy.plot(l_los[:-2] / AU, I_plot_diff[:-1], '-', color='red', lw=2)
        # I = I_0 * AU * (1 * np.sin(elongation_rad)) ** (-nu - 1) * \
        #     np.nansum(np.sin(theta_rad) ** nu * (0.15 + 0.85 * cos2lat ** 14) * np.array(VSF0) * delta_rad)
        I = I_0 * \
            np.nansum((radial_dist_los / AU) ** (-2) * n_r(radial_dist_los / AU) * R_sc * np.sin(elongation_rad) /
                      np.sin(theta_rad)**2 *
                      np.exp(-3.8 * np.sqrt(sin2lat) +
                             2 * sin2lat) * np.array(VSF0) * delta_rad) / I_0 * 4.50 / 6.61 * 1e-4
        # ax.set_xlim(l_los[l_los_sta] / AU, l_los[l_los_end] / AU)
        # ax.set_ylim(I_plot[l_los_sta], I_plot[l_los_end])
        ax.set_title(title_name, fontsize=16)
        ax.tick_params(labelsize=13)
        if show_xlabel:
            ax.set_xlabel("LOS Distance    [AU]", fontsize=15)
        if show_ylabel:
            ax.set_ylabel(r"Brightness    $\mathrm{[pMSB]}$", fontsize=15)
        ax.grid(linestyle='--')
        # plt.show()
    else:
        # I = I_0 * AU * (1 * np.sin(elongation_rad)) ** (-nu - 1) * \
        #     np.nansum(np.sin(theta_rad) ** nu * (0.15 + 0.85 * cos2lat ** 14) * np.array(VSF0) * delta_rad)
        I = I_0 * \
            np.nansum((radial_dist_los / AU) ** (-2) * n_r(radial_dist_los / AU) * R_sc * np.sin(elongation_rad) /
                      np.sin(theta_rad)**2 *
                      np.exp(-3.8 * np.sqrt(sin2lat) +
                             2 * sin2lat) * np.array(VSF0) * delta_rad) / I_0 * 4.50 / 6.61 * 1e-4
    return I


def wispr_image_forward_model(wispr_l2b_datapath, output_path="D:/Desktop/"):
    """
    Generate a model image for the same time of observation.   --- Tianhang Chen, May 08, 2024
    :param wispr_l2b_datapath:
    :param output_path:
    :return:
    """
    fits_data, fits_header = sunpy.io.read_file(wispr_l2b_datapath)[0]
    model_data = np.zeros_like(fits_data)
    y_size, x_size = fits_data.shape
    fits_datetime_str = fits_header["DATE-OBS"]
    fits_datetime = datetime.strptime(fits_datetime_str, '%Y-%m-%dT%H:%M:%S.%f')
    fits_etime = spice.datetime2et(fits_datetime)
    psp_j2000_pos, light_time = spice.spkpos('SPP', fits_etime, 'ECLIPJ2000', 'NONE', 'SUN')
    psp_j2000_pos = np.array(psp_j2000_pos) * 1e3  # unit: m

    fov_x_l = 1920 * 10e-6
    fov_y_l = 2048 * 10e-6
    fov_z_l = 28e-3
    tmp_fov_pos = np.zeros((3,), dtype=np.float_)
    for x_i in range(0, x_size, 5):
        for y_i in range(0, y_size, 5):
            tmp_fov_pos[0] = x_i / fits_data.shape[1] * fov_x_l - fov_x_l / 2
            tmp_fov_pos[1] = y_i / fits_data.shape[0] * fov_y_l - fov_y_l / 2
            tmp_fov_pos[2] = fov_z_l
            tmp_j2000_pos, _ = spice.spkcpt(tmp_fov_pos, 'SPP', 'SPP_WISPR_INNER', fits_etime,
                                            'ECLIPJ2000', 'OBSERVER', 'NONE', 'SPP')
            tmp_j2000_pos = np.array(tmp_j2000_pos[0:3])  # unit: m
            tmp_pos_lat = np.arcsin(tmp_j2000_pos[2] / np.sqrt(np.sum(tmp_j2000_pos ** 2))) * 180 / np.pi
            tmp_pos_lon = np.arccos(-((tmp_j2000_pos[0] * psp_j2000_pos[0] + tmp_j2000_pos[1] * psp_j2000_pos[1]) /
                                      np.sqrt(np.sum(tmp_j2000_pos[0:2] ** 2) * np.sum(
                                          psp_j2000_pos[0:2] ** 2)))) * 180 / np.pi
            tmp_pos_lat_rad = tmp_pos_lat * np.pi / 180
            tmp_pos_lon_rad = tmp_pos_lon * np.pi / 180
            print(x_i, y_i)
            model_data[y_i, x_i] = get_Brightness_of_F_Corona([tmp_pos_lon_rad, tmp_pos_lat_rad],
                                                              psp_j2000_pos, show_dist_var=False)
    np.save(output_path + "ModelData", model_data)
    np.save(output_path + "ObsData", fits_data)
    return model_data, fits_data


if __name__ == '__main__':
    # %%
    dist_sc = 0.3
    lon_rad = np.linspace(0, np.pi / 2, 90)
    lat_rad = np.linspace(-np.pi / 6, np.pi / 6, 60)
    # I_F = np.zeros((len(lon_rad), len(lat_rad)))
    # i_plot = 89
    # j_plot = 29
    # # for i in range(0, len(lon_rad)):  # unit: degree.
    # #     for j in range(0, len(lat_rad)):
    # # fig, axes = plt.subplots(2, 3)
    # plot_i_all = [30, 45, 59]
    # plot_j_all = [29, 59]
    # for i in range(len(plot_i_all)):  # unit: degree.
    #     for j in range(len(plot_j_all)):
    #         show_x = True
    #         show_y = True
    #         show_legend = False
    #         if i > 0:
    #             show_y = False
    #         if j == 0:
    #             show_x = False
    #         if j == 1 and i == 2:
    #             show_legend = True
    #         I_tmp = get_Brightness_of_F_Corona([lon_rad[plot_i_all[i]], lat_rad[plot_j_all[j]]],
    #                                            [dist_sc*AU * np.sin(np.pi / 3 * 2),
    #                                             dist_sc*AU * np.cos(np.pi / 3 * 2), 0],
    #                                            show_dist_var=True, ax=axes[j, i],
    #                                            show_xlabel=show_x, show_ylabel=show_y,
    #                                            title_name=r"$\gamma$ = %d${}^\circ$, $\varphi$ = %d${}^\circ$"
    #                                                       % (int(lat_rad[plot_j_all[j]] * 180 / np.pi),
    #                                                          int(lon_rad[plot_i_all[i]] * 180 / np.pi)),
    #                                            show_legend=show_legend)
    # plt.suptitle("r = %.1f  AU" % dist_sc, fontsize=18)
    # axes[1, 2].legend(fontsize=12,  bbox_to_anchor=(0.95, 1., 0.5, 0.25), ncols=1)
    fig_2, axis_2 = plt.subplots(1, 1)
    I_tmp = get_Brightness_of_F_Corona([lon_rad[30], lon_rad[30]],
                                       [dist_sc * AU * np.sin(np.pi / 3 * 2),
                                        dist_sc * AU * np.cos(np.pi / 3 * 2), 0],
                                       show_dist_var=True, ax=axis_2,
                                       show_xlabel=True, show_ylabel=True,
                                       title_name=r"$\gamma$ = %d${}^\circ$, $\varphi$ = %d${}^\circ$"
                                                  % (int(lon_rad[30] * 180 / np.pi),
                                                     int(lon_rad[30] * 180 / np.pi)),
                                       show_legend=True)
    # plt.suptitle("r = %.1f  AU" % dist_sc, fontsize=18)
    plt.show()
    # for i in range(i_plot, i_plot + 1):  # unit: degree.
    #     for j in range(j_plot, j_plot + 1):
    #         if i == i_plot and j == j_plot:
    #             I_tmp = get_Brightness_of_F_Corona([lon_rad[i], lat_rad[j]],
    #                                                [dist_sc*AU * np.sin(np.pi / 3 * 2),
    #                                                 dist_sc*AU * np.cos(np.pi / 3 * 2), 0],
    #                                                show_dist_var=True)
    #         else:
    #             I_tmp = get_Brightness_of_F_Corona([lon_rad[i], lat_rad[j]],
    #                                                [dist_sc*AU*np.sin(np.pi/3*2), dist_sc*AU*np.cos(np.pi/3*2), 0])

    #         I_F[i, j] = I_tmp

    # [lonv, latv] = np.meshgrid(lon_rad, lat_rad)
    # I_F = np.array(I_F)
    # # %%
    # plt.contourf(np.rad2deg(lon_rad), np.rad2deg(lat_rad), np.log10(np.transpose(I_F)), levels=20)
    # plt.axis('scaled')
    # plt.xlabel('beta_fov (Degree)')
    # plt.ylabel('gamma_fov (Degree)')
    # plt.title('log10(I_F)')
    # plt.colorbar()
    # # plt.clim([-6,-3])
    # plt.show()

    # I_F = np.zeros((len(lon),1))
    # for i in range(0, len(lon)):
    #     I_tmp = get_Brightness_of_F_Corona([lon[i],0])
    #     I_F[i] = I_tmp
    # print(I_F[19])
    # plt.plot(lon, I_F)
    # plt.yscale('log')
    # plt.xlabel('view_field_theta (Degree, in ecliptic plane)')
    # plt.ylabel('White Light Intensity of F corona (W m^-2 sr^-1)')
    # plt.show()
    ####################################################################################
    # l2b_datapath = ("D:/Microsoft Download/Formal Files/data file/FITS/WISPR/L2_Background/"
    #                 "WISPR_ENC10_L2b_FITS/20211118/"
    #                 "psp_L2b_wispr_20211118T004815_V1_1221.fits")
    # output_path = ("D:/Microsoft Download/Formal Files/data file/Formatted Data/NPY/"
    #                "WISPR_brightness_fwd_model_20240511/")
    # a_data, a_header = sunpy.io.read_file(l2b_datapath)[0]
    # a_map = sunpy.map.Map(a_data, a_header)
    #
    # # model_data, obs_data = wispr_image_forward_model(l2b_datapath, output_path=output_path)
    # model_data = np.load(output_path + "ModelData.npy") * 1e12
    # obs_data = np.load(output_path + "ObsData.npy") * 1e12
    # for i in range(0, model_data.shape[0], 5):
    #     for j in range(0, model_data.shape[1], 5):
    #         sta_pos = [i+1, j+1]
    #         end_pos = [min(i+5, model_data.shape[0]), min(j+5, model_data.shape[1])]
    #         model_data[sta_pos[0]:end_pos[0], sta_pos[1]:end_pos[1]] = model_data[i, j]
    #         model_data[sta_pos[0]:end_pos[0], j] = model_data[i, j]
    #         model_data[i, sta_pos[1]:end_pos[1]] = model_data[i, j]
    # fig, axes = plt.subplots(1, 2)
    # cax = add_right_cax(axes[1], pad=0.02, width=0.01)
    # ims = axes[0].imshow(obs_data, cmap=a_map.cmap, vmin=0, vmax=np.nanmax(obs_data))
    # axes[1].imshow(model_data, cmap=a_map.cmap, vmin=0, vmax=np.nanmax(obs_data))
    # axes[0].set_ylabel("Y [pixel]", fontsize=14)
    # axes[0].set_xlabel("X [pixel]", fontsize=14)
    # axes[1].set_xlabel("X [pixel]", fontsize=14)
    # axes[0].set_title("Observation  2021-11-18 T00:48:15 (UT)", fontsize=14)
    # axes[1].set_title("Forward Model", fontsize=14)
    # cbar = plt.colorbar(mappable=ims, cax=cax)
    # axes[0].tick_params(labelsize=12)
    # axes[1].tick_params(labelsize=12)
    # cbar.set_label("Brightness  [pMSB]", fontsize=13)
    # plt.show()

