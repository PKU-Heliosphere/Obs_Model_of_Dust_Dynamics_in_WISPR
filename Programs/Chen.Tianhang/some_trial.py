"""
This is just a script for some trials, usually numpy/sunpy/spicy, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sunpy.io.fits as fits
from sunpy.net import attrs as a, Fido
import PyMieScatt as Mie
import furnsh_all_kernels
import spiceypy as spice
import spacepy.pycdf as cdftool

# A = np.array([[1, 2, 3], [5, 6, 9]], dtype=float)
# B = np.array([[1], [2.5], [33.1]], dtype=float)
# C = np.dot(A, B)
# print(C)
my_pdf = cdftool.CDF('D://Microsoft Download/Formal Files/data file/CDF/Orbit02_PSP_TDS/'
                     'psp_fld_l2_f2_100bps_20190317_v02.cdf')
print(my_pdf)
phi = np.linspace(0, 180, num=1200, endpoint=False) * np.pi / 180
theta = np.linspace(-60, 60, num=30, endpoint=True) * np.pi / 180
[phi_mesh, theta_mesh] = np.meshgrid(phi, theta)
print(len(phi_mesh[0, :]))
# generate 2 2d grids for the x & y bounds
# y, x = np.mgrid[-3:3+dy:dy, -3:3+dx:dx]
# z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
# # x and y are bounds, so z should be the value *inside* those bounds.
# # Therefore, remove the last value from the z array.
# z = z[:-1, :-1]
# z_min, z_max = -abs(z).max(), abs(z).max()
#
# fig, axs = plt.subplots(2, 2)
#
# ax = axs[0, 0]
# c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# ax.set_title('pcolor')
# fig.colorbar(c, ax=ax)
#
# ax = axs[0, 1]
# c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# ax.set_title('pcolormesh')
# fig.colorbar(c, ax=ax)
#
# ax = axs[1, 0]
# c = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
#               extent=[x.min(), x.max(), y.min(), y.max()],
#               interpolation='nearest', origin='lower', aspect='auto')
# ax.set_title('image (nearest, aspect="auto")')
# fig.colorbar(c, ax=ax)
#
# ax = axs[1, 1]
# c = ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# ax.set_title('pcolorfast')
# fig.colorbar(c, ax=ax)
#
# fig.tight_layout()
# plt.show()

# my_time = a.Time('2021-01-12', '2021-01-13')
# my_instrument = a.Instrument('wispr')
# my_detector = a.Detector('outer')
# results = Fido.search(my_time, my_detector)
# results_1 = results[0]
# needed_results = results_1[results_1['fileid'] == 'data/psp/wispr/L3/orbit07/outer/20210112/psp_L3_wispr_'
#                            '20210112T234129_V1_2302.fits']
# downloaded_file = Fido.fetch(needed_results, path='D://Desktop')
#
# print(header['NSUMBAD'])
# my_path_1 = 'D://Desktop/psp_l3_wispr_20210112t234129_v1_2302.fits'
# my_path_2 = 'D://Desktop/psp_l3_wispr_20210112t000017_v1_1221.fits'
# data, header_1 = fits.read(my_path_1)[0]
# data, header_2 = fits.read(my_path_2)[0]
# print(header_1['XPOSURE'], header_2['XPOSURE'])
# print(header_1)
# print(header_2)

# ps.MieS1S2()



