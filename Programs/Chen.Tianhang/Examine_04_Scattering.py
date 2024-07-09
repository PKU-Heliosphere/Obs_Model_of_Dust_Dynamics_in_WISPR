"""
This script is for calculating the radius(r)/distance(d)/velocity(v) of dust particles relative to PSP.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sunpy.map
import spiceypy as spice
import furnsh_all_kernels
import sympy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import sunpy.io.fits
from astropy.visualization import simple_norm
import matplotlib.cm
from scattering_intensity import scattering_intensity_5 as intensity_5
from scattering_intensity import scattering_intensity_6 as intensity_6
import trace_counting_func as tcf
import copy
from plotly import graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots

my_path = 'D://Microsoft Download/Formal Files/data file/FITS/WISPR-I_ENC07_L3_FITS/20210112/psp_L3_wispr_' \
          '20210112T213015_V1_1221.fits'
data, header = sunpy.io.fits.read(my_path)[0]
header['BUNIT'] = 'MSB'
line_number = 1
my_time, my_data, point_set, k_b, n = tcf.get_point_slope(1, the_path=my_path)
length_of_streak = max(np.fabs(point_set[0, 0, 0]-point_set[0, 1, 0]), np.fabs(point_set[0, 0, 1]-point_set[0, 1, 1]))
my_intensities = tcf.intensity_plotting(my_data, point_set, k_b, n)
radius = np.linspace(1e-7, 1e-4, 1000) * 1e6  # unit: micron
distance = np.linspace(1, 1e3, 1000)  # unit: m
velocity = np.linspace(1, 1e6, 1000)  # unit: m/s
x, y = np.meshgrid(distance, radius)
# x_p, y_p = np.meshgrid(distance, velocity)
brightness = intensity_5(5, y/1e6, x, header['XPOSURE']/header['NSUMEXP'], length_of_streak, '20210112T21:30:15')
# brightness_p = intensity_6(5, 8, x_p, y_p, '20210112T21:30:15')
figure_2 = plt.figure(figsize=(9, 9))
ax_1 = figure_2.add_subplot(111)
pcm = ax_1.contourf(x, y, brightness, [my_intensities[0].min(), my_intensities[0].max()],
                    norm=colors.LogNorm(), cmap='rainbow', alpha=0.8)
# pcm = ax_1.contourf(x_p, y_p, brightness_p, [my_intensities[0].min(), my_intensities[0].max()],
#                     norm=colors.LogNorm(), cmap='rainbow', alpha=0.8)
# pcm = ax_1.contourf(x_p, y_p, brightness_p, [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
#                     norm=colors.LogNorm(), cmap='rainbow', alpha=0.8)
# cb = figure_2.colorbar(pcm, ax=ax_1)
# ax_1.set_title('Observing Brightness')
# cb.set_label('Observing Brightness [MSB]')
ax_1.set_ylim(1e-1, 1e2)
ax_1.set_xlim(1, 1e3)
ax_1.set_yscale('log')
ax_1.set_xscale('log')
ax_1.set_xlabel('distance from particle to PSP  [$m$]', fontsize=15)
ax_1.set_ylabel(r'radius of particle  [$\mu m$]', fontsize=15)
ax_1.tick_params(labelsize=15)
wished_velocity = length_of_streak * 20e-6 * distance / 28e-3 / (header['XPOSURE']/header['NSUMEXP'])  # unit: m/s
# ax_1.plot(distance, wished_velocity, color='black', lw=2, label='possible v-d relation')
streak_width = 5.66 * 20e-6
streak_width_max = 7.66 * 20e-6
wished_radius = (streak_width/2 - 28e-3/distance * np.sqrt(42e-6)) / (1 - 28e-3/distance) * (distance/28e-3 - 1)
wished_radius_max = (streak_width_max/2 - 28e-3/distance * np.sqrt(42e-6)) / (1 - 28e-3/distance) * (distance/28e-3 - 1)
ax_1.plot(distance, wished_radius*1e6, color='black', lw=2, label='possible r-d relation')
ax_1.plot(distance, wished_radius_max*1e6, color='black', lw=2)
plt.legend()
# ax_1 = figure_2.add_subplot(111, projection='3d')
# ax_1.set_zscale('log')
# ax_1.set_yscale('log')
plt.show()



