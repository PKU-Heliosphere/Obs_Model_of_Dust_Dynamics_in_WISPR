"""
This script is to read the newly released data file (level 3) of PSP/FIELDS, which contains the information of
hupervelocity dust impact events and rate.
"""

import numpy as np
import matplotlib.pyplot as plt
import spacepy.pycdf as cdf
from datetime import datetime
import os


dir_path = 'D:\\Microsoft Download\\Formal Files\\data file\\CDF\\Orbit06_FIELDS_Dust_Level3\\'
all_files = os.listdir(dir_path)
perihelion_07 = datetime.strptime('2021-01-17/17:40', '%Y-%m-%d/%H:%M')
perihelion_06 = datetime.strptime('2020-09-27/09:16', '%Y-%m-%d/%H:%M')
epoch_all = []
rate_raw = []
rate_wav = []
rate_ucc = []

# read data
for file in all_files:
    data = cdf.CDF(dir_path+file)
    for i in range(3):
        # the time interval of sample window is 8 hr, so the size of rates in every cdf file is 3.
        # this_datetime = datetime.strptime(data['psp_fld_l3_dust_V2_rate_epoch'][i], '%Y-%m-%d %H:%M:%S')
        this_datetime = data['psp_fld_l3_dust_V2_rate_epoch'][i]
        epoch_all.append((this_datetime - perihelion_06).total_seconds()/3600/24)
        # The meaning of x coordinate for plotting is 'Days from perihelion for encounter 7'
        rate_raw.append(data['psp_fld_l3_dust_V2_rate_raw'][i])
        rate_wav.append(data['psp_fld_l3_dust_V2_rate_wav'][i])
        rate_ucc.append(data['psp_fld_l3_dust_V2_rate_ucc'][i])
        # Unit of rate: counts / hour
epoch_all = np.array(epoch_all, dtype=np.float64)
rate_raw = np.array(rate_raw, dtype=np.float64) * 3600
rate_wav = np.array(rate_wav, dtype=np.float64) * 3600
rate_ucc = np.array(rate_ucc, dtype=np.float64) * 3600
# plot
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.plot(epoch_all, rate_raw, color='tab:blue', label='raw rate')
# ax.plot(epoch_all, rate_wav, color='tab:green', label='rate corrected for inhibition by plasma waves')
ax.plot(epoch_all, rate_ucc, '-o', color='tab:red', label='finally corrected')
ax.set_xlabel('Days from perihelion 2020-09-27 T09:16')
ax.set_ylabel(r'Impact rate $\mu$ ($hr^{-1}$)')
# ax.set_yscale('log')
ax.set_title('Encounter 06')
plt.legend()
plt.show()
# file_name = 'psp_fld_l3_dust_20210111_v01.cdf'
# a = cdf.CDF(dir_path+file_name)
# print(a['psp_fld_l3_dust_V2_rate_epoch'][1])
