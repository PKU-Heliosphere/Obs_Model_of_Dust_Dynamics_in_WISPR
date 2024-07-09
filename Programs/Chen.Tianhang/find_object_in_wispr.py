import numpy as np
import furnsh_kernels
from Asteroid_positions import get_aster_pos,get_body_pos
import spiceypy as spice
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
AU = 1.49e8  # km
start_time = '2021-01-14'
stop_time = '2021-01-16'
obs_time = '20210115T1230'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
obs_dt = datetime.strptime(obs_time,'%Y%m%dT%H%M')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])
step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

obs_et = spice.datetime2et(obs_dt)
obs_one = spice.datetime2et(datetime.strptime('20210115T1200','%Y%m%dT%H%M'))
obs_two = spice.datetime2et(datetime.strptime('20210115T1300','%Y%m%dT%H%M'))
obs_times=[x*(obs_two-obs_one)/step + obs_one for x in range(step)]
fullname_str = '163693 Atira (2003 CP20)'
df = pd.read_csv('data/sbdb_query_results.csv')
aster_df = df[df['full_name']==fullname_str]
spkid = df.loc[:,'spkid']
spkid=[3712675, 3791243]
for id in spkid:
    print(id)
    aster_positions = get_aster_pos(id, start_time, stop_time, observer='SUN', frame='HCI')
    isvisibe = spice.fovtrg('SPP_WISPR_OUTER',str(id),'POINT','','None','SPP',obs_et)
    if isvisibe:
        print(str(id)+' VISIBLE in WISPR_OUTER at '+obs_time)
        psp_positions, psp_LightTimes = spice.spkpos('SPP', obs_times, 'SPP_HCI', 'NONE', 'SUN')
        sun_positions, sun_LightTimes = spice.spkpos('SUN', obs_times, 'SPP_HCI', 'NONE', 'SUN')
        ast_positions, ast_LightTimes = spice.spkpos(str(id),obs_times,'SPP_HCI','NONE','SUN')
        psp_positions = psp_positions.T  # psp_positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
        psp_positions = psp_positions / AU
        sun_positions = sun_positions.T
        sun_positions = sun_positions / AU
        ast_positions = ast_positions.T
        ast_positions = ast_positions / AU

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(psp_positions[0], psp_positions[1], psp_positions[2],c='k')
        ax.scatter(ast_positions[0], ast_positions[1], ast_positions[2],c=times,cmap='jet')
        ax.scatter(sun_positions[0],sun_positions[1],sun_positions[2],c='r')
        # ax.scatter(1, 0, 0, c='red')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        plt.title('PSP'+ '(' + start_time + '_' + stop_time + ')')
        plt.show()
        lat_lst=[]
        for i in range(step):
            outerpos_tmp, _ = spice.spkcpt(ast_positions[:,i] * AU, 'SUN', 'SPP_HCI', obs_times[i], 'SPP_WISPR_OUTER',
                                              'OBSERVER', 'NONE', 'SPP')
            outerpos_lat = spice.reclat(outerpos_tmp[[2, 0, 1]])
            lat_lst.append([np.rad2deg(outerpos_lat[1]),np.rad2deg(outerpos_lat[2])])
        lat_lst = np.array(lat_lst)
        plt.scatter(lat_lst[:,0],lat_lst[:,1],c=obs_times,cmap='jet')
        plt.xlim([-29,29])
        plt.ylim([-29,29])
        plt.show()
'''found! 3712675, 3791243'''