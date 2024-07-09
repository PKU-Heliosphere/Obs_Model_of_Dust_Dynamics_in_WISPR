"""
This script is for counting all the scattering intensity from WISPR-INNER and outputting the histogram.
"""
import numpy as np
import trace_counting_func as tcf
import trace_counting_histogram as tch

trace_num = 5
my_time, my_data, point_set, k_b, n = tcf.get_point_slope(trace_num, the_path='D://Microsoft Download/Formal Files\
/data file/FITS/WISPR-I_ENC07_L3_FITS/20210111/psp_L3_wispr_20210111T170017_V1_1221.fits')
my_intensities = tcf.width_calc(my_data, point_set, n)
my_data = np.round(my_data, 18)
# tch.draw_histogram(my_time, my_data, point_set, k_b, n)
my_centroids, my_label, numbers = tcf.DBSCAN_cluster(k_b, trace_num)
print(my_centroids)
# point_set = np.array(point_set, dtype=int)
# my_intensities = np.round(my_intensities, 16)
# trfun.write_txt(my_time, trace_num, point_set, my_intensities)
