"""
This is a script for streaks examination what it is, impact-generated debris or primary interplanetary dust particles.
The 02 code is based on FIELDS data analysis.
This examination is a kind of statistics analysis of the interruptions some minutes before the WISPR steak storms.
"""
import spacepy.pycdf as cdf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import spiceypy as spice

spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/naif0012.tls')
my_cdf = cdf.CDF('D://Microsoft Download/Formal Files/data file/CDF/PSP_FIELDS_DFB_AC_BPF/'
                    'psp_fld_l2_dfb_ac_bpf_dV34hg_20190330_v01.cdf')
information = my_cdf.cdf_info()
print(information)
a = my_cdf.varget('psp_fld_l2_dfb_ac_bpf_dV34hg_peak')
the_time = my_cdf.varget('epoch')
print(a, '\n', the_time)



