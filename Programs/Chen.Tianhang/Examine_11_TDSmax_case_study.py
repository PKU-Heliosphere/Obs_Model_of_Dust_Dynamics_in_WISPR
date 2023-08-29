import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import spacepy.pycdf as cdf
import sunpy
from datetime import datetime
from datetime import timedelta
import matplotlib.colors as color
import matplotlib.cm as cm

enc07_TdsMax_path = 'D:\\Microsoft Download\\Formal Files\\data file\\CDF\\Orbit07_FIELDS_TDSmax_Level2\\'
enc06_TdsMax_path = 'D:\\Microsoft Download\\Formal Files\\data file\\CDF\\Orbit06_FIELDS_TDSmax_Level2\\'


def get_tds_data(dir_path, file_num, beg_year=2021, beg_month=1, beg_day=1):
    """
    :param dir_path: The absolute path of directory containing the TDS cdf file
    :param file_num: The number of files that you want to load and read.
    :param beg_year: The beginning year of cdf files. (Or the year of the first cdf file that you want to load and read)
    :param beg_month: The beginning month of cdf files.
    :param beg_day: The beginning day of cdf files.  (The three variables should be integer.)
    :return: TDS peak voltages of Antenna 2 (unit: mV), TDS rms voltages of Antenna 2 (unit: mV),
            Epoch of voltage datas (type: numpy.datetime64). (all of them are n*1 ndarrays)
    """
    v2_tdsmax_all = np.zeros(0, dtype=np.float64)
    v2_tdsrms_all = np.zeros(0, dtype=np.float64)
    v2_epoch_all = np.zeros(0)
    for date_num in range(file_num):
        this_date = datetime(beg_year, beg_month, beg_day) + timedelta(days=date_num)
        file_path = dir_path + 'psp_fld_l2_f2_100bps_%04d%02d%02d_v02.cdf' % (this_date.year,
                                                                              this_date.month,
                                                                              this_date.day)
        data = cdf.CDF(file_path)
        v2_tdsmax_all = np.insert(v2_tdsmax_all, v2_tdsmax_all.size,
                                  np.array(data['PSP_FLD_L2_F2_100bps_TDS_Peak_V2_mV']))
        v2_tdsrms_all = np.insert(v2_tdsrms_all, v2_tdsrms_all.size,
                                  np.array(data['PSP_FLD_L2_F2_100bps_TDS_RMS_V2_mV']))
        v2_epoch_all = np.insert(v2_epoch_all, v2_epoch_all.size,
                                  np.array(data['epoch'], dtype=np.datetime64))
        data.close()
    return v2_tdsmax_all, v2_tdsrms_all, v2_epoch_all


def retieve_impact_case(v2_tdsmax_all, v2_tdsrms_all, v2_epoch_all):
    v2_tdsmax_sel = v2_tdsmax_all[np.logical_and(np.abs(v2_tdsmax_all) >= 50,
                                                 np.abs(v2_tdsmax_all/v2_tdsrms_all) >= 100)]
    v2_tdsrms_sel = v2_tdsrms_all[np.logical_and(np.abs(v2_tdsmax_all) >= 50,
                                                 np.abs(v2_tdsmax_all / v2_tdsrms_all) >= 100)]
    v2_epoch_sel = v2_epoch_all[np.logical_and(np.abs(v2_tdsmax_all) >= 50,
                                               np.abs(v2_tdsmax_all / v2_tdsrms_all) >= 100)]
    return v2_tdsmax_sel, v2_tdsrms_sel, v2_epoch_sel


def plot_case_hist(v2_tdsmax_all, v2_tdsrms_all, title_name=""):
    v2_tdsmax_all_log = np.log10(np.abs(v2_tdsmax_all))
    v2_tdsrms_all_log = np.log10(np.abs(v2_tdsrms_all))
    v2_tdsmax_sel = v2_tdsmax_all_log[np.logical_and(v2_tdsmax_all_log >= 0.5, v2_tdsrms_all_log >= 0.)]
    v2_tdsrms_sel = v2_tdsrms_all_log[np.logical_and(v2_tdsmax_all_log >= 0.5, v2_tdsrms_all_log >= 0.)]
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    fun_norm = color.LogNorm(vmin=1e0, vmax=1e3)
    fun_hist = ax.hist2d(v2_tdsrms_sel, v2_tdsmax_sel, bins=40, cmin=0, cmax=1e3, norm=fun_norm, cmap='jet',
                         vmin=1e0, vmax=1e3)
    map = cm.ScalarMappable(norm=fun_norm, cmap='jet')
    c_bar = plt.colorbar(mappable=map, ax=ax, label='Number of Cases', ticks=[1e0, 1e1, 1e2, 1e3])
    ax.set_xlim(0, 2)
    ax.set_ylim(0.5, 3)
    ax.set_xlabel(r'$\mathrm{log_{10} (RMS\,\, mV)}$')
    ax.set_ylabel(r'$\mathrm{log_{10} (Peak\,\, mV)}$')
    ax.set_title(title_name)
    plt.show()


if __name__ == '__main__':
    # data = cdf.CDF(enc07_TdsMax_path+'psp_fld_l2_f2_100bps_20210102_v02.cdf')
    dataset_enc07_peri = get_tds_data(enc07_TdsMax_path, 5, beg_year=2021, beg_month=1, beg_day=15)
    dataset_enc06_peri = get_tds_data(enc06_TdsMax_path, 5, beg_year=2020, beg_month=9, beg_day=25)
    dataset_enc07_all = get_tds_data(enc07_TdsMax_path, 30, beg_year=2021, beg_month=1, beg_day=1)
    dataset_enc06_all = get_tds_data(enc06_TdsMax_path, 30, beg_year=2020, beg_month=9, beg_day=10)
    volt_peak_peri = np.insert(dataset_enc06_peri[0], np.size(dataset_enc06_peri[0]), dataset_enc07_peri[0])
    volt_rms_peri = np.insert(dataset_enc06_peri[1], np.size(dataset_enc06_peri[1]), dataset_enc07_peri[1])
    volt_peak_all = np.insert(dataset_enc06_all[0], np.size(dataset_enc06_all[0]), dataset_enc07_all[0])
    volt_rms_all = np.insert(dataset_enc06_all[1], np.size(dataset_enc06_all[1]), dataset_enc07_all[1])
    plot_case_hist(volt_peak_all, volt_rms_all, "All cases in Encounter 06+07")
    plot_case_hist(volt_peak_peri, volt_rms_peri, "Near-Perihelion cases in Encounter 06+07")
    # print(data)
