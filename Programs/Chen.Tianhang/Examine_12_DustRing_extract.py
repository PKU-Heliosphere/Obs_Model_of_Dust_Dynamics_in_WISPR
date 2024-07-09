import numpy as np
import scipy.signal as sci_sig
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.io
from scipy.optimize import least_squares
import copy
# import pywt


def read_wispr_file(file_path):
    fun_data, fun_header = sunpy.io.read_file(file_path)[0]  # should be level 2/3 wispr file.!!
    fun_header['BUNIT'] = 'MSB'
    fun_map = sunpy.map.Map(fun_data, fun_header)
    return fun_data, fun_header, fun_map


def fit_func_1(params, fun_x):
    c0, c1, c2 = params
    fun_y = 1 / (c0 + c1 * (fun_x) ** c2)
    return fun_y


def residuals_1(params, y_obs, x_obs):
    return y_obs - fit_func_1(params, x_obs)


def fit_func_2(params, fun_x):
    b0, b1, b2 = params
    fun_y = b0 * (b1) ** fun_x + b2
    return fun_y


def residuals_2(params, y_obs, x_obs):
    return y_obs - fit_func_2(params, x_obs)


def stage1_step1(fun_data, org_data):
    # This function is to retrieve the stable, strong white light component of corona (sunlight reflected by continuous dust cloud)(step 1)
    # Note that in this step, org_data == fun_data
    row_num, column_num = fun_data.size
    fitted_data = np.zeros_like(fun_data)
    all_params = np.zeros((row_num, 3), dtype=np.float64)
    x_pixels = np.linspace(1, column_num, num=column_num)
    fun_param_0 = [1, 1, 1]
    for each_row in range(row_num):
        fun_plsq = least_squares(residuals_1, fun_param_0, bounds=((-10, -10, -10), (10, 10, 10)),
                                 args=(fun_data[each_row, :], x_pixels))
        all_params[each_row, :] = fun_plsq.x[:]
        fitted_data[each_row, :] = fit_func_1(all_params[each_row, :], x_pixels)
    fun_final_data = org_data - fitted_data
    return fun_final_data, fitted_data, all_params


def stage1_step2(fun_data, org_data, filter_coef_num=6, filter_order=4):
    """
    This function is to retrieve the stable, strong white light component of corona (sunlight reflected by continuous
    dust cloud)(step 2)
    :param fun_data:
    :param org_data:
    :param filter_coef_num:
    :param filter_order:
    :return:
    """
    row_num, column_num = fun_data.size
    corr_coef_raw = org_data / fun_data
    corr_coef_fin = np.zeros_like(corr_coef_raw)
    for each_row in range(row_num):
        corr_coef_fin[each_row, :] = sci_sig.savgol_filter(corr_coef_raw[each_row, :], filter_coef_num, filter_order)
    eliminated_data = fun_data * corr_coef_fin
    fun_final_data = org_data - eliminated_data
    return fun_final_data, eliminated_data


def stage2_step1(fun_data, org_data):
    """
    This function is to correct the result given by stage 1, eliminating (so-called) discrete K corona structure and
    stray light. (step 1)
    Note that this step is fitting the gradient but not the fun_data itself.
    :param fun_data:
    :param org_data:
    :return:
    """
    row_num, column_num = fun_data.size
    fitted_data = np.zeros((row_num, column_num), dtype=np.float64)
    fitted_data_grad = np.zeros((row_num, column_num - 1), dtype=np.float64)
    all_params = np.zeros((row_num, 3), dtype=np.float64)
    x_pixels = np.linspace(1, column_num - 1, num=column_num - 1)
    fun_data_grad = np.diff(fun_data)
    fun_param_0 = [1, 2, 0]
    for each_row in range(row_num):
        fun_plsq = least_squares(residuals_2, fun_param_0, bounds=((-10, 0, -1), (10, 10, 1)),
                                 args=(fun_data_grad[each_row, :], x_pixels))
        all_params[each_row, :] = fun_plsq.x[:]
        fitted_data_grad[each_row, :] = fit_func_2(all_params[each_row, :], x_pixels)
    return fitted_data_grad, all_params


def stage2_step2(fun_data_grad, org_data, wt_level=6, threshold_coef=1.):
    """
    This function is to correct the result given by stage 1, eliminating (so-called) discrete K corona structure
    and stray light. (step 2)
    :param fun_data_grad:
    :param org_data:
    :param wt_level:
    :param threshold_coef:
    :return:
    """
    row_num, column_num = fun_data_grad.size
    eliminated_data = np.zeros((row_num, column_num + 1), dtype=np.float64)
    fun_final_data = np.zeros((row_num, column_num + 1), dtype=np.float64)
    org_data_grad = np.diff(org_data)
    corr_coef_raw = org_data_grad / fun_data_grad
    corr_coef_fin = np.zeros_like(corr_coef_raw)
    for each_row in range(row_num):
        wt_coeffs = pywt.wavedec(corr_coef_raw[each_row, :], 'db4', level=wt_level)
        wt_threshold = np.median(np.abs(wt_coeffs[-1])) / threshold_coef
        wt_coeffs = [pywt.threshold(wt_coeff, wt_threshold, mode='soft') for wt_coeff in wt_coeffs]
        corr_coef_fin[each_row, :] = pywt.waverec(wt_coeffs, 'db4')

    eliminated_data_grad = fun_data_grad * corr_coef_fin
    eliminated_data[:, 1:(column_num + 1)] = np.cumsum(eliminated_data_grad, axis=1)
    for each_row in range(row_num):
        eliminated_data[each_row, :] = eliminated_data[each_row, :] + org_data[each_row, 0]
    fun_final_data = org_data - eliminated_data
    return fun_final_data, eliminated_data


def stage3(fun_data, org_data, wt_level=6, threshold_coef=1.):
    row_num, column_num = fun_data.size
    eliminated_data = np.zeros((row_num, column_num), dtype=np.float64)
    fun_final_data = np.zeros_like(eliminated_data)
    corr_coef_raw = org_data / fun_data
    corr_coef_fin = np.zeros_like(corr_coef_raw)
    for each_row in range(row_num):
        wt_coeffs = pywt.wavedec(corr_coef_raw[each_row, :], 'db4', level=wt_level)
        wt_threshold = np.median(np.abs(wt_coeffs[-1])) / threshold_coef
        wt_coeffs = [pywt.threshold(wt_coeff, wt_threshold, mode='soft') for wt_coeff in wt_coeffs]
        corr_coef_fin[each_row, :] = pywt.waverec(wt_coeffs, 'db4')
    eliminated_data = fun_data * corr_coef_fin
    fun_final_data = org_data - eliminated_data
    return fun_final_data, eliminated_data


if __name__ == '__main__':



