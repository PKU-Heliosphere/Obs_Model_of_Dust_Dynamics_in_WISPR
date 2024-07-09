"""
This code includes functions needed in the counting programme.
"""

from sunpy.io import read_file
import sklearn.cluster as cluster
import sunpy.map
import matplotlib.pyplot as plt
import numpy as np
import sympy
from astropy.visualization import simple_norm
import matplotlib.colors as colors
import matplotlib.cm
import copy


def get_centroid(data_point, need_test=False):
    """
    This function refers to the Gradient Descent Algorithm.
    :param data_point: n * 2 array
    :param need_test: if you want to examine the descent speed and check if it converges.
    :return: the center point of all points input. 1 * 2 array
    """
    fun_x = sympy.symbols('x')
    fun_y = sympy.symbols('y')
    expense = 0
    vanish_point = np.zeros([2], dtype=float)

    for i_inner in range(data_point[:, 0].size - 1):
        expense = expense + sympy.sqrt((fun_x - data_point[i_inner, 0])**2 + (fun_y - data_point[i_inner, 1])**2)
        vanish_point = vanish_point + data_point[i_inner]
    vanish_point = vanish_point / data_point[:, 0].size
    descent_rate = 1
    d_x = sympy.diff(expense, fun_x)
    d_y = sympy.diff(expense, fun_y)
    for descent_count in range(300):
        vanish_point[0] = vanish_point[0] - descent_rate * d_x.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})
        vanish_point[1] = vanish_point[1] - descent_rate * d_y.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})
        if need_test is True:
            if descent_count is 0:
                expense_fun = np.array([expense.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})], dtype=float)
            else:
                expense_fun = np.insert(expense_fun, descent_count, values=np.array(
                    [expense.subs({fun_x: vanish_point[0], fun_y: vanish_point[1]})]), axis=0)
    if need_test is True:
        i_fun = np.linspace(1, 300, 300)
        p_1 = plt.figure()
        test_ax = p_1.add_subplot(111)
        test_ax.plot(i_fun, expense_fun)
        test_ax.set_xlabel('descent count')
        test_ax.set_ylabel('cost'
                           ' function')
    return vanish_point


def get_point_slope(line_number, the_path='D://Microsoft Download/Formal Files/data file/FITS/WISPR_ENC07_L3_FITS/\
20210112/psp_L3_wispr_20210112T030017_V1_1221.fits'):
    """
    NOTE: 取点顺序一定要从左到右
    :param line_number: the number of the traces gotten from the WISPR-INNER map.
    :param the_path: the path of target FITS file
    :return: obs_time: the observing time of the WISPR map. (format: YYYY-MM-DDThh:mm:ss.sss)
             true_data: the 960 * 1024 ndarray of the WISPR map (unit: MSB)
             point_set: the point couple set of all points retrieved from map(a couple contains two points in 2d frame.
        thus generating a n * 2 * 2 ndarray)
             slope_and_intercept: the slopes and intercepts of every trace(n * 2 ndarray)
             line_number: number of traces.
    """
    data, header = read_file(the_path, 'fits')[0]
    header['BUNIT'] = 'MSB'
    a_map = sunpy.map.Map(data, header)
    my_colormap = copy.deepcopy(a_map.cmap)
    true_data = copy.deepcopy(data)
    obs_time = header['DATE-BEG']
    i = 0
    while i < 1024:
        j = 0
        while j < 960:
            true_data[i, j] = data[1024-1 - i, j]
            j = j + 1
        i = i + 1
    my_fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111)
    norm_SL = colors.SymLogNorm(linthresh=0.001 * 1e-10, linscale=0.1 * 1e-10, vmin=-0.0038 * 1e-10, vmax=0.14 * 1e-10)
    plt.imshow(true_data, cmap=my_colormap, norm=norm_SL)
    ax.set_xlabel('x-axis [pixel]', fontsize=15)
    ax.set_ylabel('y-axis [pixel]', fontsize=15)
    ax.set_title('2021-01-12 03:00:17 UT', fontsize=20)
    ax.tick_params(labelsize=15)
    my_mappabel = matplotlib.cm.ScalarMappable(cmap=my_colormap, norm=norm_SL)
    plt.colorbar(label='[MSB]')

    point_set = np.zeros([line_number, 2, 2], dtype=float)
    [x_1, x_2] = plt.ginput(2, timeout=-1)
    for i in range(line_number):
        [x_1, x_2] = plt.ginput(2, timeout=-1)
        point_set[i, 0, :] = np.array(x_1, dtype=int)
        point_set[i, 1, :] = np.array(x_2, dtype=int)
        ax.plot([point_set[i, 0, 0], point_set[i, 1, 0]], [point_set[i, 0, 1], point_set[i, 1, 1]], 'r')
    plt.show()

    slope_and_intercept = np.ones([line_number, 2], dtype=float)
    for i in range(line_number):
        slope_and_intercept[i, 0] = (point_set[i, 1, 1] - point_set[i, 0, 1]) / (
                    point_set[i, 1, 0] - point_set[i, 0, 0])
        # the second index '0' means the slope
        slope_and_intercept[i, 1] = point_set[i, 1, 1] - slope_and_intercept[i, 0] * point_set[i, 1, 0]
        # the second index '1' means the intercept
    return obs_time, true_data, point_set, slope_and_intercept, line_number


def k_mean_cluster(k_and_b, line_number, n_centroids):
    """
    :param k_and_b: the slopes and intercepts of every trace(n * 2 ndarray)
    :param line_number: number of traces
    :param n_centroids: the max possible number of centroids(you should make it out by yourself)
    :return: the coordinates of all centroids(regarded as vanishing points), the labels of each line(which centroids
             it belongs to)
    """
    interceptions = []
    labels = []
    i = 0
    while i < line_number:
        j = i + 1
        while j < line_number:
            temp_interp_x = (k_and_b[j, 1] - k_and_b[i, 1]) / (k_and_b[i, 0] - k_and_b[j, 0])
            temp_interp_y = k_and_b[i, 0] * temp_interp_x + k_and_b[i, 1]
            interceptions.append([temp_interp_x, temp_interp_y])
            j = j + 1
        i = i + 1
    interceptions = np.array(interceptions, dtype=float)
    second_figure = plt.figure()
    ax = second_figure.add_subplot(111)
    for i in range(interceptions[:, 0].size):
        ax.scatter(interceptions[i, 0], interceptions[i, 1])
    centroids, line_label, _ = cluster.k_means(interceptions, n_clusters=n_centroids)
    ax.scatter(centroids[0, 0], centroids[0, 1], marker='x')
    plt.show()
    return centroids, line_label


def DBSCAN_cluster(k_and_b, line_number):
    """
       :param k_and_b: the slopes and intercepts of every trace(n * 2 ndarray)
       :param line_number: number of traces
       :return: the coordinates of all centroids(regarded as vanishing points), the labels of each line(which centroids
                it belongs to), the number of the centroids
       """
    interceptions = []
    interception_info = []  # which two lines the interception belongs to.
    line_label = np.zeros([line_number], dtype=int)
    line_flag = np.zeros([line_number], dtype=int)   # if the line has been labelled, "0" False, "1" True.
    i = 0
    while i < line_number:
        j = i + 1
        while j < line_number:
            temp_interp_x = (k_and_b[j, 1] - k_and_b[i, 1]) / (k_and_b[i, 0] - k_and_b[j, 0])
            temp_interp_y = k_and_b[i, 0] * temp_interp_x + k_and_b[i, 1]
            interceptions.append([temp_interp_x, temp_interp_y])
            interception_info.append([i, j])
            j = j + 1
        i = i + 1
    interceptions = np.array(interceptions, dtype=float)
    second_figure = plt.figure()
    ax = second_figure.add_subplot(111)
    labels = cluster.DBSCAN(eps=100, min_samples=2).fit_predict(interceptions)
    for i in range(interceptions[:, 0].size):
        if labels[i] == -1:
            ax.scatter(interceptions[i, 0], interceptions[i, 1], color='black')
        elif labels[i] == 0:
            ax.scatter(interceptions[i, 0], interceptions[i, 1], color='red')
        elif labels[i] == 1:
            ax.scatter(interceptions[i, 0], interceptions[i, 1], color='green')
        elif labels[i] == 2:
            ax.scatter(interceptions[i, 0], interceptions[i, 1], color='blue')
        elif labels[i] == 3:
            ax.scatter(interceptions[i, 0], interceptions[i, 1], color='yellow')
        else:
            ax.scatter(interceptions[i, 0], interceptions[i, 1], color='pink')
    labels = np.array(labels, dtype=int)
    num_centroids = labels.max() + 1

    for i in range(interceptions[:, 0].size):
        if labels[i] != -1:
            if line_flag[interception_info[i][0]] == 0:
                line_label[interception_info[i][0]] = labels[i]
                line_flag[interception_info[i][0]] = 1

            if line_flag[interception_info[i][1]] == 0:
                line_label[interception_info[i][1]] = labels[i]
                line_flag[interception_info[i][1]] = 1

    all_centroids = []
    for i in range(num_centroids):
        a_centroid, useless_label, _ = cluster.k_means(interceptions[labels == i], n_clusters=1)
        print(a_centroid)
        ax.scatter(a_centroid[0, 0], a_centroid[0, 1], marker='x')
        all_centroids.append(a_centroid[0])

    all_centroids = np.array(all_centroids, dtype=float)
    plt.show()
    return all_centroids, line_label, num_centroids


def get_median_value(data_array, pixel_x, pixel_y, scale=2):
    """
    :param data_array: 960*1024 ndarray
    :param pixel_x: int, coordinate of axis x corresponding to axis 1 of the data_array
    :param pixel_y: int, coordinate of axis y corresponding to axis 0 of the data_array
    :param scale: int, the scale to get the median value in the y axis( (pixel_y - scale) to (pixel_y + scale) )
    :return: the median value of the given position
    """
    i = -scale
    value_median = data_array[pixel_y, pixel_x]
    y_min = pixel_y - scale
    y_max = pixel_y + scale + 1
    value_median = np.median(data_array[y_min:y_max, pixel_x])
    return value_median


def intensity_plotting(data_array, point_set, k_and_b, line_number):
    """
    :param data_array:
    :param point_set:
    :param k_and_b:
    :param line_number:
    :return: the intensities of all traces gotten from the map.(structure: line_number * n  ndarray)
    """
    fig = plt.figure(figsize=(6, 9))
    fig.suptitle('intensity')
    all_intensity = []
    for i in range(line_number):
        ax_i = fig.add_subplot(line_number, 1, i+1)
        start_point = max(int(point_set[i, 0, 0]) - 7, 0)
        end_point = min(int(point_set[i, 1, 0]) + 7, 959)
        intensity_value = []

        for pos_x in np.linspace(start_point, end_point):
            pos_x = int(pos_x)
            pos_y = int(k_and_b[i, 0] * pos_x + k_and_b[i, 1])
            intensity_temp = get_median_value(data_array, pos_x, pos_y)
            intensity_value.append(intensity_temp)

        intensity_value = np.array(intensity_value, dtype=float)
        all_intensity.append(intensity_value)
        plot_x = (np.linspace(start_point, end_point) - start_point) / np.abs(end_point - start_point)
        plot_x = plot_x.T
        ax_i.plot(plot_x, intensity_value)
        ax_i.set_ylabel('intensity [MSB]')
        ax_i.set_yscale('log')
        ax_i.set_xticks([plot_x[0], plot_x[-1]])
        ax_i.set_xticklabels(['start', 'end'])

    plt.show()
    all_intensity = np.array(all_intensity, dtype=float)
    return all_intensity


def width_calc(data_array, point_set, line_number):
    """
    :param data_array:
    :param point_set:
    :param line_number:
    :return: the intensities of all traces gotten from the map.(structure: line_number * n  ndarray)
    """
    fig = plt.figure(figsize=(6, 9))
    fig.suptitle('intensity')
    all_intensity = []
    for i in range(line_number):
        ax_i = fig.add_subplot(line_number, 1, i+1)
        start_point_x = max(int(point_set[i, 0, 0]), 0)
        end_point_x = min(int(point_set[i, 1, 0]), len(data_array[0, :])-1)
        start_point_y = int(point_set[i, 0, 1])
        end_point_y = int(point_set[i, 1, 1])
        intensity_value = []
        if np.fabs(start_point_x-end_point_x) > np.fabs(start_point_y - end_point_y):
            for pos_x in np.linspace(start_point_x, end_point_x, endpoint=True):
                pos_x = int(pos_x)
                pos_y = int(start_point_y + (pos_x - start_point_x) / (end_point_x - start_point_x) *
                            (end_point_y - start_point_y))
                intensity_temp = data_array[pos_x, pos_y]
                intensity_value.append(intensity_temp)

            intensity_value = np.array(intensity_value, dtype=float)
            all_intensity.append(intensity_value)
            plot_x = (np.linspace(start_point_x, end_point_x) - start_point_x) / np.abs(end_point_x - start_point_x) * \
                     (np.sqrt((end_point_x - start_point_x)**2 + (end_point_y - start_point_y)**2))
            plot_x = plot_x.T
            ax_i.plot(plot_x, intensity_value)
            if i == 1:
                ax_i.set_ylabel('intensity [MSB]')
            if i == line_number - 1:
                ax_i.set_xlabel('width [pixel]')
        else:
            for pos_y in np.linspace(start_point_y, end_point_y, endpoint=True):
                pos_y = int(pos_y)
                pos_x = int(start_point_x + (pos_y - start_point_y) / (end_point_y - start_point_y) *
                            (end_point_x - start_point_x))
                intensity_temp = data_array[pos_y, pos_x]
                intensity_value.append(intensity_temp)
            intensity_value = np.array(intensity_value, dtype=float)

            all_intensity.append(intensity_value)
            plot_x = (np.linspace(start_point_y, end_point_y) - start_point_y) / np.abs(end_point_y - start_point_y) * \
                     (np.sqrt((end_point_x - start_point_x) ** 2 + (end_point_y - start_point_y) ** 2))
            plot_x = plot_x.T
            ax_i.plot(plot_x, intensity_value)
            if i == 1:
                ax_i.set_ylabel('intensity [MSB]')
            if i == line_number-1:
                ax_i.set_xlabel('width [pixel]')

    plt.show()
    all_intensity = np.array(all_intensity, dtype=float)
    return all_intensity


def write_txt(my_time, trace_number, point_set, intensities):
    filedir_path = 'D://Microsoft Download/Formal Files/data file/TXT_or_XLS/WISPR dust traces/'
    temp_file = open('D://Microsoft Download/Formal Files/data file/TXT_or_XLS/WISPR dust traces/1.txt', mode='w+')
    temp_file.write('The observing time of the WISPR map is(beginning time):\n' + my_time)
    temp_file.write('\n\n')
    temp_file.write('Number of traces is:      ' + str(trace_number) + '\n\n')
    temp_file.write('The start and end points of the traces is:\n' + '\t START \t                 END\n')
    for i in range(trace_number):
        strs = '\t ' + str(point_set[i, 0]) + ' \t ' + str(point_set[i, 1]) + '\n'
        temp_file.write(strs)
    temp_file.write('\n')
    temp_file.write('The intensity of each trace is(unit: MSB):\n')
    for i in range(trace_number):
        for j in range(intensities[i].size):
            if j == intensities[i].size - 1:
                temp_file.write(str(intensities[i, j]) + ']\n')
            elif j == 0:
                temp_file.write('[' + str(intensities[i, j]) + '        ')
            else:
                temp_file.write(str(intensities[i, j]) + '        ')
    temp_file.close()

