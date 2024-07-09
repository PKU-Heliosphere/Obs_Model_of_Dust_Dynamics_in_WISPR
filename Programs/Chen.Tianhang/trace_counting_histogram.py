
import matplotlib.pyplot as plt
import trace_counting_func as trfun


def draw_histogram(start_time, data_array, point_set, slope_interception, number):
    my_time, my_data, point_set, k_b, n = start_time, data_array, point_set, slope_interception, number
    my_intensities = trfun.intensity_plotting(my_data, point_set, k_b, n)
    counting_intensity = my_intensities.flatten()
    my_fig = plt.figure(figsize=(7, 9))
    ax = my_fig.add_subplot(111)
    my_histogram = ax.hist(counting_intensity, bins=10, edgecolor='black', facecolor='greenyellow')
    ax.set_xlabel('intensity [MSB]')
    ax.set_ylabel('counts [1]')
    ax.set_title('The Distribution of the dust traces')
    plt.show()

