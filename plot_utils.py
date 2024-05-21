import numpy as np

def movingaverage(arr, window_size):

    moving_averages = []

    # Store cumulative sums of array in cum_sum array
    cum_sum = np.cumsum(arr)
    i = 1

    # Loop through the array elements
    while i <= len(arr):

        if i <= window_size:
            window_average = cum_sum[i-1] / i
        else:
            window_average = (cum_sum[i-1]-cum_sum[i-window_size-1]) / window_size

        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1


    return moving_averages