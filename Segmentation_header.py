# Here lies the constants and functions to be used in main.py of LLR_Data_Segmentation.
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# matplotlib.use('Agg')
import gc

data_header = ['Time', 'Des_X_Pos', 'Des_Y_Pos', 'X_Pos', 'Y_Pos', 'OptoForce_X', 'OptoForce_Y', 'OptoForce_Z',
               'OptoForce_Z_Torque', 'Theta_1', 'Theta_2', 'Fxy_Mag', 'Fxy_Angle', 'CorrForce_X', 'Corr_Force_Y',
               'Target_Num', 'X_Vel', 'Y_Vel', 'Vxy_Mag', 'Vxy_Angle', 'Dist_From_Target', 'Disp_Btw_Pts', 'Est_Vel',
               'To_From_Home', 'Num_Prev_Targets', 'Resistance']


def get_first_target_row(data):
    # Return the value of each of the first sets of data.
    to_from_home = data[:, data_header.index('To_From_Home')]
    diff = np.diff(to_from_home)
    new_target = (np.asarray(np.where(diff != 0)).squeeze()).reshape((-1, 1))
    new_target += 1
    return new_target


def butterworth_filter(data, freq):
    # 2nd order 20 Hz Butterworth filter applied along the time series dimension of as many columns are passed to it.
    # Will likely just be used to filter the velocity data, since a second order 20Hz Butterworth filter has already been applied to the force data.
    sos = signal.butter(2, freq, 'lowpass', fs=100, output='sos')  # Filter cutoff of
    filtered_data = signal.sosfiltfilt(sos, data, axis=0)
    return filtered_data


def find_first_min(t, f, t_targets, my_percent):
    # Using the location of the first time stamp of each target, and the force, calculate the first minimum force
    # achieved in highly filtered force.
    first_min = []
    my_min_series = (np.asarray(signal.argrelmin(f)).flatten()).reshape((-1, 1))
    my_min_series = np.squeeze(my_min_series)
    # Based on Algorithm 1, find the first instance of min_ind that is greater than each subsequent t_target.
    for k, i in enumerate(t_targets):
        # Calculate the top of the force range in this target segment.
        my_start = i[0]
        if k == t_targets.shape[0]-1:
            my_end = t.shape[0]-1
        else:
            my_end = t_targets[k+1]
            my_end = my_end[0]
        segment = f[my_start:my_end + 1]
        my_max = np.max(segment)
        my_min = np.min(segment)
        my_rng = my_max-my_min
        my_thresh = my_max-my_rng*my_percent
        # Retrieve the set of calculated minimum values and peaks that occur after i.
        j2 = [j for j in my_min_series if j > my_start if j < my_end]
        # Only keep j2 and j3 values that are less than my_end, as done above in list comprehension.
        # Then, only keep index values for j2 where the value is less than the selected threshold.
        j2 = [j for j in j2 if f[j] < my_thresh]
        if len(j2) == 0:  # Deal with case where no more minimums occur.
            j2 = my_start
        else:
            j2 = j2[0]
        first_min.append(np.round(j2, 2))
        # NOTE: THERE WILL BE AN ERROR BETWEEN j2 = i and j2 = a value in min_time.
    return first_min, my_min


def find_first_min_2(segment, my_percent_low, my_start):
    # Since this is per segment now, I can just take the segmented data, look for minimum values, identify when and
    # where they happen in sequence and if they meet the magnitude requirements of qualifying as the first min.
    # Return a single index value.
    # Step 1: In the segment, identify the minimums and maximums.
    my_min_series = (np.asarray(signal.argrelmin(segment)).flatten()).reshape((-1,))
    if my_min_series.shape[0] > 1:  # Do not squeeze the dimensions if there is only a single (or no) returned value.
        my_min_series = np.squeeze(my_min_series)  # my_min_series is the index of the location of minimums.
    # Step 2: Calculate the minimum and maximum values in the series. Then calculated the range, and the threshold based
    # on the provided threshold for which values will constitute a minimum.
    my_max = np.max(segment)
    my_min = np.min(segment)
    my_rng = my_max - my_min
    my_thr = my_min + (my_rng * my_percent_low)
    # Since I've already separated out the segment of the data, no need to use list comprehension to extract out the
    # mins from that segment only. But, use list comprehension to extract the minimums below my_thr.
    valid_mins = [j for j in my_min_series if segment[j] < my_thr]
    if len(valid_mins) == 0:
        first_min = my_start
    else:
        first_min = my_start + valid_mins[0]
    valid_mins += my_start  # For this and first_min, addressing the actual index relative to the entire signal,
    my_min_series += my_start
    return first_min, valid_mins, my_min_series


def plot_force_and_start_of_task_and_lowest_force_lines(t, t_targets, f, v, d, firstmin_pts, fpeaks_pts, fmins_pts, save_stuff):
    # This function plots the recorded force for an entire task vs time, as well as straight lines for the both the
    # first time stamp of each target, and the time stamp of each line.
    # Plot what this looks like.
    ms = 10
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 8)
    ax2 = ax.twinx()
    ax.axvline(t_targets[0], color='k', label="Start of Task")
    [ax.axvline(_t_targets, color='k') for _t_targets in t_targets]
    ax.plot(t, d/10, 'm-', label="Distance from Target [cm] (left y-axis)")
    ax.plot(t, f, 'b-', label="Force Magnitude, 3 Hz filter (left y-axis)")
    ax2.plot(t, v, 'g-', label='Velocity, 3 Hz filter (right y-axis)')
    ax.plot(firstmin_pts[0], firstmin_pts[1], 'c.', label="First Minimum in Force Application After New Target",
            markersize=ms)
    ax.plot(fpeaks_pts[0], fpeaks_pts[1], 'r+', label="Peak in Applied Force - Ballistic Movement", markersize=ms)
    ax.plot(fmins_pts[0], fmins_pts[1], 'c|', label="End of Ballistic Movement", markersize=ms)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    ax.grid()
    ax.set_title('Force Applied During Entire Task Set for ' + save_stuff[0])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax2.set_ylabel('Velocity [mm/s]')
    fig.savefig(save_stuff[1])
    fig.clf()
    plt.close()
    gc.collect()


def split_by_indices(data, indices):
    # Split the arrays according to the indices that correspond to when a new target is being responded to.
    start = 0
    ragged_list = []
    # Note that first_min is already the array of all the indices for packing and unpacking the arrays.
    for i in indices:
        ragged_list.append(data[start:i, :])
        start = i
    # We will not append the last section of the array because it likely contains erroneous data.
    ragged_list.append(data[start:, :])
    return ragged_list


def find_min_and_max_peak(data, minima_prominence, maxima_prominence):
    # Find the minimum values and peaks in selected data.
    # To find the minimum values using find_peaks function, first multiply series by -1.
    neg_data = data*(-1)
    minima = signal.find_peaks(neg_data, prominence=minima_prominence)
    maxima = signal.find_peaks(data, prominence=maxima_prominence)
    # Conclusion: using find_peaks with prominence in order to identify minima and maxima in the signal is too unreliable.
    return minima, maxima


def find_first_max(segment, my_percent_high, my_start, my_end):
    # Extract all of the maximum values.
    my_max_series = (np.asarray(signal.argrelmax(segment)).flatten()).reshape((-1,))
    # Identify the threshold.
    my_max = np.max(segment)
    my_min = np.min(segment)
    my_rng = my_max - my_min
    my_thr = my_max - my_rng * my_percent_high
    valid_maxs = [j for j in my_max_series if segment[j] > my_thr]
    if len(valid_maxs) == 0:
        first_max = my_end
    else:
        first_max = my_start + valid_maxs[0]
    valid_maxs += my_start  # For this and first_min, addressing the actual index relative to the entire signal,
    my_max_series += my_start
    return first_max, valid_maxs, my_max_series


def find_end_of_initial_reach_old(data):
    # First identify the displacement column of the data.
    t = data[:, data_header.index('Time')]
    d = data[:, data_header.index('Dist_From_Target')]
    min_prominence = (10, None)  # Likely in mm.
    max_prominence = (None, 25)
    new_reach_index = []
    # Then pass this data to the find_min_peak function. Hopefully some smoothing is not needed.
    minima, maxima = find_min_and_max_peak(d, min_prominence, max_prominence)
    # Extract the indices separately for each minima and maxima, as well as the prominence value. Then, idnetify the time at which each occurs.
    minima_indices = minima[0]
    maxima_indices = maxima[0]
    # For every minima in the data that occurs, then select the first maxima that occurs after that minima value.
    for i in minima_indices:
        j2 = [j for j in maxima_indices if j > i]
        if len(j2) > 0:
            # Then another return to the target will have occurred, and the reach can be considered to have occurred.
            new_reach_index.append(j2[0])

    return new_reach_index, minima, maxima
# Note that it is possible for new_reach_index to be empty.


def save_npz(my_array, my_list, save_folder, file_name):
    # Save the array with the list of indices to make packing and unpacking the array into a ragged array fairly simple.
    # convert the generated list to a numpy array.
    ending = '.npz'
    my_list = np.asarray(my_list)
    my_array = np.asarray(my_array, dtype=object)
    # check if the file is control or patient.
    # Add the data as a worksheet to the workbook.
    file_name_list = file_name.split('_')
    # Remove the .npy ending from file_name.
    new_file_name = '_'.join(file_name_list[:6])
    save_name = save_folder + '/' + new_file_name + '_' + ending
    np.savez(save_name, target_data=my_array, unpacking_indices=my_list)


def find_end_of_initial_reach(data, start, my_max, max_index, my_end, percent):
    # Already have the location of the desired peak value. Get the peak_index relative to the segment start and end.
    segment = data[start:my_end+1]
    max_index = max_index - start
    # Calculate the minimum value of force applied during the segment.
    my_min = np.min(segment)
    # Calculate the range of the force applied in the segment.
    my_range = my_max - my_min
    # Get the force value that represents the chosen percentage band from the lowest applied force.
    min_limit = my_min + my_range * percent
    # Get the data between the index of the segment where the selected peak force occurs and the end of the segment.
    neg_segment = segment[max_index:]*-1
    # Retrieve the index of the minimum values after the peak occurrence.
    mins_index = signal.find_peaks(neg_segment)[0]
    mins_index += max_index
    new_min = segment[-1]
    new_min_index = my_end
    for i in mins_index:
        temp = segment[i]
        if temp < min_limit:
            new_min = temp
            new_min_index = start + i
            break
    return new_min, new_min_index
