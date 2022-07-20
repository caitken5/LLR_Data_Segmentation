# Here lies the constants and functions to be used in main.py of LLR_Data_Segmentation.
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
matplotlib.use('Agg')
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


def find_first_min(t, f, t_targets, peaks):
    # Using the location of the first time stamp of each target, and the force, calculate the first minimum force
    # achieved in highly filtered force.
    first_min = []
    my_min = (np.asarray(signal.argrelmin(f)).flatten()).reshape((-1, 1))
    # Based on Algorithm 1, find the first instance of min_ind that is greater than each subsequent t_target.
    for i in t_targets:
        j2 = [j for j in my_min if j > i]
        j3 = [j for j in peaks if j > i]  # What if j3 is equal to 0?
        if len(j2) == 0:  # Deal with case where no more minimums occur.
            j2 = i[0]
        else:
            j2 = j2[0][0]
        if len(j3) == 0:  # Deal with case where no more maximums occur.
            j3 = t.shape[0] - 1  # This value should be chosen such that only the really prominent peaks are used.
        else:
            j3 = j3[0]
        if j2 > j3:
            j2 = i[0]  # If the minimum value occurs after a clear peak in movement, let the first value of the new task
            # start time be the original start task time.
        first_min.append(np.round(j2, 2))
        # NOTE: THERE WILL BE AN ERROR BETWEEN j2 = i and j2 = a value in min_time.
    return first_min, my_min


def plot_force_and_start_of_task_and_lowest_force_lines(t, t_targets, f, d, peaks, first_min, save_name, save_folder):
    # This function plots the recorded force for an entire task vs time, as well as straight lines for the both the
    # first time stamp of each target, and the time stamp of each line.
    # Plot what this looks like.
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 8)
    ax.axvline(t_targets[0], color='k', label="Start of Task")
    [ax.axvline(_t_targets, color='k') for _t_targets in t_targets]
    # ax.plot(t, f, 'g-', label="Force Magnitude, no filter")
    ax.plot(t, d/10, 'm-', label="Distance from Target [cm]")
    ax.plot(t, f, 'b-', label="Force Magnitude, 3 Hz filter")
    ax.axvline(first_min[0], color='g', label="First Minimum in Force Application After New Target")
    [ax.axvline(_my_min, color='g') for _my_min in first_min]
    ax.axvline(peaks[0], color='r', label='Peak in Applied Force - Ballistic Movement')
    [ax.axvline(_my_peaks, color='r') for _my_peaks in peaks]
    plt.legend()
    plt.title('Force Applied During Entire Task Set for ' + save_name + '\n Start of Task (black) and Lowest Force Applied (green)')
    # TODO: Add scale for second y-axis in cm.
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    # plt.show()  # TODO: REMOVE THIS WHEN DONE TESTING CALCULATION OF LOCATION OF NEW REACH.
    plt.savefig(save_folder)
    fig.clf()
    plt.close()
    gc.collect()


# TODO: Test split_by_indices.
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


# TODO: Test find_min_and_max_peak.
def find_min_and_max_peak(data, minima_prominence, maxima_prominence):
    # Find the minimum values and peaks in selected data.
    # To find the minimum values using find_peaks function, first multiply series by -1.
    neg_data = data*(-1)
    minima = signal.find_peaks(neg_data, prominence=minima_prominence)
    maxima = signal.find_peaks(data, prominence=maxima_prominence)
    return minima, maxima


# TODO: Test find_end_of_initial_reach.
def find_end_of_initial_reach(data):
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
