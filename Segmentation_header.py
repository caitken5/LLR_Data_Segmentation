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
    min_time = t[my_min]
    # Based on Algorithm 1, find the first instance of min_ind that is greater than each subsequent t_target.
    for i in t_targets:
        j2 = [j for j in min_time if j > i]
        if len(j2) == 0:
            j2 = i
        j3 = [j for j in peaks if j > i]
        if j2 > j3:
            j2 = i  # If the minimum value occurs after a clear peak in movement, let the first value of the new task
            # start time be the original start task time.
            # TODO: See if this if statement works as desired.
            # TODO: Might also just exclude the data where this occurs.
        first_min.append(np.round(j2[0][0], 2))
        # NOTE: THERE WILL BE AN ERROR BETWEEN j2 = i and j2 = a value in min_time.
        # TODO: Ensure first_min provides the correct time and is not off by 0.01.
    return first_min, my_min, min_time


def plot_force_and_start_of_task_and_lowest_force_lines(t, t_targets, f, d, peaks, first_min, save_name, save_folder):
    # This function plots the recorded force for an entire task vs time, as well as straight lines for the both the
    # first time stamp of each target, and the time stamp of each line.
    # Plot what this looks like.
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 8)
    [ax.axvline(_t_targets, color='k') for _t_targets in t_targets]
    # ax.plot(t, f, 'g-', label="Force Magnitude, no filter")
    ax.plot(t, d/10, 'm-', label="Distance from Target [cm]")
    ax.plot(t, f, 'b-', label="Force Magnitude, 3 Hz filter")
    [ax.axvline(_my_min, color='g') for _my_min in first_min]
    [ax.axvline(_my_peaks, color='r') for _my_peaks in peaks]
    plt.legend()
    plt.title('Force Applied During Entire Task Set for ' + save_name + '\n Start of Task (black) and Lowest Force Applied (green)')
    # TODO: Add scale for second y-axis in cm.
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
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
    d = data[:, data_header.index('Dist_From_Target')]
    min_prominence = 5  # Likely in mm.
    max_prominence = 5
    new_reach_index = []
    # Then pass this data to the find_min_peak function. Hopefully some smoothing is not needed.
    minima, maxima = find_min_and_max_peak(d, min_prominence, max_prominence)
    # For every minima in the data that occurs, then select the first maxima that occurs after that minima value.
    for i in minima:
        j2 = [j for j in maxima if j > i]
        if j2.shape[0] > 0:
            # Then another return to the target will have occurred, and the reach can be considered to have occurred.
            new_reach_index.append(j2[0][0])
    return new_reach_index, minima, maxima
# Note that it is possible for new_reach_index to be empty.
