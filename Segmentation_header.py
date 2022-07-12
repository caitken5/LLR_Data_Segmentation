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
            break
        j3 = [j for j in peaks if j > i]
        if j2 > j3:
            j2 = i  # If the minimum value occurs after a clear peak in movement, let the first value of the new task
            # start time be the original start task time.
            # TODO: See if this if statement works as desired.
        first_min.append(np.round(j2[0][0], 2))
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
