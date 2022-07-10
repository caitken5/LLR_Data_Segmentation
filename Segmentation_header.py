# Here lies the constants and functions to be used in main.py of LLR_Data_Segmentation.
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
