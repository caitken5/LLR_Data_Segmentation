# This code will be used to segment the data according to the task number, and at the low point of the exerted force,
# which will be treated as the indicator that a new segment of data has occurred.
# Files are saved into two different folders based on their file type; individual numpy files for each task will be
# saved to a NUMPY_FILES folder. Files that are of the whole task (with first and last targets removed) will be saved to
# a NPZ_FILES folder.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


import Segmentation_header as h


# TODO: Segment data using algorithm that takes the minimum of the applied force after a new target has been reached up
#  until the next like segment for the next task.

# Set up the file paths.
source_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/3_LLR_DATA_CLEANING/" \
                "NUMPY_FILES"
test_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/TEST_FOLDER"
npy_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "NUMPY_FILES"
npz_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "NPZ_FILES"
graph_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "GRAPHS"
# The graph folder will be useful for plotting the newly segmented data, identifying where errors may be occurring,
# and developing a plan of action for accounting for it.
plot_lowest_force_lines = True

if __name__ == '__main__':
    # Loop through numpy files that have been cleaned.
    for file in os.listdir(test_folder):
        if file.endswith('.npy'):
            # Load the data.
            source_file = test_folder + '/' + file
            print(file)
            data = np.load(source_file)
            # Identify locations of change between target locations by using To_From_Home column. Add 1 to get correct
            # index.
            target_row = h.get_first_target_row(data)
            # Get the time locations of the start of a new task.
            t_targets = data[target_row, h.data_header.index('Time')]
            t = data[:, h.data_header.index('Time')]
            v = data[:, h.data_header.index('Vxy_Mag')]
            d = data[:, h.data_header.index('Dist_From_Target')]
            f = data[:, h.data_header.index('Fxy_Mag')]
            # Apply filters to force and velocity data, but force is probably more likely to represent intention than velocity.
            freq1 = 3
            f1 = h.butterworth_filter(f, freq1)
            # find first location of minimum force applied after the start of each task.
            # TODO: Observe how find_peaks function that implements peak prominence is able to identify the peaks in
            #  the signal.
            peaks, properties = signal.find_peaks(f, prominence=2)
            peaks = t[peaks]
            first_min, my_min, min_time = h.find_first_min(t, f1, t_targets, peaks)
            save_name = file[:-12] + '_force_lowest_min_after_task_start'
            save_folder = graph_folder + '/' + save_name
            if plot_lowest_force_lines:
                h.plot_force_and_start_of_task_and_lowest_force_lines(t, t_targets, f1, d, peaks, first_min, save_name,
                                                                      save_folder)



