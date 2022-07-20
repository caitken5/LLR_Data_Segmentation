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
npz_t_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "NPZ_FILES_BY_TARGET"
npz_f_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "NPZ_FILES_BY_END_OF_FORCE_PROFILE"
graph_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "GRAPHS_BY_END_OF_FORCE_PROFILE"
# The graph folder will be useful for plotting the newly segmented data, identifying where errors may be occurring,
# and developing a plan of action for accounting for it.
plot_lowest_force_lines = True
separate_movement_phases = False
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
            peaks_time = t[peaks]
            first_min, my_min = h.find_first_min(t, f1, target_row, peaks)
            first_min_time = t[first_min]
            save_name = file[:-12] + '_force_lowest_min_after_task_start'
            save_folder = graph_folder + '/' + save_name
            if plot_lowest_force_lines:
                h.plot_force_and_start_of_task_and_lowest_force_lines(t, t_targets, f1, d, peaks_time, first_min_time, save_name,
                                                                      save_folder)
            # Split_by_indices for standard function saving.
            split_by_target = h.split_by_indices(data, np.squeeze(target_row))
            h.save_npz(split_by_target, target_row, npz_t_folder, file)
            # Split_by_indices for new start of force exertion.
            split_by_force = h.split_by_indices(data, first_min)
            h.save_npz(split_by_force, first_min, npz_f_folder, file)
            if separate_movement_phases:
                # Identify if a new reach occurs.
                new_reach_index, d_minima, d_maxima = h.find_end_of_initial_reach(data)
                new_reach_time = t[new_reach_index]
                t_minima = t[d_minima[0]]
                t_maxima = t[d_maxima[0]]
                # Plot the new_reach_index along with the start of each task and the start
                fig, ax = plt.subplots()
                fig.set_size_inches(25, 8)
                # d_d = np.diff(d)  # Through visual inspection, this did not seem to be a good way to distinguish a new reach.
                # Then it will probably be a matter of identifying a zero crossing? We'll see how easy it is...
                [ax.axvline(_t_targets, color='k') for _t_targets in t_targets]
                ax.plot(t, d, 'm-', label="Distance from Target [cm]")
                [ax.axvline(_my_min, color='g') for _my_min in first_min_time]
                ax.plot(t_minima, d[d_minima[0]], 'x', label="Minima in distance from target")
                ax.plot(t_maxima, d[d_maxima[0]], 'x', label="Maxima in distance from target")
                ax.plot(t, f1*5, label='Applied Force')
                # ax.plot(t, v*100, label='Velocity')  # Velocity seems unrelated to the reaches I'm trying to describe.
                # ax.plot(new_reach_time, d[new_reach_index], color='o-', label="New reach indicated")
                # plt.plot(t[:-1], d_d*10, label='Derivative of distance from target')
                plt.xlabel('Time [s]')
                plt.ylabel('Distance from Target [cm]')
                plt.legend()
                plt.show()
                if len(new_reach_index) > 0:
                    # Combine the first_min index with this and re-sort.
                    multi_split_index = []  # Identify if I can rewrite this as a list literal, and how.
                    multi_split_index.append(first_min)
                    multi_split_index.append(new_reach_index)
                    multi_split_index.sort()  # TODO: Confirm that this sorts from least to most.

            # Split_by_indices for separation of reaches for each task ie. does it take multiple times to complete the reach?
