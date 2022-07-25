# This code will be used to segment the data according to the task number, and at the low point of the exerted force,
# which will be treated as the indicator that a new segment of data has occurred.
# Files are saved into two different folders based on their file type; individual numpy files for each task will be
# saved to a NUMPY_FILES folder. Files that are of the whole task (with first and last targets removed) will be saved to
# a NPZ_FILES folder.

import os
import numpy as np
import matplotlib.pyplot as plt

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
demo_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "DEMO_GRAPHS"
# The graph folder will be useful for plotting the newly segmented data, identifying where errors may be occurring,
# and developing a plan of action for accounting for it.
plot_lowest_force_lines = True
separate_movement_phases = False
if __name__ == '__main__':
    # Loop through numpy files that have been cleaned.
    for file in os.listdir(source_folder):
        if file.endswith('.npy'):
            # Load the data.
            source_file = source_folder + '/' + file
            print(file)
            data = np.load(source_file)
            # Identify locations of change between target locations by using To_From_Home column. Add 1 to get correct
            # index.
            target_row = h.get_first_target_row(data)
            target_row = np.squeeze(target_row)
            # Get the time locations of the start of a new task.
            t_targets = data[target_row, h.data_header.index('Time')]
            t = data[:, h.data_header.index('Time')]
            v = data[:, h.data_header.index('Vxy_Mag')]
            d = data[:, h.data_header.index('Dist_From_Target')]
            f = data[:, h.data_header.index('Fxy_Mag')]
            # Apply filters to force and velocity data, but force is probably more likely to represent intention
            # than velocity. In fact, the production of the force (as expected) proceeds the velocity measured by the
            # device.
            freq1 = 3
            f1 = h.butterworth_filter(f, freq1)
            v1 = h.butterworth_filter(v, freq1)
            my_percent_high = 0.5  # Note that this is the same minimum that I select later on.
            my_percent_low = 0.3
            # Use t_targets to segment the data.
            sig = np.copy(f1)  # Use sig so that I can generalize this code later on with velocity if I need to.
            # Create some lists to append calculated values to.
            firstmin_indices = []
            acceptmin_indices = np.asarray([])
            allmin_indices = np.asarray([])
            for i, my_start in enumerate(target_row):
                # We start from the value of my_start, and retrieve the end target, which has value t_targets[i+1]
                # Take care of the case where the last value is selected, so target_row[i+1] would return an error.
                if i == target_row.shape[0] - 1:
                    my_end = t.shape[0] - 1  # Here if we have reached the very last segment.
                else:
                    my_end = target_row[i+1]  # Use the indices rather than the numeric value in the recording time in
                # for the segmentation algorithm.
                segment = sig[my_start:my_end]
                firstmin_index, acceptmin_index, allmin_index = h.find_first_min_2(segment, my_percent_low, my_start)
                firstmin_indices.append(firstmin_index)
                acceptmin_indices = np.hstack((acceptmin_indices, acceptmin_index))
                allmin_indices = np.hstack((allmin_indices, allmin_index))

            # Ensure both acceptmin_indices and allmin_indices are converted to int64 from float64.
            acceptmin_indices = acceptmin_indices.astype(dtype=int)
            allmin_indices = allmin_indices.astype(dtype=int)
            firstmin_time = t[firstmin_indices]
            firstmin_value = sig[firstmin_indices]
            firstmin_pts = np.vstack((firstmin_time, firstmin_value))

            acceptmin_time = t[acceptmin_indices]
            acceptmin_value = sig[acceptmin_indices]
            acceptmin_pts = np.vstack((acceptmin_time, acceptmin_value))

            allmin_time = t[allmin_indices]
            allmin_value = sig[allmin_indices]
            allmin_pts = np.vstack((allmin_time, allmin_value))

            # Now use a for loop to identify the maximums, and the end of reach values.
            firstmax_indices = []
            acceptmax_indices = np.asarray([])
            allmax_indices = np.asarray([])
            for i, my_start in enumerate(firstmin_indices):
                # Use the new segments to identify the maximum values.
                if i == len(firstmin_indices) - 1:
                    my_end = t.shape[0] - 1  # Here if we have reached the very last segment.
                else:
                    my_end = firstmin_indices[i+1]
                # Now separate out the segment between the firstmin_indices.
                segment = sig[my_start:my_end]
                firstmax_index, acceptmax_index, allmax_index = h.find_first_max(segment, my_percent_low, my_start, my_end)
                firstmax_indices.append(firstmax_index)
                acceptmax_indices = np.hstack((acceptmax_indices, acceptmax_index))
                allmax_indices = np.hstack((allmax_indices, allmax_index))
                # TODO: Identify all the minima in this set.
                # TODO: Identify the end of the initial reach.
                # TODO: Identify the acceptable minimum that occurs prior to the maximum value in the reach.

            # Ensure both acceptmin_indices and allmin_indices are converted to int64 from float64.
            acceptmax_indices = acceptmax_indices.astype(dtype=int)
            allmax_indices = allmax_indices.astype(dtype=int)
            firstmax_time = t[firstmax_indices]
            firstmax_value = sig[firstmax_indices]
            firstmax_pts = np.vstack((firstmax_time, firstmax_value))

            acceptmax_time = t[acceptmax_indices]
            acceptmax_value = sig[acceptmax_indices]
            acceptmax_pts = np.vstack((acceptmax_time, acceptmax_value))

            allmax_time = t[allmax_indices]
            allmax_value = sig[allmax_indices]
            allmax_pts = np.vstack((allmax_time, allmax_value))
            # TODO: Test above code by making a graph. Then, add other parts.
            ms = 10
            fig, ax = plt.subplots()
            fig.set_size_inches(25, 8)
            ax.axvline(t_targets[0], color='k', label="Start of Task")
            [ax.axvline(_t_targets, color='k') for _t_targets in t_targets]
            ax.plot(t, sig, 'b-', label="Force Magnitude, 3 Hz filter")
            # ax.plot(allmin_pts[0], allmin_pts[1], 'bo', label="All Minima")
            # ax.plot(acceptmin_pts[0], acceptmin_pts[1], 'c.', label="All Minimum Values Below Threshold")
            ax.plot(firstmin_pts[0], firstmin_pts[1], 'r+', label="First Minimum in Force Application After New Target",
                    markersize=ms)
            # ax.plot(allmax_pts[0], allmax_pts[1], 'bo', label="All Maxima")
            ax.plot(acceptmax_pts[0], acceptmax_pts[1], 'c.', label="All Maximum Values Above Threshold")
            ax.plot(firstmax_pts[0], firstmax_pts[1], 'r+', label="First Valid Maximum in Force Application After "
                                                                  "New Force Segment", markersize=ms)
            ax.grid()
            ax.legend()
            ax.set_title('Identifying Segments from Force Data | Identifying Maximum of Force Profile | ' + file)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Force [N]')
            plt.show()
            plt.close(fig)

            # TODO: When done testing graphs, put code for graphs into a function.
            # IMPORTANT: Code now only works up to here. The rest will be dealt with/deleted soon.
            firstmin = f1[firstmin_index]
            firstmin_time = t[firstmin_index]
            # Use first_min array to segment data and identify the maximum force in that segment.
            fpeaks = []
            fpeaks_index = []
            fmins = []
            fmins_index = []
            for i in range(len(firstmin_index)):
                start = firstmin_index[i]
                if i == len(firstmin_index) - 1:
                    my_end = f1.shape[0]-1
                else:
                    my_end = firstmin_index[i+1]
                f_max, f_index = h.retrieve_maximum_value_per_target(f1, start, my_end)
                fpeaks.append(f_max)
                fpeaks_index.append(f_index)
                # While I'm here, identify the next minimum value in the force after the peak of the initial movement
                # has been reached.
                f_min, f_min_index = h.find_end_of_initial_reach(f1, start, f_max, f_index, my_end, 0.5)
                fmins.append(f_min)
                fmins_index.append(f_min_index)
            fpeaks_time = t[fpeaks_index]
            fmins_time = t[fmins_index]
            # Reorganize the data into paired sets to pass to the function for easier plotting of points and readability.
            firstmin_pts = np.vstack((firstmin_time, firstmin))
            fpeaks_pts = np.vstack((fpeaks_time, fpeaks))
            fmins_pts = np.vstack((fmins_time, fmins))
            save_name = file[:-12] + '_force_lowest_min_after_task_start'
            save_folder = graph_folder + '/' + save_name
            save_stuff = (save_name, save_folder)
            if plot_lowest_force_lines:
                h.plot_force_and_start_of_task_and_lowest_force_lines(t, t_targets, f1, v1, d, firstmin_pts, fpeaks_pts,
                                                                      fmins_pts, save_stuff)
            # Split_by_indices for standard function saving.
            split_by_target = h.split_by_indices(data, np.squeeze(target_row))
            h.save_npz(split_by_target, target_row, npz_t_folder, file)
            # Split_by_indices for new start of force exertion.
            split_by_force = h.split_by_indices(data, firstmin_index)
            h.save_npz(split_by_force, firstmin_index, npz_f_folder, file)
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
                [ax.axvline(_my_min, color='g') for _my_min in firstmin_time]
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
                    multi_split_index.append(firstmin_index)
                    multi_split_index.append(new_reach_index)
                    multi_split_index.sort()  # TODO: Confirm that this sorts from least to most.

            # Split_by_indices for separation of reaches for each task ie. does it take multiple times to complete the reach?
