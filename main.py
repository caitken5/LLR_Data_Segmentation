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
npy_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "NUMPY_FILES"
npz_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "NPZ_FILES"
graph_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
             "GRAPHS"
# The graph folder will be useful for plotting the newly segmented data, identifying where errors may be occurring,
# and developing a plan of action for accounting for it.

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
            # Get the time locations of the start of a new task.
            t_targets = data[target_row, h.data_header.index('Time')]
            t = data[:, h.data_header.index('Time')]
            v = data[:, h.data_header.index('Vxy_Mag')]
            d = data[:, h.data_header.index('Dist_From_Target')]
            f = data[:, h.data_header.index('Fxy_Mag')]
            # Apply filters to force and velocity data, but force is probably more likely to represent intention than velocity.
            freq1 = 3
            f1 = h.butterworth_filter(f, freq1)
            # Identify, based on Algorithm 1 in notes.
            first_min = []
            my_min = (np.asarray(signal.argrelmin(f1)).flatten()).reshape((-1, 1))
            min_time = t[my_min]
            # Based on Algorithm 1, find the first instance of min_ind that is greater than each subsequent t_target.
            for i in t_targets:
                j2 = [j for j in min_time if j > i]
                first_min.append(np.round(j2[0][0], 2))
            # Plot what this looks like.
            fig, ax = plt.subplots()
            fig.set_size_inches(25, 8)
            [ax.axvline(_t_targets, color='k') for _t_targets in t_targets]
            # ax.plot(t, f, 'g-', label="Force Magnitude, no filter")
            ax.plot(t, f1, 'b-', label="Force Magnitude, 3 Hz filter")
            [ax.axvline(_my_min, color='g') for _my_min in first_min]
            # ax.plot(t, d/500, 'b-', label="Distance from Target")
            plt.legend()
            plt.show()
            plt.close()



