# This code will be used to segment the data according to the task number, and at the low point of the exerted force,
# which will be treated as the indicator that a new segment of data has occurred.
# Files are saved into two different folders based on their file type; individual numpy files for each task will be
# saved to a NUMPY_FILES folder. Files that are of the whole task (with first and last targets removed) will be saved to
# a NPZ_FILES folder.

import os
import numpy as np

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

