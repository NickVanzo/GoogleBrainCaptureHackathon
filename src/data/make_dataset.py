import os
import mne
import numpy as np
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt
import sys

sys.path[0] = os.getcwd()

from src.data.utils.eeg import get_raw
from src.data.processing import load_data_dict, get_data
from src.data.conf.eeg_annotations import braincapture_annotations

import logging

# Suppress logger messages from MNE-Python
mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.ERROR)

# Fetch the data
os.chdir("/home/jupyter")
braincapture_data_folder = r'copenhagen_medtech_hackathon/BrainCapture Dataset/'
subject_folders = os.listdir(braincapture_data_folder)
number_files = np.sum([len(os.listdir(braincapture_data_folder + folder)) for folder in subject_folders])


# labels = True for SL or False for UL.
data_dict = load_data_dict(data_folder_path=braincapture_data_folder, annotation_dict=braincapture_annotations, tmin=-0.5, tlen=6, labels=True)


all_subjects = list(data_dict.keys())
X, y = get_data(data_dict, all_subjects)


os.chdir(os.path.join(os.getcwd(), "GoogleBrainCaptureHackathon", "data", "processed"))
torch.save(X, "X_data.pt")
torch.save(y, "y_data.pt")
print('X saved, X.shape:', X.shape)
print('y saved, y.shape:', y.shape)