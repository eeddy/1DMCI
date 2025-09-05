from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
import numpy as np 
from Models.CNN import *

fix_random_seed(42)

fe = FeatureExtractor()
om = OfflineMetrics()

# This will automatically download the EPN dataset if you don't have it. The default split is 306 train and 306 test users.
dataset = get_dataset_list(cross_user=True)['EMGEPN612']()
data = dataset.prepare_data(split=True)

# Remove bad subjects. This was done because many subjects would elict both flexion and extension when returning to rest, for example.
subjects = np.hstack([np.load('Other/train_accuracies.npy'), np.load('Other/test_accuracies.npy')])
good_subjects = np.where(subjects > 0.95)[0]

# Get rid of pinch class and isolate a validation set (you can change the validation if you want)
train_data = data['Train']
test_data = data['Test']
train_data = train_data + test_data
train_data = train_data.isolate_data("classes", [0,2,3], fast=True)
valid_data = train_data.isolate_data("subjects", list(good_subjects[430:450]), fast=True)
train_data = train_data.isolate_data("subjects", list(good_subjects[0:430]), fast=True)

print('Loaded ODH...')

# Extracting Windows and Active Thresholding
train_windows, train_meta = train_data.parse_windows(40, 5)
valid_windows, valid_meta = valid_data.parse_windows(40, 5)

# Using 5 standard deviations above no motion to active threshold the data (removing transient data)
nm_windows = train_windows[np.where(np.array(train_meta['classes']) == 0)]
nm_means = np.mean(np.abs(nm_windows), axis=2)
threshold = np.mean(nm_means) + 5 * np.std(nm_means)

active_train_windows = train_windows[np.where(np.array(train_meta['classes']) != 0)]
active_tw_means = np.mean(np.abs(active_train_windows), axis=2)
relabeled_tw = np.where(active_tw_means.mean(axis=1) < threshold)
train_meta['classes'][relabeled_tw] = 0 
print("Relabeled " + str(relabeled_tw[0].shape[0] / active_train_windows.shape[0]) + " of the training windows as non-active.")

active_valid_windows = valid_windows[np.where(np.array(valid_meta['classes']) != 0)]
active_val_means = np.mean(np.abs(active_valid_windows), axis=2)
relabeled_val = np.where(active_val_means.mean(axis=1) < threshold)
valid_meta['classes'][relabeled_val] = 0 
print("Relabeled " + str(relabeled_val[0].shape[0] / active_valid_windows.shape[0]) + " of the validation windows as non-active.")

mapping = {0: 0, 2: 1, 3: 2}
train_labels = np.array([mapping[l] for l in train_meta['classes']])
valid_labels = np.array([mapping[l] for l in valid_meta['classes']])

print("HERE")
model = torch.load('Other/UI_CNN.model', weights_only=False)
data = model.conv_only(torch.tensor(train_windows, dtype=torch.float32))

print("THERE")