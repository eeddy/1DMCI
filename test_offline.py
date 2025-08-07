import torch 
import libemg 
import numpy as np 

model = libemg.emg_predictor.EMGClassifier(None)
model.model = torch.load('UI_CNN.model', weights_only=False)
# model.add_rejection(0.97)
# model.add_majority_vote(5)

dataset_folder = 'TestData/' 
regex_filters = [
    libemg.data_handler.RegexFilter(left_bound = "C_", right_bound="_R", values = ["0","1","2"], description='classes'),
    libemg.data_handler.RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2", "3", "4"], description='reps'),
    libemg.data_handler.RegexFilter(left_bound = "/S", right_bound="/", values = ["0", "1", "4", "5", "6"], description='subject'),
]

offline_dh = libemg.data_handler.OfflineDataHandler()
offline_dh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")
train_windows, train_metadata = offline_dh.parse_windows(40, 5)

# Relabel Windows 
nm_windows = train_windows[np.where(np.array(train_metadata['classes']) == 0)]
nm_means = np.mean(np.abs(nm_windows), axis=2)
threshold = np.mean(nm_means) + 3 * np.std(nm_means)

active_train_windows = train_windows[np.where(np.array(train_metadata['classes']) != 0)]
active_tw_means = np.mean(np.abs(active_train_windows), axis=2)
relabeled_tw = np.where(active_tw_means.mean(axis=1) < threshold)
train_metadata['classes'][relabeled_tw] = 0 
print("Relabeled " + str(relabeled_tw[0].shape[0] / active_train_windows.shape[0]) + " of the training windows as non-active.")


for t_i, t in enumerate(train_metadata["classes"]):
    if t == 2:
        train_metadata["classes"][t_i] = 1 
    elif t == 1: 
        train_metadata["classes"][t_i] = 2

preds = model.run(train_windows)[0]
ca = libemg.offline_metrics.OfflineMetrics().get_CA(preds, train_metadata['classes'])
conf_mat = libemg.offline_metrics.OfflineMetrics().get_CONF_MAT(train_metadata['classes'], preds)
libemg.offline_metrics.OfflineMetrics().visualize_conf_matrix(conf_mat)
print("HERE")