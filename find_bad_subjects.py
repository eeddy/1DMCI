from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
import numpy as np 
import libemg 

fe = FeatureExtractor()
om = OfflineMetrics()

dataset = get_dataset_list(cross_user=True)['EMGEPN612']()
data = dataset.prepare_data(split=True)

accuracies = []

train_data = data['Train']
test_data = data['Test']

# for s in range(0, 306):
#     s_data = train_data.isolate_data("subjects", [s], fast=True)
#     s_data = s_data.isolate_data("classes", [0,2,3], fast=True)
#     s_data_train = s_data.isolate_data("reps", list(range(0,20)), fast=True)
#     s_data_test = s_data.isolate_data("reps", list(range(20,25)), fast=True)

#     train_windows, train_meta = s_data_train.parse_windows(40, 5)
#     test_windows, valid_meta = s_data_test.parse_windows(40, 5)

#     nm_windows = train_windows[np.where(np.array(train_meta['classes']) == 0)]
#     nm_means = np.mean(np.abs(nm_windows), axis=2)
#     threshold = np.mean(nm_means) + 3 * np.std(nm_means)

#     active_train_windows = train_windows[np.where(np.array(train_meta['classes']) != 0)]
#     active_tw_means = np.mean(np.abs(active_train_windows), axis=2)
#     relabeled_tw = np.where(active_tw_means.mean(axis=1) < threshold)
#     train_meta['classes'][relabeled_tw] = 0 
#     print("Relabeled " + str(relabeled_tw[0].shape[0] / active_train_windows.shape[0]) + " of the training windows as non-active.")

#     active_valid_windows = test_windows[np.where(np.array(valid_meta['classes']) != 0)]
#     active_val_means = np.mean(np.abs(active_valid_windows), axis=2)
#     relabeled_val = np.where(active_val_means.mean(axis=1) < threshold)
#     valid_meta['classes'][relabeled_val] = 0 
#     print("Relabeled " + str(relabeled_val[0].shape[0] / active_valid_windows.shape[0]) + " of the validation windows as non-active.")

#     mapping = {0: 0, 2: 1, 3: 2}
#     train_labels = np.array([mapping[l] for l in train_meta['classes']])
#     valid_labels = np.array([mapping[l] for l in valid_meta['classes']])

#     train_feats = fe.extract_features(['WENG'], train_windows, feature_dic={'WENG_fs': 200})
#     test_feats = fe.extract_features(['WENG'], test_windows, feature_dic={'WENG_fs': 200}) 

#     feature_dic = {
#         'training_features': train_feats,
#         'training_labels': train_labels,
#     }

#     classifier = libemg.emg_predictor.EMGClassifier(model='LDA')
#     classifier.fit(feature_dic)
#     preds, _ = classifier.run(test_feats)
    
#     accuracies.append(om.get_CA(preds, valid_labels))
#     print('Subject ' + str(s) + ' ' + str(OfflineMetrics().get_CA(preds, valid_labels)))

# np.save('Results/train_accuracies.npy', np.array(accuracies))

for s in range(306, 612):
    s_data = test_data.isolate_data("subjects", [s], fast=True)
    s_data = s_data.isolate_data("classes", [0,2,3], fast=True)
    s_data_train = s_data.isolate_data("reps", list(range(0,20)), fast=True)
    s_data_test = s_data.isolate_data("reps", list(range(20,25)), fast=True)

    train_windows, train_meta = s_data_train.parse_windows(40, 5)
    test_windows, valid_meta = s_data_test.parse_windows(40, 5)

    nm_windows = train_windows[np.where(np.array(train_meta['classes']) == 0)]
    nm_means = np.mean(np.abs(nm_windows), axis=2)
    threshold = np.mean(nm_means) + 3 * np.std(nm_means)

    active_train_windows = train_windows[np.where(np.array(train_meta['classes']) != 0)]
    active_tw_means = np.mean(np.abs(active_train_windows), axis=2)
    relabeled_tw = np.where(active_tw_means.mean(axis=1) < threshold)
    train_meta['classes'][relabeled_tw] = 0 
    print("Relabeled " + str(relabeled_tw[0].shape[0] / active_train_windows.shape[0]) + " of the training windows as non-active.")

    active_valid_windows = test_windows[np.where(np.array(valid_meta['classes']) != 0)]
    active_val_means = np.mean(np.abs(active_valid_windows), axis=2)
    relabeled_val = np.where(active_val_means.mean(axis=1) < threshold)
    valid_meta['classes'][relabeled_val] = 0 
    print("Relabeled " + str(relabeled_val[0].shape[0] / active_valid_windows.shape[0]) + " of the validation windows as non-active.")

    mapping = {0: 0, 2: 1, 3: 2}
    train_labels = np.array([mapping[l] for l in train_meta['classes']])
    valid_labels = np.array([mapping[l] for l in valid_meta['classes']])

    train_feats = fe.extract_features(['WENG'], train_windows, feature_dic={'WENG_fs': 200})
    test_feats = fe.extract_features(['WENG'], test_windows, feature_dic={'WENG_fs': 200}) 

    feature_dic = {
        'training_features': train_feats,
        'training_labels': train_labels,
    }

    classifier = libemg.emg_predictor.EMGClassifier(model='LDA')
    classifier.fit(feature_dic)
    preds, _ = classifier.run(test_feats)
    
    accuracies.append(om.get_CA(preds, valid_labels))
    print('Subject ' + str(s) + ' ' + str(OfflineMetrics().get_CA(preds, valid_labels)))

np.save('Results/test_accuracies.npy', np.array(accuracies))