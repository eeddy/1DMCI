from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
import numpy as np 
from Models.MLP import * 
from Models.CNN import *

fix_random_seed(42)

fe = FeatureExtractor()
om = OfflineMetrics()

# (1) This will automatically download the EPN dataset if you don't have it. The default split is 306 train and 306 test users.
dataset = get_dataset_list(cross_user=True)['EMGEPN612']()
data = dataset.prepare_data(split=True)

# (2) Get rid of pinch class and isolate a 6 person validation set (you can change the validation if you want)
train_data = data['Train']
train_data = train_data.isolate_data("classes", [0,2,3], fast=True)
valid_data = train_data.isolate_data("subjects", list(range(255, 306)), fast=True)
train_data = train_data.isolate_data("subjects", list(range(0,255)), fast=True)
test_data = data['Test']
test_data = test_data.isolate_data("classes", [0,2,3], fast=True)
train_data = train_data + test_data

print('Loaded ODH...')

# (3) Extracting Windows and Active Thresholding
train_windows, train_meta = train_data.parse_windows(30, 5)
valid_windows, valid_meta = valid_data.parse_windows(30, 5)
mapping = {0: 0, 2: 1, 3: 2}
train_meta['classes'] = np.array([mapping[l] for l in train_meta['classes']])
valid_meta['classes'] = np.array([mapping[l] for l in valid_meta['classes']])

# (4) Fit the model 
train_dataloader = make_data_loader_CNN(train_windows, train_meta['classes'], batch_size=10000)
valid_dataloader = make_data_loader_CNN(valid_windows, valid_meta['classes'], batch_size=10000)
dataloader_dictionary = {"training_dataloader": train_dataloader, "validation_dataloader": valid_dataloader}
cnn = CNN(train_meta['classes'], n_channels = train_windows.shape[1], n_samples  = train_windows.shape[2])
dl_dictionary = {"learning_rate": 1e-2, "num_epochs": 5, "verbose": True, "patience": 5}
cnn.fit(**dataloader_dictionary, **dl_dictionary)
torch.save(cnn, 'Results/UI_CNN.model')