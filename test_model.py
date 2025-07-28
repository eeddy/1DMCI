import torch
from libemg.datasets import *
from libemg.datasets import OfflineMetrics

dataset = get_dataset_list()['OneSubjectMyo']()
odh = dataset.prepare_data()

model = torch.load('UI_CNN.model', weights_only=False)

windows, meta = odh['All'].parse_windows(30, 5)
preds = model.predict(windows)

correction = [4, 1, 0, 3, 2]

test_labels = []
for l in meta['classes']:
    test_labels.append(correction[l])

