from libemg.datasets import *
dataset = get_dataset_list()['OneSubjectMyo']()
odh = dataset.prepare_data()

print("HERE")