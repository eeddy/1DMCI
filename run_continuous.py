import torch
import libemg 
import numpy as np 
import time 
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    _, smm = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smm)
    model = libemg.emg_predictor.EMGClassifier(None)
    model.model = torch.load('Other/UI_CNN.model', weights_only=False)
    model.add_rejection(0.9)
    online = libemg.emg_predictor.OnlineEMGClassifier(model, 40, 5, odh, None, std_out=False)
    online.run(block=True)