import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle 

class CNN(nn.Module):
    def __init__(self, labels, n_channels, n_samples, dropout=0.2):
        super().__init__()

        n_classes = len(np.unique(labels))
        # Compute class weights as it is an inbalanced dataset 
        _, class_counts = np.unique(labels, return_counts=True)
        self.class_weights = class_counts / class_counts.sum()
        self.class_weights = 1 / self.class_weights
        self.class_weights = self.class_weights / self.class_weights.sum()
        self.dropout = nn.Dropout(dropout)

        # let's have 3 convolutional layers that taper off
        l0_filters = n_channels
        l1_filters = 128
        l2_filters = 64
        l3_filters = 32
        # let's manually setup those layers
        # simple layer 1
        self.conv1 = nn.Conv1d(l0_filters, l1_filters, kernel_size=5)
        self.bn1   = nn.BatchNorm1d(l1_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # simple layer 2
        self.conv2 = nn.Conv1d(l1_filters, l2_filters, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(l2_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # simple layer 3
        self.conv3 = nn.Conv1d(l2_filters, l3_filters, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(l3_filters)
        # and we need an activation function:
        self.act = nn.ReLU()
        self.save_log = {"training_loss": [], "validation_loss": [], "training_accuracy": [], "validation_accuracy": []}

        # now we need to figure out how many neurons we have at the linear layer
        # we can use an example input of the correct shape to find the number of neurons
        example_input = torch.zeros((1, n_channels, n_samples),dtype=torch.float32)
        conv_output   = self.conv_only(example_input)
        size_after_conv = conv_output.view(-1).shape[0]

        # now we can define a linear layer that brings us to the number of classes
        self.output_layer = nn.Linear(size_after_conv, n_classes)        

    def conv_only(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = self.conv_only(x)
        x = x.view(x.shape[0],-1)
        x = self.output_layer(x)
        return x
    
    def fit(self, training_dataloader, validation_dataloader, learning_rate=1e-2, num_epochs=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float32).to(device))
        log = {"training_loss": [], "validation_loss": [], "training_accuracy": [], "validation_accuracy": []}

        for epoch in range(num_epochs):
            self.train()
            for data, labels in training_dataloader:
                optimizer.zero_grad()
                data = data.to(device)
                labels = labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                acc = sum(torch.argmax(output, 1) == labels) / labels.shape[0]
                log["training_loss"].append((epoch, loss.item()))
                log["training_accuracy"].append((epoch, acc))

            scheduler.step()
            self.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, labels in validation_dataloader:
                    data = data.to(device)
                    labels = labels.to(device)
                    output = self.forward(data)
                    loss = loss_function(output, labels)
                    acc = sum(torch.argmax(output, 1) == labels) / labels.shape[0]
                    log["validation_loss"].append((epoch, loss.item()))
                    log["validation_accuracy"].append((epoch, acc))

           
            epoch_trloss = np.mean([i[1] for i in log['training_loss'] if i[0] == epoch])
            epoch_tracc = np.mean([i[1].cpu() for i in log['training_accuracy'] if i[0] == epoch])
            epoch_valacc = np.mean([i[1].cpu() for i in log['validation_accuracy'] if i[0] == epoch])
            epoch_valloss = np.mean([i[1] for i in log['validation_loss'] if i[0] == epoch])

            # Save average to logs 
            self.save_log["training_loss"].append(epoch_trloss)
            self.save_log["validation_loss"].append(epoch_valloss)
            self.save_log["training_accuracy"].append(epoch_tracc)
            self.save_log["validation_accuracy"].append(epoch_valacc)

            print(f"{epoch}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}  val_loss:{epoch_valloss:.2f}  val_acc:{epoch_valacc:.2f}")
        
        # save the logs 
        pickle.dump(self.save_log, open('CNN_training_log.pkl', 'wb'))

        self.eval()
        self.to('cpu')

    def predict(self, x, device='cpu'):
        self.to(device)
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward(x.to(device))
        return np.array([p.argmax().item() for p in preds])

    def predict_proba(self, x, device='cpu'):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        y = self.forward(x.to(device))
        return y.cpu().detach().numpy()
    
#------------------------------------------------#
#             Make it repeatable                 #
#------------------------------------------------#
def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

#------------------------------------------------#
#            Interfacing with data               #
#------------------------------------------------#
# we require a class for our dataset that has the windows and classes saved
# it needs to have a __getitem__ method that returns the data and label for that id.
# it needs to have a __len__     method that returns the number of samples in the dataset.
class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]

def make_data_loader_CNN(windows, classes, batch_size=64):
    # first we make the object that holds the data
    obj = DL_input_data(windows, classes)
    # and now we make a dataloader with that object
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True,
    collate_fn = collate_fn)
    return dl

def collate_fn(batch):
    # this function is used internally by the dataloader (see line 46)
    # it describes how we stitch together the examples into a batch
    signals, labels = [], []
    for signal, label in batch:
        # concat signals onto list signals
        signals += [signal]
        labels += [label]
    # convert back to tensors
    signals = torch.stack(signals)
    labels = torch.stack(labels).long()
    return signals, labels
