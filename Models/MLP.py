import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np


class MLP(nn.Module):
    def __init__(self, labels, n_features, layers=[128, 64, 32], dropout=0.2):
        n_classes = len(np.unique(labels))
        # Compute class weights as it is an inbalanced dataset 
        _, class_counts = np.unique(labels, return_counts=True)
        self.class_weights = class_counts / class_counts.sum()
        self.class_weights = 1 / self.class_weights
        self.class_weights = self.class_weights / self.class_weights.sum()

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.initial_layer = nn.Linear(n_features, layers[0])
        self.layer1 = nn.Linear(layers[0], layers[1])
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.batch_norm1 = nn.BatchNorm1d(layers[0])
        self.batch_norm2 = nn.BatchNorm1d(layers[1])
        self.batch_norm3 = nn.BatchNorm1d(layers[2])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(layers[-1], n_classes)
        self.n_classes = n_classes

    def forward_once(self, out):
        out = self.initial_layer(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

    def forward(self, x):
        out = self.forward_once(x)
        out = self.output_layer(out)
        return out

    def fit(self, tr_dl, val_dl, learning_rate=1e-2, num_epochs=400, verbose=True, patience=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.2, weight=torch.tensor(self.class_weights, dtype=torch.float32).to(device))
        log = {"training_loss": [], "validation_loss": [], "training_accuracy": [], "validation_accuracy": []}
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.train()
            for data, labels in tr_dl:
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

            self.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, labels in val_dl:
                    data = data.to(device)
                    labels = labels.to(device)
                    output = self.forward(data)
                    loss = loss_function(output, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / len(val_dl.dataset)
            val_loss /= len(val_dl)
            log["validation_loss"].append((epoch, val_loss))
            log["validation_accuracy"].append((epoch, accuracy))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

            scheduler.step()
            if verbose:
                epoch_trloss = np.mean([i[1] for i in log['training_loss'] if i[0] == epoch])
                epoch_tracc = np.mean([i[1].cpu() for i in log['training_accuracy'] if i[0] == epoch])
                print(f"{epoch}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}  val_loss:{val_loss:.2f}  val_acc:{accuracy:.2f}")

        self.eval()
        self.to('cpu')

    def predict(self, x, device='cpu'):
        self.to(device)
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward(x.to(device))
        return np.array([p.argmax().item() for p in preds])

    def predict_proba(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward(x.to('cpu'))
        return preds.detach().cpu().numpy()

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

def make_data_loader(windows, classes, batch_size=1000):
    obj = DL_input_data(windows, classes)
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True)
    return dl