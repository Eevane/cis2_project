import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import time

# Dataloader
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class JointDataset(Dataset):
    def __init__(self, file_path, seq_len=5, mode='train', split_ratio=0.7, in_mean=None, in_std=None, tar_mean=None, tar_std=None):
        self.seq_len = seq_len
        assert mode in ['train', 'test', 'evaluation'], "mode should be 'train', 'test' or 'evaluation'"

        # load data
        df = pd.read_csv(file_path)
        raw_in = torch.tensor(df.iloc[:, 2:8].values, dtype=torch.float32)
        raw_tar = torch.tensor(df.iloc[:, 8:].values, dtype=torch.float32)

        # divide training/testing dataset
        split_index = int(len(raw_in) * split_ratio)
        if mode == 'train':
            raw_in = raw_in[:split_index]
            raw_tar = raw_tar[:split_index]
        elif mode == 'test':
            raw_in = raw_in[split_index - seq_len + 1:]
            raw_tar = raw_tar[split_index - seq_len + 1:]
        else:
            assert all(p is not None for p in [in_mean, in_std, tar_mean, tar_std]), \
            "evaluation mode should have mean/std of training dataset."
        # record the range of current dataset
        self.target_range = raw_tar.max(dim=0).values - raw_tar.min(dim=0).values

        # Z-score normalization
        if mode == 'train':
            # compute from training dataset
            self.input_mean = raw_in.mean(dim=0)
            self.input_std = raw_in.std(dim=0) + 1e-8
            self.target_mean = raw_tar.mean(dim=0)
            self.target_std = raw_tar.std(dim=0) + 1e-8
        else:
            # use mean-std of training dataset to normalize testing dataset
            assert torch.is_tensor(in_mean) and torch.is_tensor(in_std)
            assert torch.is_tensor(tar_mean) and torch.is_tensor(tar_std)
            self.input_mean = in_mean
            self.input_std = in_std
            self.target_mean = tar_mean
            self.target_std = tar_std

        norm_inputs = (raw_in - self.input_mean) / self.input_std
        norm_targets = (raw_tar - self.target_mean) / self.target_std

        # sliding window
        self.inputs = []
        self.targets = []
        for i in range(len(norm_inputs) - seq_len + 1):
            input_seq = norm_inputs[i:i+seq_len]           # shape: (seq_len, 6)
            target_val = norm_targets[i+seq_len-1]         # shape: (3,)
            self.inputs.append(input_seq)
            self.targets.append(target_val)
        self.inputs  = torch.stack(self.inputs) 
        self.targets  = torch.stack(self.targets) 

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



# model
class JointLSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, lstm_layers=1, output_size=3):
        super(JointLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, 256), nn.LayerNorm(256), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(True))
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(True))
        self.regression = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #print("x shape:", x.shape) 
        lstm_out, _ = self.lstm(x)   # -> (batch_size, seq_len, input_size)
        last_time_step = lstm_out[:, -1, :]
        x = self.dropout(self.fc1(last_time_step))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = self.regression(x)
        return x

# training
def train(component, model, training_dataloader, testing_dataloader, optimizer, criterion, device, epochs=20, ckpt_path='training_results'):
    model.to(device)

    best_test_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in training_dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(training_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Training average Loss: {avg_loss:.4f}")

        # model evaluation on testing dataset
        running_test_loss = 0.0
        model.eval()
        with torch.inference_mode():
            for x_test, y_test in testing_dataloader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                pred_test = model(x_test)
                test_loss = criterion(pred_test, y_test)
                running_test_loss += test_loss.item()

            avg_test_loss = running_test_loss / len(testing_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Testing average Loss: {avg_test_loss:.4f}")
            print("")
        
        # save the best result
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_test_loss
            }, f'{ckpt_path}/best-{component}.pth.tar')
            print(f"New best model saved at epoch {best_epoch}, best test loss: {best_test_loss:.4f}")

    # Save final results
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, f'{ckpt_path}/final-{component}.pth.tar')


if __name__ == "__main__":
    component = 'puppet-Last'
    training_file_path = f"../../Dataset/train_0627/{component}ThreeJoints.csv"
    #testing_file_path = "../../Dataset/testing_0620/master1-FirstThreeJoints.csv"
    output_path = "training_results/0704"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    batch_size = 64
    lr = 1e-4
    num_epochs = 80
    seq_len= 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dataset = JointDataset(training_file_path,seq_len=seq_len,mode='train')
    in_mean, in_std, tar_mean, tar_std = training_dataset.input_mean, training_dataset.input_std, training_dataset.target_mean, training_dataset.target_std
    testing_dataset = JointDataset(training_file_path, seq_len=seq_len, mode='test', in_mean=in_mean, in_std=in_std, tar_mean=tar_mean, tar_std=tar_std)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    model = JointLSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # save statistics info
    np.savez(f"{output_path}/{component}-stat_params.npz",
        input_mean=in_mean.cpu(),
        input_std=in_std.cpu(),
        target_mean=tar_mean.cpu(),
        target_std=tar_std.cpu(),
        seq_len=seq_len)

    train(component, model, training_dataloader, testing_dataloader, optimizer, criterion, device, epochs=num_epochs, ckpt_path=output_path)
