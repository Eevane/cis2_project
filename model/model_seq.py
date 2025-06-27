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
    def __init__(self, file_path, seq_len=5, mode='train', split_ratio=0.7, mean=None, std=None):
        self.seq_len = seq_len
        assert mode in ['train', 'test'], "mode should be 'train' or 'test'"

        # load data
        df = pd.read_csv(file_path)
        inputs = df.iloc[:, 2:8].values.astype('float32')   # (N, 6)
        targets = df.iloc[:, 8:].values.astype('float32')   # (N, 3)

        # divide training/testing dataset
        total_len = len(inputs)
        split_index = int(total_len * split_ratio)
        if mode == 'train':
            inputs = inputs[:split_index]
            targets = targets[:split_index]
        else:
            inputs = inputs[split_index - seq_len + 1:]  # 向前补 seq_len-1 保证滑窗连续
            targets = targets[split_index - seq_len + 1:]

        # Z-score normalization
        if mean is None or std is None:
            input_mean = inputs.mean(axis=0)
            input_std = inputs.std(axis=0) + 1e-8
            target_mean = targets.mean(axis=0)
            target_std = targets.std(axis=0) + 1e-8
        else:
            assert torch.is_tensor(mean) and torch.is_tensor(std)
            mean = mean.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
            input_mean = mean
            input_std = std
            target_mean = targets.mean(axis=0)
            target_std = targets.std(axis=0) + 1e-8
        inputs = (inputs - input_mean) / input_std

        # sliding window
        self.input_mean = torch.tensor(input_mean)
        self.input_std = torch.tensor(input_std)
        self.target_mean = torch.tensor(target_mean)
        self.target_std = torch.tensor(target_std)
        self.inputs = []
        self.targets = []
        for i in range(len(inputs) - seq_len + 1):
            input_seq = inputs[i:i+seq_len]           # shape: (seq_len, 6)
            target_val = targets[i+seq_len-1]         # shape: (3,)
            self.inputs.append(torch.tensor(input_seq, dtype=torch.float32))
            self.targets.append(torch.tensor(target_val, dtype=torch.float32))

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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #print("x shape:", x.shape)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] # (batch_size, seq_len, input_size)
        x = self.dropout(self.fc1(last_time_step))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = self.regression(x)
        return x

# training
def train(component, model, training_dataloader, testing_dataloader, optimizer, criterion, device, epochs=20, ckpt_path='training_results'):
    model.to(device)

    # store mean-std of input data
    input_std = training_dataloader.dataset.input_std.to(device)
    input_mean = training_dataloader.dataset.input_mean.to(device)
    target_std = training_dataloader.dataset.target_std.to(device)
    target_mean = training_dataloader.dataset.target_mean.to(device)

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
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.4f}")
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
            print(f"Epoch {epoch+1}/{epochs}, Testing Loss: {test_loss.item():.4f}")
            print(f"Epoch {epoch+1}/{epochs}, Testing average Loss: {avg_test_loss:.4f}")
            print("")

    # Save final results
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, f'{ckpt_path}/bi-{component}-{str("%.4f" % avg_loss)}.pth.tar')

    np.savez(f"{ckpt_path}/bi-{component}-norm_params.npz",
         input_mean=input_mean.cpu().numpy(),
         input_std=input_std.cpu().numpy(),
         target_mean=target_mean.cpu().numpy(),
         target_std=target_std.cpu().numpy())


if __name__ == "__main__":
    component = 'puppet-Last'
    training_file_path = f"../../Dataset/train_0622/{component}ThreeJoints.csv"
    #testing_file_path = "../../Dataset/testing_0620/master1-FirstThreeJoints.csv"
    output_path = "training_results/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    batch_size = 64
    lr = 1e-4
    num_epochs = 40
    seq_len= 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dataset = JointDataset(training_file_path,seq_len=seq_len,mode='train')
    mean, std = training_dataset.input_mean, training_dataset.input_std
    testing_dataset = JointDataset(training_file_path, seq_len=seq_len, mode='test', mean=mean, std=std)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    model = JointLSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train(component, model, training_dataloader, testing_dataloader, optimizer, criterion, device, epochs=num_epochs, ckpt_path=output_path)
