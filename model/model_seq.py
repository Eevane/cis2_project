import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import time

# Dataloader
class JointDataset(Dataset):
    def __init__(self, file_path,seq_len):
        self.inputs = []
        self.targets = []
        # read file
        df = pd.read_csv(file_path)
        inputs = df.iloc[:, 2:8].values.astype('float32')   # shape: (N, 6)
        targets = df.iloc[:, 8:].values.astype('float32')   # shape: (N, 3)

        # normalize
        self.input_mean = inputs.mean(axis=0)
        self.input_std = inputs.std(axis=0) + 1e-8  
        inputs = (inputs - self.input_mean) / self.input_std

        # slide window
        self.inputs = []
        self.targets = []
        for i in range(len(inputs) - seq_len + 1):
            input_seq = inputs[i:i+seq_len]           # shape: (seq_len, 6)
            target_val = targets[i+seq_len-1]         # shape: (3,)
            self.inputs.append(torch.tensor(input_seq))
            self.targets.append(torch.tensor(target_val))


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
        self.dropout = nn.Dropout(0.15)

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
def train(model, training_dataloader, testing_dataloader, optimizer, criterion, device, epochs=20, ckpt_path='training_results'):
    model.to(device)

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

        if avg_loss < 3e-2:
            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'{ckpt_path}/m-{str(time.time())}-{str("%.4f" % avg_loss)}.pth.tar')
            
    # Save final results
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'{ckpt_path}/m-{str(time.time())}-{str("%.4f" % avg_loss)}.pth.tar')


if __name__ == "__main__":
    training_file_path = "../../Dataset/train_0620/puppet-FirstThreeJoints.csv"
    testing_file_path = "../../Dataset/testing_0620/puppet-FirstThreeJoints.csv"
    output_path = "training_results/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    batch_size = 64
    lr = 1e-4
    num_epochs = 100
    seq_len= 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dataset = JointDataset(training_file_path,seq_len=seq_len)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    testing_dataset = JointDataset(testing_file_path,seq_len=seq_len)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    model = JointLSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train(model, training_dataloader, testing_dataloader, optimizer, criterion, device, epochs=num_epochs, ckpt_path=output_path)
