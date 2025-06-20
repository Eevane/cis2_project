import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# Dataloader
class JointDataset(Dataset):
    def __init__(self, file_path):
        self.inputs = []
        self.targets = []
        with open(file_path, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().replace(',', ' ').split()))
                if len(values) == 9:
                    self.inputs.append(values[:6])
                    self.targets.append(values[6:])
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 增加时间维度 seq_len=1
        return self.inputs[idx].unsqueeze(0), self.targets[idx]


# model
class JointLSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, lstm_layers=1, output_size=3):
        super(JointLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] # (batch_size, seq_len, input_size)
        x = self.relu(self.fc1(last_time_step))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


# training
def train(model, dataloader, optimizer, criterion, device, epochs=20, ckpt_path='checkpoint.pth'):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)


if __name__ == "__main__":
    file_path = "data.txt"
    batch_size = 32
    lr = 1e-3
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = JointDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = JointLSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train(model, dataloader, optimizer, criterion, device, epochs=num_epochs, ckpt_path="joint_lstm_ckpt.pth")
