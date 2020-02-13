import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2870, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 50)
        self.norm1 = nn.BatchNorm1d(256)
        self.norm2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout2d(p=0.1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x