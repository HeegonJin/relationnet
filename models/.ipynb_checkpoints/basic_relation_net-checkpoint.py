'''Basic Relation Network in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class basic_relation_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(basic_relation_net, self).__init__()        
        self.fc1   = nn.Linear(input_size, hidden_size)
        self.bn1   = nn.BatchNorm1d(hidden_size)
        self.do1   = nn.Dropout(p=0.5)
        self.fc2   = nn.Linear(hidden_size, hidden_size)
        self.bn2   = nn.BatchNorm1d(hidden_size)
        self.do2   = nn.Dropout(p=0.5)
        self.fc3   = nn.Linear(hidden_size, output_size)
        self.bn3   = nn.BatchNorm1d(output_size)
        self.do3   = nn.Dropout(p=0.2)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.do1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.do2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.do3(out)
        return out
