import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class lstm_relation_net(nn.Module):
    def __init__(self, num_models, input_size, encode_size, hidden_size, num_classes, num_layers):
        super(lstm_relation_net, self).__init__()
        
        # define the properties
        self.input_size = input_size
        self.encode_size = encode_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        self.fc_in = nn.ModuleList([nn.Linear(input_size[i], encode_size) for i in range(num_models)]) # intput fully connected layer
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(encode_size)for i in range(num_models)])
        self.lstm = nn.LSTM(input_size=encode_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True) # lstm cell
        self.bn2   = nn.BatchNorm1d(2 * hidden_size) # bidirectional
        self.fc_out = nn.Linear(in_features=2 * self.hidden_size, out_features=self.num_classes) # output fully connected layer
        
    def forward(self, inputs):
        embed_inputs = torch.empty((inputs.shape[0], len(self.input_size), self.encode_size)).to(inputs.device)
        idx = 0
        for i_model in range(len(self.fc_in)):
            embed_inputs[:, i_model] = self.fc_in[i_model](inputs[:, idx:idx+self.input_size[i_model]])
            embed_inputs[:, i_model] = self.bn1[i_model](embed_inputs[:, i_model])
            idx += self.input_size[i_model]
        embed_inputs = F.relu(embed_inputs)
        self.lstm.flatten_parameters()
        outputs, (h_n, c_n) = self.lstm(embed_inputs)
        outputs = torch.cat((outputs[:, -1, :self.hidden_size], outputs[:, 0, self.hidden_size:]), 1)
        outputs = self.bn2(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc_out(outputs)
        return outputs