from torch.utils.data import Dataset
import torch
import numpy as np

class RelationData(Dataset): 
    def __init__(self, stacked_prediction, target):
        self.x_data = stacked_prediction
        self.y_data = target
#         self.teacher_outputs = teacher_outputs
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.x_data[idx, :]))
        y = torch.from_numpy(np.array(self.y_data[idx]))
#         teacher_outputs = torch.from_numpy(np.array(self.teacher_outputs[idx]))

        return x, y