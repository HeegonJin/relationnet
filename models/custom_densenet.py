import torch
import torchvision
import torch.nn as nn


class CustomDensenet(nn.Module):

    def __init__(self):
        super(CustomDensenet, self).__init__()
        
        self.base_model = torchvision.models.densenet201(pretrained=False)
        self.base_model.classifier = nn.Identity()
        self.fc = torch.nn.Sequential(
                    torch.nn.Linear(1920, 1024, bias = True),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(1024, 512, bias = True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(512, 104))
        
    def forward(self, inputs):
        x = self.base_model(inputs)
        return self.fc(x)