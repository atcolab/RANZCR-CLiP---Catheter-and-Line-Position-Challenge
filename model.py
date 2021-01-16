import pretrainedmodels

import torch
import torch.nn as nn

model_name = 'resnet34'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

class resnet34(nn.Module):

    def __init__(self, model=model):
        super(resnet34, self).__init__()

        model = model
        model = list(model.children())
        model = nn.Sequential(*model[:-2])

        self.base_model = model
        self.adaptivepooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=512, out_features=11)

    def forward(self, x):

        x = self.base_model(x)
        x = self.adaptivepooling(x)
        x = self.flatten(x)

        x1 = self.fc1(x)

        return x1