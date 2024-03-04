import torch
import torch.nn as nn
import torch.nn.functional as F

import os


class Deep_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.norm_layer = nn.LayerNorm(512)
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, output_size)
        self.norm_layer2 = nn.LayerNorm(output_size)
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x = self.norm_layer(x)
        x = self.prelu(self.norm_layer(self.linear1(x)))
        x = self.prelu(self.linear2(x))
        x = self.linear3(x)
        # x = self.norm_layer2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = "./model_folder"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
