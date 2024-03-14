import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os



class Deep_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.norm_layer = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(hidden_size, output_size)
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

class DQCNN(nn.Module):
    def __init__(self, input_shape, n_actions, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4, padding=0),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.PReLU(),
        )

        conv_out_size = self.get_conv_out_size(input_shape)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.PReLU(),
            nn.Linear(256, n_actions),
        )
        self.output_size = n_actions

    def get_conv_out_size(self, image_dim):
        return np.prod(self.conv(torch.rand(*image_dim)).data.shape)

    def forward(self, inp):

        x = self.conv(inp)
        x = self.fc(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = "./model_folder"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
