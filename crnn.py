import torch
import torchvision
import numpy as np

class CaptchaCRNN(torch.nn.Module):
    def __init__(self, w2i, i2w):
        super().__init__()
        self.w2i = w2i
        self.i2w = i2w

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding="same", stride=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding="same", stride=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding="same", stride=1)
        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding="same", stride=1)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, padding="same", stride=1)
        self.conv6 = torch.nn.Conv2d(512, 512, kernel_size=3, padding="same", stride=1)
        self.conv7 = torch.nn.Conv2d(512, 512, kernel_size=2, padding="valid", stride=1)

        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.maxpool1x2 = torch.nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.batchnorm2d = torch.nn.BatchNorm2d(512)

        self.lstm1 = torch.nn.LSTM(input_size=1024, hidden_size=256, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)
        self.dense = torch.nn.Linear(512, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool2x2(x)
        x = self.conv2(x)
        x = self.maxpool2x2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool1x2(x)
        x = self.conv5(x)
        x = self.batchnorm2d(x)
        x = self.conv6(x)
        x = self.batchnorm2d(x)
        x = self.maxpool1x2(x)
        x = self.conv7(x)
        x = self.map_to_sequence(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dense(x)
        # x = torch.nn.Softmax(dim=2)(x)
        # x = torch.argmax(x, dim=2)
        # x = self.transcription(x)
        # x = self.clean(x)
        return x
    
    def map_to_sequence(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.flatten(x, start_dim=2)
        return x
