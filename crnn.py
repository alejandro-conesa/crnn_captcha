import torch

class CaptchaCRNN(torch.nn.Module):
    def __init__(self, w2i, i2w):
        super().__init__()
        self.w2i = w2i
        self.i2w = i2w

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding="same", stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding="same", stride=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding="same", stride=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding="same", stride=1)

        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=(2,2))
        self.maxpool2x1 = torch.nn.MaxPool2d(kernel_size=(2,1))

        self.batchnorm32 = torch.nn.BatchNorm2d(32)
        self.batchnorm64 = torch.nn.BatchNorm2d(64)
        self.batchnorm128 = torch.nn.BatchNorm2d(128)
        self.batchnorm256 = torch.nn.BatchNorm2d(256)

        self.relu = torch.nn.ReLU()

        self.lstm1 = torch.nn.LSTM(input_size=768, hidden_size=256, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)
        self.dense = torch.nn.Linear(512, len(self.w2i))

        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm32(x)
        x = self.relu(x)
        x = self.maxpool2x2(x)

        x = self.conv2(x)
        x = self.batchnorm64(x)
        x = self.relu(x)
        x = self.maxpool2x2(x)

        x = self.conv3(x)
        x = self.batchnorm128(x)
        x = self.relu(x)
        x = self.maxpool2x1(x)

        x = self.conv4(x)
        x = self.batchnorm256(x)
        x = self.relu(x)
        x = self.maxpool2x1(x)

        x = self.map_to_sequence(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x
    
    def map_to_sequence(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.flatten(x, start_dim=2)
        return x
