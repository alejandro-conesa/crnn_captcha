import torch
import torchvision

class CaptchaCRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()