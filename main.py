import torch
import torchvision
import lightning as L
from lightning.fabric import seed_everything

aumentos = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ]
)

class CRNNDataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        
