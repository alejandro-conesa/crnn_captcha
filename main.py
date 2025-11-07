import torch
import torchvision
import lightning as L
from lightning.fabric import seed_everything
from preparacion_datos import CaptchaDataset

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
        self.full_dataset = CaptchaDataset(augments = aumentos)
        self.stage = None
    
    def setup(self, stage):
        self.stage = stage
        if self.stage == "fit":
            self.train_dataset, self.val_dataset, _ = self.full_dataset.get_splits()
        elif self.stage == "test":
            self.test_dataset = self.full_dataset.get_splits()[2]
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )



