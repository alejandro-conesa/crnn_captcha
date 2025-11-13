import lightning as L
import torch

class CRNNDataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.full_dataset = dataset
        self.stage = None
    
    def setup(self, stage):
        self.stage = stage
        if self.stage == "fit":
            self.train_dataset, self.val_dataset, _ = self.full_dataset.get_splits()
        elif self.stage == "test":
            self.test_dataset = self.full_dataset.get_splits()[2]
    
    def train_dataloader(self):
        self.train_dataset, _, _ = self.full_dataset.get_splits()
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def val_dataloader(self):
        _, self.val_dataset, _ = self.full_dataset.get_splits()
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )