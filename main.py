import torch
import torchvision
import lightning as L
from lightning.fabric import seed_everything
from fire import Fire
from datamodule import CRNNDataModule
from preparacion_datos import CaptchaDataset
from crnn import CaptchaCRNN
import numpy as np
from utils import Utils
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ]
)

BATCH_SIZE = 32

class Lightning_CRNN(L.LightningModule):
    def __init__(self, modelo, lr, wd):
        super().__init__()
        self.modelo = modelo
        self.lr = lr
        self.wd = wd
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        with torch.no_grad():
            self.modelo.dense.bias.data[0] = -5.0
    
    def forward(self, x):
        return self.modelo(x)
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.modelo(inputs) # [BS, 50, 20]

        target = Utils.batch_w2i(target, self.modelo.w2i)
        target = target.permute(1, 0) # [BS, 5]

        if batch_idx == 0:
            # print(output.size())
            prediccion = output.softmax(dim = 2)
            prediccion = prediccion.argmax(dim = 2)
            prediccion = Utils.transcription(prediccion, self.modelo.i2w)
            prediccion = Utils.clean(prediccion)
            print(prediccion)
            # print(target.size())
            print(Utils.transcription(target, self.modelo.i2w))

        # revisar lógica permutación y loss
        output = output.permute(1, 0, 2)
        log_prob = torch.nn.functional.log_softmax(output, dim=2)
        T = log_prob.size(0)
        current_batch_size = log_prob.size(1)
        input_lengths = torch.full((current_batch_size,), T, dtype=torch.long)
        target_lengths = torch.full((current_batch_size,), 5, dtype=torch.long)

        loss = self.ctc_loss(log_prob, target, input_lengths, target_lengths)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.modelo(inputs)
        target = Utils.batch_w2i(target, self.modelo.w2i)
        target = target.permute(1, 0)

        prediccion = output.softmax(dim = 2)
        prediccion = prediccion.argmax(dim = 2)
        prediccion = Utils.transcription(prediccion, self.modelo.i2w)
        prediccion = Utils.clean(prediccion)
        ground_truth = Utils.transcription(target, self.modelo.i2w)

        # lista con strings objetivo para calcular cer
        prediccion_string = ["".join(sublist) for sublist in prediccion.tolist()]
        # print(target.size())
        ground_truth_string = ["".join(sublist) for sublist in ground_truth.tolist()]

        output = output.permute(1, 0, 2)
        log_prob = torch.nn.functional.log_softmax(output, dim=2)
        T = log_prob.size(0)
        current_batch_size = log_prob.size(1)
        input_lengths = torch.full((current_batch_size,), T, dtype=torch.long)
        target_lengths = torch.full((current_batch_size,), 5, dtype=torch.long)

        loss = self.ctc_loss(log_prob, target, input_lengths, target_lengths)
        self.log("train_loss", loss, on_epoch=True)

        cer = [Utils.calculate_cer(gts, ps) for gts, ps in zip(ground_truth_string, prediccion_string)]
        cer_mean = sum(cer)/len(cer)
        self.log("character_error_rate", cer_mean, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}
        }

def main(name, seed, accelerator, devices=1):
    seed_everything(seed)

    dataset = CaptchaDataset(augments=transforms)
    dm = CRNNDataModule(dataset=dataset, batch_size=BATCH_SIZE)
    w2i, i2w = dataset.get_w2i()
    modelo = Lightning_CRNN(CaptchaCRNN(w2i, i2w), 1e-4, 1e-5)

    # wandb = WandbLogger(project="crnn_captcha", name=name, log_model=False)
    # # early_stopping = EarlyStopping(monitor="character_error_rate", mode="min", verbose=True, patience=4, min_delta=0.02)
    # ckpt = ModelCheckpoint(
    #     dirpath="weights",
    #     filename=f"{100}ep-{name}",
    #     verbose=True,
    #     monitor="character_error_rate", 
    #     mode="min"
    # )
    summary = ModelSummary(max_depth=3)


    trainer = L.Trainer(accelerator=accelerator, devices=devices, max_epochs=200, callbacks=[summary])
    trainer.fit(modelo, datamodule=dm)

    # print(np.array(dataset.labels))

    # for data in dm.train_dataloader():
    # #     # data es una lista con el tensor de la imagen y una tupla que contiene SOLO el string objetivo
    #     output_batch = modelo(data[0])
    #     print(output_batch.shape)
    #     # convierte al mismo formato que el output
    #     # print(np.array([[letra for letra in label] for label in data[1]]))
    #     break

if __name__ == '__main__':
    Fire(main)
