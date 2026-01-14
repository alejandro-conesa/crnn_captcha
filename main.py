import torch
import torchvision
import lightning as L
from lightning.fabric import seed_everything
from fire import Fire
from datamodule import CRNNDataModule
from preparacion_datos import CaptchaDataset
from crnn import CaptchaCRNN
import numpy as np
from torchmetrics import Accuracy
from utils import Utils
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger

aumentos = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ]
)

BATCH_SIZE = 4

class Lightning_CRNN(L.LightningModule):
    def __init__(self, modelo, lr):
        super().__init__()
        self.modelo = modelo
        self.lr = lr
    
    def forward(self, x):
        return self.modelo(x)
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.modelo(inputs)
        target = Utils.batch_w2i(target, self.modelo.w2i)

        if batch_idx == 0:
            # print(output.size())
            prediccion = output.softmax(dim = 2)
            prediccion = prediccion.argmax(dim = 2)
            prediccion = Utils.transcription(prediccion, self.modelo.i2w)
            prediccion = Utils.clean(prediccion)
            print(prediccion)
            # print(target.size())
            print(Utils.transcription(target, self.modelo.i2w))

        flatten_target = torch.flatten(target.clone())

        output = output.permute(1, 0, 2)
        log_prob = torch.nn.functional.log_softmax(output, dim=2)
        T = log_prob.size(0)
        input_lengths = torch.full((BATCH_SIZE,), T, dtype=torch.long)
        target_lengths = torch.full((BATCH_SIZE,), 5, dtype=torch.long)

        loss = torch.nn.functional.ctc_loss(log_prob, flatten_target, input_lengths, target_lengths)
        self.log("train_loss", loss, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.modelo(inputs)
        target = Utils.batch_w2i(target, self.modelo.w2i)

        prediccion = output.softmax(dim = 2)
        prediccion = prediccion.argmax(dim = 2)
        prediccion = Utils.transcription(prediccion, self.modelo.i2w)
        prediccion = Utils.clean(prediccion)
        ground_truth = Utils.transcription(target, self.modelo.i2w)

        # lista con strings objetivo para calcular cer
        prediccion_string = ["".join(sublist) for sublist in prediccion.tolist()]
        # print(target.size())
        ground_truth_string = ["".join(sublist) for sublist in ground_truth.tolist()]

        flatten_target = torch.flatten(target.clone())

        output = output.permute(1, 0, 2)
        log_prob = torch.nn.functional.log_softmax(output, dim=2)
        T = log_prob.size(0)
        input_lengths = torch.full((BATCH_SIZE,), T, dtype=torch.long)
        target_lengths = torch.full((BATCH_SIZE,), 5, dtype=torch.long)

        loss = torch.nn.functional.ctc_loss(log_prob, flatten_target, input_lengths, target_lengths)
        self.log("train_loss", loss, on_epoch=True)

        cer = [Utils.calculate_cer(gts, ps) for gts, ps in zip(ground_truth_string, prediccion_string)]
        cer_mean = sum(cer)/len(cer)
        self.log("character_error_rate", cer_mean, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

def main(name, seed, accelerator, devices=1):
    seed_everything(seed)

    dataset = CaptchaDataset(augments=aumentos)
    dm = CRNNDataModule(dataset=dataset, batch_size=BATCH_SIZE)
    w2i, i2w = dataset.get_w2i()
    modelo = Lightning_CRNN(CaptchaCRNN(w2i, i2w), 0.001)

    wandb = WandbLogger(project="crnn_captcha", name=name, log_model=False)
    summary = ModelSummary(max_depth=3)


    trainer = L.Trainer(accelerator=accelerator, devices=devices, max_epochs=10, callbacks=[summary])
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
