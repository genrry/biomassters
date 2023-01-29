import torch
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp
from losses import RMSELoss

LR = 5e-4

class Model(LightningModule):

    def __init__(self, in_channels):
        super().__init__()
        
        self.segmentation_model = smp.Unet(
            encoder_name="efficientnet-b5",
            encoder_weights= None, # 'imagenet' weights don't seem to help so start clean 
            in_channels=in_channels,                 
            classes=1,                     
        )
        self.loss_fn = RMSELoss()

    def forward(self, x):
        agbm = self.segmentation_model(x)
        return agbm
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.segmentation_model.parameters(), lr=LR)

    
    def training_step(self, train_batch, batch_idx):
        image = train_batch['image']
        label = train_batch['label']
        pred = self.forward(image)
        loss = self.loss_fn(pred, label)
        self.log("train_loss", torch.round(loss, decimals=5))  
        return loss

    def validation_step(self, val_batch, batch_idx):
        image = val_batch['image']
        label = val_batch['label']
        pred = self.forward(image)
        loss = self.loss_fn(pred, label)
        self.log("val_loss", torch.round(loss, decimals=5))  

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()