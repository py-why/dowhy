import torch
from torch.nn import functional as F
import pytorch_lightning as pl

class Algorithm(pl.LightningModule):
    def __init__(self, model, optimizer, lr, weight_decay, betas, momentum):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr = lr  
        self.weight_decay = weight_decay  
        self.betas = betas
        self.momentum = momentum
        
    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, val_batch, batch_idx):
        
        x = val_batch[0]
        y = val_batch[1]

        out = self.model(x)    
        loss = F.cross_entropy(out, y)
        acc = (torch.argmax(out, dim=1) == y).float().mean()

        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):

        x = test_batch[0]
        y = test_batch[1]

        out = self.model(x)   
        loss = F.cross_entropy(out, y) 
        acc = (torch.argmax(out, dim=1) == y).float().mean()

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas
            )
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )

        return optimizer