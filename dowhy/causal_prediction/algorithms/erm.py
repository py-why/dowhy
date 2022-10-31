import torch
from torch.nn import functional as F
from torch import nn
from dowhy.causal_prediction.algorithms.base_algorithm import Algorithm

class ERM(Algorithm):
    def __init__(self, model, optimizer='Adam', lr=1e-3, weight_decay=0., betas=(0.9, 0.999), momentum=0.9):
        super().__init__(model, optimizer, lr, weight_decay, betas, momentum)


    def training_step(self, train_batch, batch_idx):

        print('train batch', len(train_batch), len(train_batch[0]), train_batch[0][0].shape)
        x = torch.cat([x for x, y, _, _ in train_batch])
        y = torch.cat([y for x, y, _, _ in train_batch])

        out = self.model(x)   
        loss = F.cross_entropy(out, y)
        acc = (torch.argmax(out, dim=1) == y).float().mean()

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss
        
    
    