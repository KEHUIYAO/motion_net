import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler




class LightningMotionNet(pl.LightningModule):
    """Pytorch lightning training process for unidirectional convlstm
    """

    def __init__(self, motion_net,
                 learning_rate=1e-3):
        """

        """
        super(LightningMotionNet, self).__init__()
        self.motion_net = motion_net
        self.loss_function = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, images, actions, state, image_mask):

        return self.motion_net(images, actions, state, image_mask)

    def training_step(self, batch, batch_idx):
        [_, mask, images_input, actions, state, images_true] = batch
        pred = self.forward(images_input, actions, state, mask)
        pred = torch.stack(pred, dim=1)  # Bx(T-1)xCxHxW
        loss = self.loss_function(pred, images_true)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        "the function is called after every epoch is completed"

        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #print(avg_loss)
        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):

        [_, mask, images_input, actions, state, images_true] = batch
        pred = self.forward(images_input, actions, state, mask)
        pred = torch.stack(pred, dim=1)  # Bx(T-1)xCxHxW
        loss = self.loss_function(pred, images_true)

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        # calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        self.log('val_loss', avg_loss)  # metric to be tracked
        
        # print(avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'val_loss'}}

