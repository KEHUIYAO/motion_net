import sys
sys.path.insert(1, '../src')


from motion_net import MotionNet
from simulation_dataset import MovingMNIST2
from simulation_dataset_datamodule import DatasetDataModule
from trainer import LightningMotionNet
import numpy as np
import torch
import pytorch_lightning as pl
from visualization import plot_spatio_temporal_data


root = '.、'
n_frames = 20
num_digits = 2
image_size = 64
digit_size = 28
N = 1000 # total number of samples including training and validation data
mask = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
data = MovingMNIST2( root,
                     n_frames,
                     mask,
                     num_digits,
                     image_size,
                     digit_size,
                     N,
                     transform=None,
                     use_fixed_dataset=False)
data_module = DatasetDataModule(data, 4, 0.5)

motion_net = MotionNet(channels=1, state_dim=0, action_dim=0)
learning_rate = 1e-4
model = LightningMotionNet(motion_net, learning_rate)

# load from checkpoint
try:
    model.load_from_checkpoint(checkpoint_path='motion_net.ckpt', motion_net=motion_net, learning_rate=learning_rate)
except:
    print('fail to load the model')
    pass


if __name__ == "__main__":
    max_epoch = 20
    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=max_epoch, gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=max_epoch)
    trainer.fit(model, data_module)

trainer.save_checkpoint("motion_net.ckpt")