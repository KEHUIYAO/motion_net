import sys
sys.path.insert(1, '../src')


from motion_net import MotionNet
from simulation_dataset import MovingMNIST2
from simulation_dataset_datamodule import DatasetDataModule
from trainer import LightningMotionNet
from torch.utils.data import DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from visualization import plot_spatio_temporal_data


root = '.'
n_frames = 20
num_digits = 2
image_size = 64
digit_size = 28
N = 1 # total number of samples including training and validation data
mask = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
test_data = MovingMNIST2( root,
                     n_frames,
                     mask,
                     num_digits,
                     image_size,
                     digit_size,
                     N,
                     transform=None,
                     use_fixed_dataset=False,
                     random_state=1)
test_data_loader = DataLoader(test_data, 1)

motion_net = MotionNet(channels=1, state_dim=0, action_dim=0, stp=False, cdna=True, dna=False)
learning_rate = 1e-4
model = LightningMotionNet(motion_net, learning_rate)

# load from checkpoint
try:
    model.load_from_checkpoint(checkpoint_path='motion_net_cdna.ckpt', motion_net=motion_net, learning_rate=learning_rate)
except:
    print('fail to load the model')
    pass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for idx, batch in enumerate(test_data_loader):
    idx, mask, images_input, actions, state, images_true = batch
    pred = model(images_input, actions, state, mask)
    pred = torch.stack(pred, dim=1)  # Bx(T-1)xCxHxW
    # loss = self.loss_function(pred, images_true)
    pred = pred.cpu().detach().numpy().squeeze(0).squeeze(1)  # (T-1)xHxW
    images_true = images_true.cpu().detach().numpy().squeeze(0).squeeze(1)  # (T-1)xHxW

    mask = mask[0, :]  # mask is T

    loss = np.mean((pred[mask[1:] == 0, ...] - images_true[mask[1:] == 0, ...])**2)  # only test the prediction on the unobserved data
    print(loss)

    plot_spatio_temporal_data(images_true, save_fig=True, fig_name='motion_net_with_cdna_true', mask=mask[1:])  # plot the true frames
    plot_spatio_temporal_data(pred, save_fig=True, fig_name='motion_net_with_cdna_pred', mask=mask[1:])  # plot the predicted frames

    if idx == 0:
        break