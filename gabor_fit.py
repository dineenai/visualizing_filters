import numpy as np
import matplotlib.pyplot as plt
!pip install git+https://github.com/mohammadbashiri/fitgabor.git


from fitgabor import GaborGenerator, trainer_fn
from fitgabor.utils import gabor_fn

# Create a model neuron
import torch
from torch import nn
from torch.nn import functional as F

class Neuron(nn.Module):
    def __init__(self, rf):
        super().__init__()
        h, w = rf.shape
        self.rf = torch.tensor(rf.reshape(1, 1, h, w).astype(np.float32))
        
    def forward(self, x):
        return F.elu((x * self.rf).sum()) + 1


theta = -np.pi/4
rf = gabor_fn(theta, sigma=7, Lambda=14, psi=np.pi/2, gamma=1, center=(15, 15), size=(64, 64))



fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
ax.imshow(rf);
ax.set(xticks=[], yticks=[]);


neuron = Neuron(rf)

# Create a gabor generator
_, _, h, w = neuron.rf.shape
torch.manual_seed(420)
gabor_gen = GaborGenerator(image_size=(h, w))

learned_rf = gabor_gen().squeeze().cpu().data.numpy()
true_rf = neuron.rf.squeeze().cpu().data.numpy()

# Gabor vs. true RF before training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
ax1.imshow(learned_rf);
ax1.set(xticks=[], yticks=[], title="True RF")

ax2.imshow(true_rf);
ax2.set(xticks=[], yticks=[], title="Learned gabor");


# Train the gabor generator to maximizes the model output

from torch import optim

gabor_gen, evolved_rfs = trainer_fn(gabor_gen, neuron,save_rf_every_n_epoch=100)

# Learning evolution of the gabor generator
len(evolved_rfs)
n_rows = 4
n_cols = (len(evolved_rfs) + n_rows - 1) // n_rows

fig, axes = plt.subplots(n_rows, n_cols, dpi=100, figsize=(20, 12))

for ind, ax in enumerate(axes.flat):
    if ind < len(evolved_rfs):
        ax.imshow(evolved_rfs[ind])
        ax.set(xticks=[], yticks=[])
    else:
        ax.axis('off')


# Gabor vs. true RF after training
learned_rf = gabor_gen().squeeze().cpu().data.numpy()
true_rf = neuron.rf.squeeze().cpu().data.numpy()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)

# ax2.imshow(true_rf);
# ax1.set(xticks=[], yticks=[], title="True RF")

# ax1.imshow(learned_rf);
# ax2.set(xticks=[], yticks=[], title="Learned gabor");

#What are the parameters?

      

a=gabor_gen.center.detach().numpy()
print(f"{a},{np.array(gabor_gen.image_size)},{gabor_gen.sigma.detach().numpy()},{np.array(gabor_gen.theta)}")
















