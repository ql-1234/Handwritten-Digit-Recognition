import numpy as np
import torch
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


dataset = mnist.MNIST(root='./data/train', download=True,
                      train=True, transform=ToTensor())

print('dataset content info:')
print('   dataset.data.shape =', dataset.data.shape)
print('dataset.targets.shape =', dataset.targets.shape)
print('       dataset length =', len(dataset))
labels, counts = np.unique(dataset.targets, return_counts=True)
for label, count in zip(labels, counts):
  print('label {} count {}'.format(label, count))

print('dataset[0] info:')
img, target = dataset.__getitem__(0)
print('img.shape =', img.shape)
print('   target =', target)

print('dataset[1] info:')
img, target = dataset[1]
print('img.shape =', img.shape)
print('   target =', target)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

total_epoch = 10
for epoch in range(total_epoch):
  for idx, (x, y) in enumerate(dataloader):
    print('epoch {} iter {}:'.format(epoch, idx))
    print('x.shape =', x.shape)
    print('x.shape =', y.shape)
    exit()
