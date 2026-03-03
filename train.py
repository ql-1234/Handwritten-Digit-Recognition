from model import Model
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading dataset ...')
    trn_dataset = mnist.MNIST(root='./data/train', train=True,
                              transform=ToTensor(), download=True)
    tst_dataset = mnist.MNIST(root='./data/test', train=False,
                              transform=ToTensor(), download=True)

    print('building dataloader ...')
    batch_size = 64
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=True)

    model = Model().to(device)
    sgd = SGD(model.parameters(),lr=1e-2,momentum=0.9)
    loss_fn = CrossEntropyLoss()

    print('start training ...')

    total_epoch = 100
    prev_acc = 0.
    for current_epoch in range(total_epoch):
        model.train()
        for idx, (trn_x, trn_y) in enumerate(trn_loader):
            trn_x = trn_x.to(device)
            trn_y = trn_y.to(device)
            sgd.zero_grad()
            trn_y_pred = model(trn_x.float())
            loss = loss_fn(trn_y_pred, trn_y.long())
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        for idx, (tst_x, tst_y) in enumerate(tst_loader):
            tst_x = tst_x.to(device)
            tst_y = tst_y.to(device)
            tst_y_pred = model(tst_x.float()).detach()
            tst_y_pred = torch.argmax(tst_y_pred, dim=-1)
            current_correct_num = tst_y_pred == tst_y
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]

        acc = all_correct_num / all_sample_num
        print('epoch [{:03d}/{:03d}] accuracy: {:.3f}'.format(
            current_epoch + 1, total_epoch, acc), flush=True)
        if not os.path.isdir('models'):
            os.mkdir('models')
        if acc > prev_acc:
            save_file = 'models/lenet-mnist.pkl'
            print('saving current model to {}'.format(save_file))
            torch.save(model.state_dict(), save_file)
        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc

    print('model finished training.')