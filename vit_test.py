# 开发者 haotian
# 开发时间: 2023/3/1 19:36
import time

import torch

from vision_transformer import ViT

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import io
from PIL import Image


def test():
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)
    preds = v(img)

    print(preds.shape)


def vit_train():
    cifar10_train = torchvision.datasets.CIFAR10(root='E:/BaiduNetdiskDownload', train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())

    batch_size = 16
    dataloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True)

    num_epochs = 500
    model = ViT(
        image_size=32,
        patch_size=16,
        num_classes=10,
        dim=128,
        depth=4,
        heads=8,
        mlp_dim=256,
        dropout=0.,
        emb_dropout=0.
    )
    model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_f = torch.nn.MSELoss()
    loss_f = loss_f.cuda()
    for i in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        t_r: int = 0
        for idx, data in enumerate(dataloader):

            images = data[0].float()
            labels = data[1]

            labels_one_hot = F.one_hot(labels, num_classes=10).float()

            images = images.cuda()
            labels_one_hot = labels_one_hot.cuda()

            preds = model(images)

            pred_labels = torch.argmax(preds, dim=-1)

            for j in range(len(pred_labels)):
                if labels[j] == pred_labels[j]:
                    t_r += 1
            # print(preds.shape, labels.shape)
            loss = loss_f(preds, labels_one_hot)
            # print(type(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f' epoch: {i} loss: {total_loss / len(dataloader)}  precison: {t_r/len(dataloader)}  time: {time.time() - start_time}')

        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), f'./model/cifar_10_vit_loss_{total_loss / len(dataloader)}_p_{t_r/len(dataloader)}.pth')


def vit_test():
    cifar10_test = torchvision.datasets.CIFAR10(root='E:/BaiduNetdiskDownload', train=False, download=True,
                                                transform=torchvision.transforms.ToTensor())

    batch_size = 16
    dataloader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, drop_last=True)
    model = ViT(
        image_size=32,
        patch_size=16,
        num_classes=10,
        dim=128,
        depth=4,
        heads=8,
        mlp_dim=256,
        dropout=0.,
        emb_dropout=0.
    )

    model.load_state_dict(torch.load('./model/cifar_10_vit_loss_0.000152640524902381.pth'))
    model = model.cuda()
    model.eval()

    t_r: int = 0
    for idx, data in enumerate(dataloader):
        images = data[0]
        labels = data[1]

        images = images.cuda()
        out = model(images)
        preds = torch.argmax(out, dim=-1)
        for i in range(len(out)):
            if preds[i] == labels[i]:
                t_r += 1

    print('准确率： ', t_r / len(cifar10_test))


if __name__ == '__main__':
    vit_train()







