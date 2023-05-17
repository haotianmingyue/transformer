# 开发者 haotian
# 开发时间: 2023/3/3 20:24

import torch.nn as nn
import torch
import torchvision
import time
import torch.nn.functional as F

from swin_transformer import SwinTransformer


def train(model):
    cifar10_train = torchvision.datasets.CIFAR10(root='E:/BaiduNetdiskDownload', train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())

    batch_size = 16
    dataloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True)

    num_epochs = 500

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

        print(
            f' epoch: {i} loss: {total_loss / len(dataloader)}  precison: {t_r / len(dataloader)}  time: {time.time() - start_time}')

        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       f'./model/cifar_10_swin_loss_{total_loss / len(dataloader)}_p_{t_r / len(dataloader)}.pth')


if __name__ == '__main__':
    model = SwinTransformer(
        img_size=32, patch_size=4, in_chans=3, num_classes=10,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, fused_window_process=False
    )

    train(model)

    # x = torch.randn(1, 3, 32, 32)
    # y = model(x)
    # # print(model)
    # print(y.shape)