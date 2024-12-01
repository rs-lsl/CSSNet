# -*- coding: utf-8 -*-
"""
License: MIT
@author: lsl
E-mail: 987348977@qq.com
低分辨率实施
"""
import sys

sys.path.append("/home/aistudio/code")
sys.path.append("/home/aistudio/code/pansharpening/ours")
sys.path.append("/home/aistudio/code/pansharpening/Pan-GAN")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

# from Pfnet_structure_real import Pf_net  # 还有 Pfnet_structure2
from ours.ournet_structure_red_1014 import Our_net  # 还有 Pfnet_structure2
# from pangan_structure_red import lap_conv
from ours.ournet_dataset import Mydata
# from loss_function import Uiqi
# from loss_function import Uiqi_pan
from loss_function import SSIMLoss
from metrics import psnr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#  hyper parameters
test_batch_size = 1
# cpu_place = torch.cpu()

def high_level(hs, high_hs, batch_size, band_hs, win_size, h, w):
    index_map = torch.argmax(high_hs, dim=1)  # N H W
    index_map = index_map.reshape([batch_size, -1])  # N H*W
    # min_index = torch.min(index_map)
    # index_map = rescale_value * (index_map - min_index) / (torch.max(index_map) - min_index)

    sort_index = torch.argsort(index_map, dim=1)

    hs0 = hs.reshape([batch_size, band_hs, -1])#.cpu().numpy()
    sorted_hs = []
    for i in range(batch_size):
        sorted_hs.append(hs0[i, :, sort_index[i, :]])
        
    sorted_hs = torch.concat(sorted_hs, 0)
    sorted_hs = sorted_hs.reshape([batch_size, band_hs, win_size, int(h*w/win_size)])
    std_sorted_hs = torch.std(sorted_hs, dim=2)
    return torch.log(torch.sum(std_sorted_hs))


def ournet(train_ms_image, train_pan_image, train_label,
           test_ms_image, test_pan_image, test_label, num_epochs=50,
           mid_ch=64, batch_size=10, learning_rate=1e-4,
           ratio=4, name='zy'):
    # 影像维数
    # hs_data是真反射率数据
    _, band_hs, h, w = train_ms_image.shape
    _, band_ms, H, W = train_pan_image.shape
    num_patch = test_ms_image.shape[0]

    # num_epochs_pre = 1
    # num_epochs = 500

    print(train_ms_image.shape)
    print(test_ms_image.shape)
    #  定义数据和模型
    dataset0 = Mydata(train_ms_image, train_pan_image, train_label)
    train_loader = torch.utils.data.DataLoader(dataset0, num_workers=0, batch_size=batch_size,
                                        shuffle=True, drop_last=True)

    dataset1 = Mydata(test_ms_image, test_pan_image, test_label)
    test_loader = torch.utils.data.DataLoader(dataset1, num_workers=0, batch_size=test_batch_size,
                                        shuffle=False, drop_last=False)

    model = Our_net(band_hs, band_ms, mid_ch=mid_ch, ratio=ratio).to(device)

    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif len(param.shape) > 1:
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
        else:
            nn.init.constant_(param, 0)
            
    # last_recon = recon(band_hs, ks=3)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    Pixel_loss = nn.L1Loss()
    # lap_conv0 = lap_conv(1)
    ssim_loss = SSIMLoss(window_size=11, sigma=1.5, data_range=1, channel=band_hs)

    loss_save = []
    loss_save_test = []

    # scheduler_pre = optim.lr.MultiStepDecay(1e-4, [100], gamma=0.1, verbose=False)

    # if name == 'zy' or 'chikusei':400
    step = [200, 400]
    print('setp', step)

    # optimizer_pre = optim.Adam(learning_rate=scheduler_pre, parameters=encoder_hs_net.parameters()+
    #     encoder_ms_net.parameters()+decoder_hs_net.parameters()+decoder_ms_net.parameters())
    optimizer = optim.Adam(lr=learning_rate, params=model.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, step, gamma=0.5)

    win_size = 9  # compute the variance. it should be divided by the h*w

    for epoch in range(num_epochs):
        time0 = time.time()
        loss_total = 0.0
        loss_total_test = 0.0
        # psnr0 = 0.0

        model.train()

        scheduler.step(epoch)
        for i, (hs, ms, labels) in enumerate(train_loader):

            # lap = torch.to_tensor(lap_matrix, dtype='float32')
            # result, result_hs = model(images_hs, images_pan)
            
            # mean_hs = torch.mean(hs, axis=1, keepdim=True)
            # mean_hs_expand = torch.tile(mean_hs, [1, band_ms, 1, 1])
            # result_hs2 = low_transformer(hs, ms, mean_hs_expand)
            
            # result_hs = down_dim(torch.concat([result_hs, result_hs2], axis=1))
            # print(result_hs.shape)
            hs = hs.to(device, dtype=torch.float32)
            ms = ms.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()

            result_hs, high_hs, high_ms = model(hs, ms)

            hs_var = high_level(hs, high_hs, batch_size, band_hs, 9, h, w)
            ms_var = high_level(ms, high_ms, batch_size, band_ms, 16, H, W)
            ssim_loss_term = ssim_loss(labels, result_hs)
            hs_loss = Pixel_loss(result_hs, labels)
            # print(torch.log(hs_var), torch.log(ms_var))

            loss = hs_loss + hs_var*0.01 + ms_var*0.01 + ssim_loss_term*0.1 #+ bicubic_down_loss*0.1 + bicubic_up_loss*0.1

            loss.backward()
            optimizer.step()

            loss_total += loss.numpy()

        if ((epoch + 1) % 50) == 0:
            psnr0 = 0.0

            model.eval()

            with torch.no_grad():
                for (hs, ms, image_label) in test_loader:
                    result_hs, _, _ = model(hs, ms)

                    psnr0 += psnr(result_hs.numpy(), image_label.numpy())
            print('epoch %d of %d, using time: %.2f , loss of train: %.4f, Psnr: %.4f' %
                  (epoch + 1, num_epochs, time.time() - time0, loss_total, psnr0 / num_patch))
            # print('epoch %d of %d, using time: %.2f , loss of train: %.4f' %
            #       (epoch + 1, num_epochs, time.time() - time0, loss_total))

    # torch.save(model.state_dict(), '/home/aistudio/result/parameters/ournet_hm_'+name+'.pth')
    # 测试模型
    # model_dict =torch.load('/home/aistudio/result/parameters/ournet_hm_'+name+'.pth')    # 加载训练好的模型参数
    # model.set_state_dict(model_dict)

    model.eval()
    image_all = []
    # label_all = []
    with torch.no_grad():

        time_all = []
        for (hs, ms, _) in test_loader:
            time0 = time.time()
            result_hs, _, _ = model(hs, ms)

            time_all.append(time.time() - time0)
            # outputs = outputs_temp.cpu()
            image_all.append(result_hs)
        print("ournet:", np.mean(np.asarray(time_all)))

        a = torch.concat(image_all, 0)
        # print((base[0].parameters())[0])

    return a
