import sys 
sys.path.append('/home/aistudio/code')
sys.path.append('/home/aistudio/code/SSR')
import numpy as np
import pandas as pd
import os
import gc
import random
import time
import signal
import cv2
import paddle
import torch

# from function import transpose_banddim
# from function import generate_cropdata
# from function import Crop_traindata
# from function import blur_downsampling
from function import Downsampler
from function import all_valid
# from vca import vca

def read_srf(filename, skiprows=11):
    data = np.loadtxt(filename, skiprows=skiprows)
    return data

def read_band(filename):  # 读取wv2影像波段和波宽

    data = open(filename, 'r')
    f = data.read()

    band = str.split(f, ',')
    band = [float(x) for x in band]
    length = len(np.array(band))
    band_center = np.array(band)[:int(length/2)]
    band_width = np.array(band)[int(length/2):]

    if band_center[0] > 100:
        return band_center, band_width
    else:
        return band_center*1000, band_width*1000

def generate_srf(filename4):  # 产生和每个高光谱波段
    srf = read_srf(filename4[0])
    srf[:, 0] = srf[:, 0] * 1000
    srf = srf[:, 0:7]  # 放弃海岸线和近红波段

    band_center, band_width = read_band(filename4[1])
    left_band = band_center - band_width / 2.0
    right_band = band_center + band_width / 2.0

    srf_simu = np.expand_dims(np.zeros(6), axis=0)
    for i0 in range(len(band_center)):
        index = np.where((srf[:, 0] >= left_band[i0]) & (srf[:, 0] <= right_band[i0]))
        srf_simu = np.concatenate((srf_simu, np.mean(srf[index, 1:], axis=1)), axis=0)

    band_center = np.expand_dims(band_center, axis=0)
    print(srf_simu.shape)
    srf_simu = np.concatenate((np.transpose(band_center), srf_simu[1:, :]), axis=1)

    return srf_simu

def generate_data(ratio=4):

    patch_size = 64
    band_num = 126
    img_num = 2
    ratio_hs = 16
    ratio_mss = 4
    endmembers_num = 5
    band_mss = 5
    # ratio = 4

    filename5 = [
    '/home/aistudio/work/little_file/envi_plot_wv2.txt',
    '/home/aistudio/work/little_file/hyperspectral_band_Chikusei.txt'
    ]
    srf_simu = np.float32(generate_srf(filename5))
    # srf_pan = np.expand_dims(np.expand_dims(np.load('/home/aistudio/data/data96268/srf_pan.npy'), axis=-1), axis=-1)
    # srf_pan = np.expand_dims(np.expand_dims(np.load('/home/aistudio/work/Chikusei/srf_pan_Chikusei.npy'), axis=-1), axis=-1)

    noise_mean = 0.0
    noise_var = 0.0001

    filename4 = '/home/aistudio/data/data140223/Chikusei.npy'
    save_dir = '/home/aistudio/work/Chikusei_hm'

    original_hs = np.float32(np.load(filename4))  / 16384.0
    original_hs = original_hs[:, :(original_hs.shape[1] - original_hs.shape[1]%64), :(original_hs.shape[2] - original_hs.shape[2]%64)]

    _, row_ori, col_ori = original_hs.shape
    print('original_msi.shape:', original_hs.shape)
    print('max value:', np.max(original_hs))
    # print(i)
    # np.save(save_dir + "/Chikusei_hrhs.npy", original_hs)

    Downsampler4 = Downsampler(ratio, kernel_size=9, padding=3)

    temp_blur_hs = Downsampler4(torch.tensor(np.expand_dims(original_hs, axis=0), dtype=torch.float32'))
    print(temp_blur_hs.shape)
    # np.save(save_dir + "/Chikusei_lrhs.npy", np.squeeze(temp_blur_hs.numpy(), axis=0))

    # 产生多光谱数据
    result = []
    for i2 in range(band_mss):  # worldview2 SRF range of pan include five ms bands
        srf_simu_expand0 = np.expand_dims(np.expand_dims(srf_simu[:, i2 + 2], axis=-1), axis=-1)
        result.append(np.expand_dims(np.sum(original_hs * \
                                            srf_simu_expand0, axis=0) / np.sum(srf_simu_expand0), axis=0))
    result = np.concatenate(result, axis=0)
    # blur_result = blur_downsampling(result, ratio=ratio_mss)
    # for i3 in range(band_mss):
    #     result[i3, :, :] = result[i3, :, :] + \
    #                             np.random.normal(noise_mean, noise_var ** 0.5, [row_ori, col_ori])
    print(result.shape)
    # np.save(save_dir + "/Chikusei_hrms.npy", result)

    return np.squeeze(temp_blur_hs.numpy(), axis=0), result, original_hs

def crop_data(hs_data, pan_data, label_data, training_size=64):
    train_ratio = 0.8
    train_factor = 1
    test_factor = 1  # 数值越大表明测试影像数量越少
    save_num = 5  # 存储测试影像数目
    band = 126
    training_size = 64  # training patch size
    testing_size = 48  # testing patch size
    ratio = 3
    save_dir = '/home/aistudio/work/Chikusei_hm'

    # 加载数据
    # base_dir0 = '/home/aistudio/work/Line'
    # base_dir1 = '/home/aistudio/data/data96268/Line'

    region = 'ziyuan_hm'

    # hs_data = np.float32(np.load('/home/aistudio/work/ziyuan_hm/hs_90m.npy'))
    # pan_data = np.float32(np.load('/home/aistudio/work/ziyuan_hm/ms_30m.npy'))
    # label_data = np.float32(np.load('/home/aistudio/work/ziyuan_hm/hs_30m.npy'))

    # hs_test = np.float32(np.load('/home/aistudio/work/ziyuan_hm/hs_30m.npy'))
    # pan_test = np.float32(np.load('/home/aistudio/work/ziyuan_hm/ms_10m.npy'))

    train_hs_image, train_pan_image, train_label = \
        Crop_traindata_three(hs_data,             # 可调整size
                            pan_data,
                            label_data,
                            training_size,
                            test=False, step_facter=2)

    # test_hs_image, test_pan_image = \
    #     Crop_traindata_two(hs_test,
    #                     pan_test,
    #                     testing_size,
    #                     test=True, step_facter=1)

    # print('train size:' + str(train_hs_image.shape))
    # print('test size:' + str(test_hs_image.shape))
    # np.save(save_dir + "/hrhs.npy", train_label)
    # np.save(save_dir + "/lrhs.npy", train_hs_image)
    # np.save(save_dir + "/hrms.npy", train_pan_image)

    return train_hs_image, train_pan_image, train_label

def generate_data_hp(ratio_hs=4):

    patch_size = 64
    band_num = 126
    img_num = 2
    # ratio_hs = 16
    ratio_mss = 4
    endmembers_num = 5
    band_mss = 5
    ratio = 4

    filename5 = [
    '/home/aistudio/work/little_file/envi_plot_wv2.txt',
    '/home/aistudio/work/little_file/hyperspectral_band_Chikusei.txt'
    ]
    srf_simu = np.float32(generate_srf(filename5))
    srf_pan = srf_simu[:, 1][:, None, None]
    # srf_pan = np.expand_dims(np.expand_dims(np.load('/home/aistudio/data/data96268/srf_pan.npy'), axis=-1), axis=-1)
    # srf_pan = np.expand_dims(np.expand_dims(np.load('/home/aistudio/work/Chikusei/srf_pan_Chikusei.npy'), axis=-1), axis=-1)

    noise_mean = 0.0
    noise_var = 0.0001

    dowmsample16 = Downsampler(ratio_hs, kernel_size=9, padding=3)

    path = '/home/aistudio/data/data95831/Chikusei.npy'
    original_msi = np.float32(np.load(path)) / 16384.0
    # original_msi = np.random.rand(126, 1000, 1000)
    band, row, col = original_msi.shape

    original_msi = original_msi[:, :(row - row%(ratio_hs*4)), :(col - col%(ratio_hs*4))]
    print('original_msi.shape:', original_msi.shape)
    print('max value:', np.max(original_msi))

    # ratio 16
    temp_blur = dowmsample16(paddle.to_tensor(np.expand_dims(original_msi, axis=0), dtype='float32'))
    print(temp_blur.shape)
    temp_blur = np.squeeze(temp_blur.numpy())
    # _, rows, cols = temp_blur.shape
    # blur_data = []
    # for i3 in range(temp_blur.shape[0]):
    #     blur_data.append(np.expand_dims(temp_blur[i3, :, :] +
    #                                     np.random.normal(noise_mean, noise_var ** 0.5, [rows, cols]), axis=0))
    # blur_data = np.concatenate(blur_data, axis=0)
    # print('blur_data.shape:' + str(blur_data.shape))

    # simulated pan image
    temp_pan = np.expand_dims(np.sum(original_msi * srf_pan, axis=0) / np.sum(srf_pan), axis=0)
    print('temp_pan.shape:' + str(temp_pan.shape))

    return temp_blur, temp_pan, original_msi

def crop_data_hp(hs_data, pan_data, label_data, training_size=64, ratio=16, step_facter=1):
    train_ratio = 0.8
    train_factor = 1
    test_factor = 1  # 数值越大表明测试影像数量越少
    save_num = 5  # 存储测试影像数目
    band = 126
    # training_size = 64  # training patch size
    testing_size = 48  # testing patch size
    # ratio = 3
    save_dir = '/home/aistudio/work/Chikusei_hp'

    # 加载数据
    # base_dir0 = '/home/aistudio/work/Line'
    # base_dir1 = '/home/aistudio/data/data96268/Line'

    region = 'ziyuan_hm'

    # hs_data = np.float32(np.load('/home/aistudio/work/ziyuan_hm/hs_90m.npy'))
    # pan_data = np.float32(np.load('/home/aistudio/work/ziyuan_hm/ms_30m.npy'))
    # label_data = np.float32(np.load('/home/aistudio/work/ziyuan_hm/hs_30m.npy'))

    # hs_test = np.float32(np.load('/home/aistudio/work/ziyuan_hm/hs_30m.npy'))
    # pan_test = np.float32(np.load('/home/aistudio/work/ziyuan_hm/ms_10m.npy'))

    train_hs_image, train_pan_image, train_label = \
        Crop_traindata_three(hs_data,             # 可调整size
                            pan_data,
                            label_data,
                            training_size,
                            test=False, step_facter=step_facter, ratio=ratio)

    # test_hs_image, test_pan_image = \
    #     Crop_traindata_two(hs_test,
    #                     pan_test,
    #                     testing_size,
    #                     test=True, step_facter=1)

    # print('train size:' + str(train_hs_image.shape))
    # print('test size:' + str(test_hs_image.shape))
    # np.save(save_dir + "/hrhs.npy", train_label)
    # np.save(save_dir + "/lrhs.npy", train_hs_image)
    # np.save(save_dir + "/hrms.npy", train_pan_image)

    return train_hs_image, train_pan_image, train_label

def Crop_traindata_two(image_ms, image_pan, size, test=False, step_facter=1, ratio=3):
    image_ms_all = []
    image_pan_all = []

    """crop images"""
    temp_name = 'test' if test else 'train'
    print('croping ' + temp_name + ' images...')
    print(image_ms.shape)

    for j in range(0, image_ms.shape[1] - size, int(size/step_facter)):
        for k in range(0, image_ms.shape[2] - size, int(size/step_facter)):
            temp_image_ms = image_ms[:, j :j + size, k :k + size]
            temp_image_pan = image_pan[:, j*ratio :(j + size)*ratio, k*ratio :(k + size)*ratio]

            if all_valid(temp_image_ms, axis=0):
                image_ms_all.append(temp_image_ms)
                image_pan_all.append(temp_image_pan)

    image_ms_all = np.array(image_ms_all, dtype='float32')
    image_pan_all = np.array(image_pan_all, dtype='float32')

    print("size of test_lrmsimage:{}".
            format(image_ms_all.shape))
    print("size of test_hrpanimage:{}".
            format(image_pan_all.shape))

    return image_ms_all, image_pan_all

def Crop_traindata_three(image_ms, image_pan, image_label, size, test=False, step_facter=1, ratio=3):
    image_ms_all = []
    image_pan_all = []
    label = []

    """crop images"""
    temp_name = 'test' if test else 'train'
    print('croping ' + temp_name + ' images...')
    print(image_ms.shape)

    for j in range(0, image_ms.shape[1] - size, int(size/step_facter)):
        for k in range(0, image_ms.shape[2] - size, int(size/step_facter)):
            temp_image_ms = image_ms[:, j :j + size, k :k + size]
            temp_image_pan = image_pan[:, j*ratio :(j + size)*ratio, k*ratio :(k + size)*ratio]
            temp_label = image_label[:, j*ratio :(j + size)*ratio, k*ratio :(k + size)*ratio]

            if all_valid(temp_image_ms, axis=0):
                # print(temp_image_pan.shape)
                image_ms_all.append(temp_image_ms)
                image_pan_all.append(temp_image_pan)
                label.append(temp_label)

    print(len(image_pan_all))
    image_ms_all = np.array(image_ms_all, dtype='float32')
    image_pan_all = np.array(image_pan_all, dtype='float32')
    image_label_all = np.array(label, dtype='float32')

    print("size of lrmsimage:{}".
            format(image_ms_all.shape))
    print("size of hrpanimage:{}".
            format(image_pan_all.shape))
    print("size of labelimage:{}".
            format(image_label_all.shape))

    return image_ms_all, image_pan_all, image_label_all


def unsupervise_crop():
    img_num = 1
    train_ratio = 0.8
    train_factor = 1
    test_factor = 1  # 数值越大表明测试影像数量越少
    save_num = 5  # 存储测试影像数目
    band = 126
    training_size = 12*8  # training patch size
    testing_size = 16 * 16  # testing patch size
    ratio = 16

    region = 'jiaxing'
    base_dir1 = '/home/aistudio/work/'+region+'/Line'
    base_dir0 = '/home/aistudio/data/data96268/Line'

    # 多张影像
    hs_data = []
    # mss_data = []
    pan_data = []
    label_data = []
    # abun16 = []
    for i2 in range(img_num):
 
        hs_data.append(np.float32(np.load('/home/aistudio/data/data95831/Chikusei.npy') / 16383.0 ))

    # 安全
    assert len(hs_data) == img_num

    result = []
    '''产生训练和测试数据'''

    for i in range(len(hs_data)):
        result.append(Crop_traindata_sig(hs_data[i], size=training_size))

    result = np.concatenate(result, 0)
    print('croped image size:' + str(result.shape))
    print(np.max(result))
    # print(j)

    np.save('/home/aistudio/work/Chikusei/label96.npy', result)

def generate_patch_ms(train_num=1000, ms_band=4):
    hs_data = np.load('/home/aistudio/work/jiaxing/label_hs.npy')
    print(hs_data.shape)
    # test_data = hs_data[1000:, :, :, :]

    filename5 = [
    '/home/aistudio/work/little_file/envi_plot_wv2.txt',
    '/home/aistudio/work/little_file/hyperspectral_band_jiaxing.txt'
    ]
    srf_simu = np.float32(generate_srf(filename5))
    print(srf_simu.shape)

    noise_mean = 0.0
    noise_var = 0.0001

    result = []
    for i2 in range(ms_band):  # worldview2 SRF range of pan include five ms bands
        srf_simu_expand0 = np.reshape(srf_simu[:, i2 + 2], [1, srf_simu.shape[0], 1, 1])
        result.append(np.expand_dims(np.sum(hs_data * \
                                            srf_simu_expand0, axis=1) / np.sum(srf_simu_expand0), axis=1))
    result = np.concatenate(result, axis=1)
    # blur_result = blur_downsampling(result, ratio=ratio_mss)
    print(result.shape)
    print(result[train_num:, :, :, :].shape)
    # for i3 in range(ms_band):
    #     result[:, i3, :, :] = result[:, i3, :, :] + \
    #                             np.random.normal(noise_mean, noise_var ** 0.5, [result.shape[0], result.shape[2], result.shape[3]])

    np.save('/home/aistudio/work/jiaxing/train_ms.npy', result)
    np.save('/home/aistudio/work/jiaxing/test_ms.npy', result[train_num:, :, :, :])


if __name__ == '__main__':
    # generate_data()
    crop_data()
    # unsupervise_crop()
    # generate_patch_ms()

    # hs = np.load('/home/aistudio/data/data96268/Line2-Geo.npy')[:, 400:600, 1000:5000]
    # ms = np.load('/home/aistudio/work/jiaxing/Line-Geo2_ms.npy')[:, 400:600, 1000:5000]
    # filename5 = [
    # '/home/aistudio/work/little_file/envi_plot_wv2.txt',
    # '/home/aistudio/work/little_file/hyperspectral_band_jiaxing.txt'
    # ]
    # srf_simu = np.float32(generate_srf(filename5))
    # srf_simu = np.transpose(srf_simu[:, 2:7])
    # # print(np.sum(srf_simu, axis=1).shape)
    # print(np.tile(np.expand_dims(np.sum(srf_simu, axis=1), axis=1), [1, 5]))
    # # print(srf_pan.shape)
    # print(hs.shape)
    # print(ms.shape)

    # endmember_ms, _, _ = vca(np.reshape(ms, [ms.shape[0], ms.shape[1]*ms.shape[2]]), 5)
    # endmember_hs, _, _ = vca(np.reshape(hs, [hs.shape[0], hs.shape[1]*hs.shape[2]]), 5)
    # endmember_hs2 = np.dot(srf_simu, endmember_hs) / np.tile(np.expand_dims(np.sum(srf_simu, axis=1), axis=1), [1, 5])

    # loss_csv = pd.DataFrame(np.asarray(endmember_hs2))
    # loss_csv.to_csv('/home/aistudio/result/endmember_hs2.csv')

