import numpy as np
import pandas as pd
import os
import time
import cv2

from metrics import ref_evaluate, no_ref_evaluate

from ours.ournet_red_826 import ournet

if __name__ == '__main__':
    # args = parse_args()
    # opt = option.parse(args.opt, args.root_path, is_train=True)

    train_ziyuan = False
    train_chikusei = True
    train_xiongan = False
    if train_ziyuan:
        # band_ms = [1, 2, 3, 4, 5]   # wv2
        from save_image_ziyuan_hp import generate_data, crop_data

        name = 'zy'
        train_num = 350  # please adjust this number according to the crop size
        test_num = 50
        num_epochs = 500
        ms, pan, label = generate_data(ratio=12)
        ms_crop, pan_crop, label_crop = crop_data(ms, pan, label, training_size=16, step_facter=2)

        ms_crop = np.array(ms_crop, dtype='float32')
        pan_crop = np.array(pan_crop, dtype='float32')
        label_crop = np.array(label_crop, dtype='float32')

        del ms, pan, label

    elif train_chikusei:
        from save_image_chikusei_reduce import generate_data_hp, crop_data_hp

        name = 'chikusei'
        ratio_hs = 16
        ms, pan, label = generate_data_hp(ratio=ratio_hs)
        ms_crop, pan_crop, label_crop = crop_data_hp(ms, pan, label, training_size=16)

        # test the code
        # ms_crop = np.ones([5,128,16,16], dtype='float32')
        # pan_crop = np.ones([5,1,16*16,16*16], dtype='float32')
        # label_crop = np.ones([5,128,16*16,16*16], dtype='float32')

        if ratio_hs == 16:
            train_num = 200
            test_num = 50
            num_epochs = 500
        elif ratio_hs == 12:
            train_num = 500
            test_num = 29
            num_epochs = 500
        elif ratio_hs == 4:
            train_num = 1180
            test_num = 80
            num_epochs = 500

        # del ms
        # del pan
        # del label

    elif train_xiongan:
        from save_image_xiongan_reduce import generate_data_hp, crop_data_hp

        name = 'xiongan'
        save_dir = '/home/aistudio/work/Xiongan_hp'
        ratio_hs = 16
        ms, pan, label = generate_data_hp(ratio_hs=ratio_hs)
        ms_crop, pan_crop, label_crop = crop_data_hp(ms, pan, label, training_size=16, ratio=ratio_hs, step_facter=1)
        # np.array(ms_crop, dtype='float32')
        # paddle.to_tensor(ms_crop1, dtype=paddle.float16, place=paddle.CUDAPlace(0))
        # label_crop =  paddle.to_tensor(np.load(save_dir + "/hrhs.npy"), dtype=paddle.float32, place=paddle.CUDAPlace(0))
        # ms_crop = paddle.to_tensor(np.load(save_dir + "/lrhs.npy"), dtype=paddle.float32, place=paddle.CUDAPlace(0))
        # pan_crop = paddle.to_tensor(np.load(save_dir + "/hrpan.npy"), dtype=paddle.float32, place=paddle.CUDAPlace(0))

        if ratio_hs == 16:
            train_num = 60
            test_num = 10
            num_epochs = 500
        elif ratio_hs == 12:
            train_num = 520
            test_num = 50
            num_epochs = 500
        elif ratio_hs == 4:
            train_num = 1180
            test_num = 80
            num_epochs = 500

        # del ms, pan, label

    else:
        print(1)

    mid_ch = 64
    learning_rate = 5e-4
    batch_size = 5 if name == 'xiongan' else 10

    # 监督学习
    index = np.arange(ms_crop.shape[0])
    np.random.seed(1000)
    np.random.shuffle(index)

    train_ms_image = ms_crop[index[:train_num], :, :, :]
    train_pan_image = pan_crop[index[:train_num], :, :, :]
    train_label = label_crop[index[:train_num], :, :, :]

    # index2 = 45
    test_ms_image = ms_crop[index[-test_num:], :, :, :]
    test_pan_image = pan_crop[index[-test_num:], :, :, :]
    test_label = label_crop[index[-test_num:], :, :, :]

    del ms_crop, pan_crop, label_crop

    print(train_ms_image.shape)
    print(train_pan_image.shape)
    print(train_label.shape)

    print(test_ms_image.shape)
    print(test_pan_image.shape)
    print(test_label.shape)

    print(np.max(test_ms_image))
    print(np.max(test_pan_image))
    print(np.max(test_label))

    print(np.mean(test_ms_image))
    print(np.mean(test_pan_image))
    print(np.mean(test_label))

    # np.save('/home/aistudio/result/cave_lrhs0.npy', test_ms_image[0, :, :, :])
    # np.save('/home/aistudio/result/cave_ms0.npy', test_pan_image[0, :, :, :])
    # np.save('/home/aistudio/result/cave_hrhs0.npy', test_label[0, :, :, :])

    ratio = int(test_pan_image.shape[2] / test_ms_image.shape[2])
    print('ratio: ', ratio)

    '''setting save parameters'''
    save_num = 10  # 存储测试影像数目
    save_images = False
    # save_channels = [0, 1, 3]  # BGR for worldview2 image
    save_dir = []
    for i7 in range(save_num):
        save_dir.append('/home/aistudio/result/result_hp_' + name + '_cross/results' + str(i7) + '/')
        if save_images and (not os.path.isdir(save_dir[i7])):
            os.makedirs(save_dir[i7])

    '''定义度量指标和度量函数'''
    ref_results = {}
    ref_results.update({'metrics: ': '  PSNR,    SSIM,    SAM,    ERGAS,   SCC,     Q,     RMSE'})  # 记得更新下面数组长度
    no_ref_results = {}
    no_ref_results.update({'metrics: ': '  D_lamda, D_s,    QNR'})
    len_ref_metrics = 7
    len_no_ref_metrics = 3

    result = []
    result_diff = []
    metrics_result_ref = []  # 存储测试影像指标
    metrics_result_noref = []  # 存储测试影像指标

    test_ournet = True

    '''ournet method'''
    if test_ournet:
        # num_epochs = 1
        print('evaluating ournet method')
        fused_image = ournet(
            train_ms_image, train_pan_image, train_label,
            test_ms_image, test_pan_image, test_label,
            num_epochs=num_epochs, mid_ch=mid_ch, learning_rate=learning_rate,
            batch_size=batch_size, ratio=ratio, name=name)

        fused_image = fused_image.numpy()

        ref_results_all = []
        # noref_results_all = []
        for i5 in range(test_ms_image.shape[0]):
            temp_ref_results = ref_evaluate(np.transpose(fused_image[i5, :, :, :], [1, 2, 0]),
                                            np.transpose(test_label[i5, :, :, :], [1, 2, 0]), scale=ratio)
            # temp_noref_results = no_ref_evaluate(transpose_banddim(fused_image[i5, :, :, :], band=0),
            #                                      transpose_banddim(test_pan_image[i5, :, :, :], band=0),
            #                                      transpose_banddim(test_ms_image[i5, :, :, :], band=0), scale=ratio)

            ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))
            # noref_results_all.append(np.expand_dims(temp_noref_results, axis=0))

        ref_results_all = np.concatenate(ref_results_all, axis=0)
        # noref_results_all = np.concatenate(noref_results_all, axis=0)
        ref_results.update({'our       ': np.nanmean(ref_results_all, axis=0)})
        # no_ref_results.update({'our       ': np.mean(noref_results_all, axis=0)})
        # print(np.max(test_ms_image))
        if save_images:
            for j5 in range(save_num):
                np.save(save_dir[j5] + 'our_result_' + name + str(j5) + '.npy',
                        fused_image[j5, :, :, :])
                # np.save(save_dir[j5] + 'our_result' + str(j5) + '.npy',
                #         fused_image[j5, :, :, :])
                result_diff = (
                    np.mean(np.abs(fused_image[j5, :, :, :] - test_label[j5, :, :, :]), axis=0, keepdims=True))
                np.save(save_dir[j5] + 'our_result_diff_' + name + str(j5) + '.npy',
                        result_diff)

                np.save(save_dir[j5] + 'label_' + name + str(j5) + '.npy',
                        test_label[j5, :, :, :])

        # metrics_result_ref.append(np.nanmean(ref_results_all, axis=0, keepdims=True))
        # metrics_result_noref.append(np.mean(noref_results_all, axis=0, keepdims=True))

    print('################## reference comparision #######################')
    for index1, i in enumerate(ref_results):
        if index1 == 0:
            print(i, ref_results[i])
        else:
            print(i, [round(j, 4) for j in ref_results[i]])
    print('################## reference comparision #######################')

    # metrics_result = np.concatenate(metrics_result_ref, axis=0)
    # metrics_index = pd.DataFrame(np.asarray(metrics_result))
    # metrics_index.to_csv('/home/aistudio/result/metrics_index_ref.csv')
