"""
加入了空间配准模块,提取更多的具有判别性的局部特征
"""
# import sys
#
# sys.path.append("/home/aistudio/code")
import numpy as np
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
import time
from function import extract_image_patches, \
    reduce_mean, reduce_sum, same_padding, lap_conv

up_ratio = 2  # 基本不变
kernelsize_temp = 3
kernelsize_temp2 = 5  # 空间注意力细节程度，越大细节越大
padding_mode = 'circular'
pi = 3.14159265

# from methods.Pfnet.Pfnet import log
# nn.initializer.set_global_initializer(nn.initializer.Normal(mean=0.0, std=1))
# nn.initializer.set_global_initializer(nn.initializer.Uniform())
# nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))


def extract_image_patches2(images, ksizes, strides, rates, shifts, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.shape) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.shape
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    images = torch.roll(images, shifts=shifts[0], dims=2)
    images = torch.roll(images, shifts=shifts[1], dims=3)

    unfold = nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.LeakyReLU()):

        m = [default_conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
        # for layer in m:
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.weight.data.fill_(1)  # 初始化 gamma 为 1
        #         layer.bias.data.fill_(0)

class simple_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernelsize: int = 3):
        super(simple_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU())

    def forward(self, x: torch.Tensor):
        return self.net(x)


class basic_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(basic_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, output_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(output_channel))

    def forward(self, x: torch.Tensor):
        return self.conv2(self.conv1(x))


class res_net_nobn(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(res_net_nobn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Conv2d(mid_channel, output_channel, kernelsize, stride=1,
                               padding_mode=padding_mode,
                               padding=int(kernelsize // 2))

    def forward(self, x: torch.Tensor):
        temp = self.conv1(x)
        temp2 = self.conv2(temp)
        return temp2  # + x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=64, multiply=True):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                )
        self.multiply = multiply
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.multiply == True:
            return x * y
        else:
            return y

class res_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(res_net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Conv2d(mid_channel, output_channel, kernelsize, stride=1,
                               padding_mode=padding_mode,
                               padding=int(kernelsize // 2))

    def forward(self, x: torch.Tensor):
        temp = self.conv1(x)
        temp2 = self.conv2(temp)
        return temp2 + x


# 得到高光谱的high level特征
class encoder_hs(nn.Module):
    def __init__(self, band_in, ks=5, ratio=4, len_res=5, mid_channel=64):
        super(encoder_hs, self).__init__()
        self.ratio = ratio

        self.conv = nn.Sequential(  # 处理全色
            nn.Conv2d(band_in, mid_channel, ks, padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channel, mid_channel, ks, padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(mid_channel))

        self.res0 = nn.ModuleList([res_net(mid_channel, mid_channel, mid_channel=mid_channel,
                                          kernelsize=ks) for _ in range(len_res)])
        for layer in self.conv:
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)  # 初始化 gamma 为 1
                layer.bias.data.fill_(0.1)
    def forward(self, hs):
        x2 = self.conv(hs)
        # print(0, torch.any(torch.isnan(x2)))
        for i in range(len(self.res0)):
            x2 = self.res0[i](x2)
        # print(1, torch.any(torch.isnan(x2)))
        return x2


# 得到多光谱的high level特征
class encoder_ms(nn.Module):
    def __init__(self, band_in, ks=5, ratio=4, len_res=5, mid_channel=64):
        super(encoder_ms, self).__init__()
        self.ratio = ratio

        self.conv = nn.Sequential(  # 处理全色
            nn.Conv2d(band_in, int(mid_channel / 2), ks, padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(int(mid_channel / 2)),
            nn.LeakyReLU(),
            nn.Conv2d(int(mid_channel / 2), mid_channel, ks, padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(mid_channel))

        # self.dense = Dense_block(ks=3, mid_channel=mid_channel, len_dense=5)

        # self.att = attention(ks=3, mid_ch=mid_channel)

        # self.dense2 = Dense_block(ks=3, mid_channel=mid_channel, len_dense=5)

        self.res0 = nn.ModuleList([res_net(mid_channel, mid_channel, mid_channel=mid_channel,
                                          kernelsize=ks) for _ in range(len_res)])
        # self.act = nn.Tanh()

        for layer in self.conv:
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)  # 初始化 gamma 为 1
                layer.bias.data.fill_(0.1)
    def forward(self, ms):
        x2 = self.conv(ms)
        # x2 = self.dense2(self.att(self.dense(x0)))
        # print(2, torch.any(torch.isnan(x2)))
        for i in range(len(self.res0)):
            x2 = self.res0[i](x2)
        # print(3, torch.any(torch.isnan(x2)))
        return x2


# 三个上采样融合网络具有相同的设计但是不同的参数
class Dense_block(nn.Module):
    def __init__(self, ks=3, mid_channel=64, len_dense=5):
        super(Dense_block, self).__init__()

        self.resnet = nn.ModuleList([res_net_nobn(mid_channel * (i + 1), mid_channel,
                                                 mid_channel=mid_channel, kernelsize=ks) for i in
                                    range(len_dense)])  # 修改

        self.down_layer = simple_net(mid_channel * (len_dense + 1), mid_channel)

    def forward(self, x):
        temp_result = self.resnet[0](x)
        result = torch.concat((x, temp_result), 1)

        for i in range(1, 5):
            temp_result = self.resnet[i](result)
            result = torch.concat((result, temp_result), 1)

        return self.down_layer(result) + x


class cross_scale_attention(nn.Module):
    def __init__(self, in_ch, band_hs, shifts, output_pad=0, ks=5, mid_ch=64, ratio=4, stride=4, softmax_scale=10):
        super(cross_scale_attention, self).__init__()

        self.scale = ratio
        self.stride = stride
        self.ks = ks
        self.softmax_scale = softmax_scale
        self.mid_ch = mid_ch
        self.band_hs = band_hs
        self.in_ch = in_ch
        self.output_pad = output_pad # if (ratio % 2) == 1 else 0
        self.shifts = shifts

        self.conv_q = BasicBlock(self.in_ch, self.mid_ch, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())
        self.conv_k = BasicBlock(self.in_ch, self.mid_ch, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())
        self.conv_v = BasicBlock(band_hs, band_hs, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())
        self.conv_result = BasicBlock(band_hs, band_hs, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())

    def forward(self, ms, pan, pan2):  # ms为原始分辨率多光谱影像或多光谱细节
        # 处理k
        k_fea = self.conv_k(pan2)
        # print('1111', torch.any(torch.isnan(k_fea)))
        N, _, h, w = k_fea.shape

        k_patch = extract_image_patches2(k_fea, ksizes=[self.ks, self.ks],
                                        strides=[self.stride, self.stride], rates=[1, 1], shifts=self.shifts, padding='same')

        k_patch = k_patch.reshape([N, self.mid_ch, self.ks, self.ks, -1]).permute(0, 4, 1, 2, 3)
        k_group = torch.split(k_patch, 1, dim=0)
        # 处理q
        q_fea = self.conv_q(pan)
        # print('2222', torch.any(torch.isnan(q_fea)))
        q_group = torch.split(q_fea, 1, dim=0)  # 作为被卷积的对象
        # 处理v
        v_fea = self.conv_v(ms)
        # print('3333', torch.any(torch.isnan(q_fea)))
        v_patch = extract_image_patches2(v_fea, ksizes=[self.ks, self.ks],
                                        strides=[self.stride, self.stride], rates=[1, 1], shifts=self.shifts, padding='same')

        v_patch = v_patch.reshape([N, self.band_hs, self.ks, self.ks, -1]).permute(0, 4, 1, 2, 3)
        v_group = torch.split(v_patch, 1, dim=0)
        result = []
        softmax_scale = self.softmax_scale

        for q, k, v in zip(q_group, k_group, v_group):
            k0 = k[0]
            k0_max = torch.max(torch.sqrt(reduce_sum(torch.pow(k0, 2), axis=[1, 2, 3],
                                                       keepdim=True)))
            k0 = k0 / (k0_max + 1e-5)
            # print(k0_max)
            # print('4444', torch.any(torch.isnan(k0_max)))
            # print('5555', torch.any(torch.isnan(k0)))

            # print(k0.shape)
            q0 = q[0]
            # print(q0.shape)
            q0 = same_padding(torch.unsqueeze(q0, 0), ksizes=[self.ks, self.ks], strides=[self.stride, self.stride],
                              rates=[1, 1])
            weight = F.conv2d(q0, k0, stride=self.stride)

            weight_norm = F.softmax(weight * softmax_scale, dim=1)

            v0 = v[0]
            # print(weight_norm.shape)
            deconv_map = F.conv_transpose2d(weight_norm, v0, stride=self.stride, output_padding=self.output_pad,
                                            padding=0)

            deconv_map = deconv_map / 6

            result.append(deconv_map)

        result = torch.concat(result, dim=0)
        return self.conv_result(result)


class Conv_spe(nn.Module):
    def __init__(self, band_hs, band_ms):
        super(Conv_spe, self).__init__()
        """
        convolution operation on spectral/band dimension. The output attention map is compute on global spatial field.
        input:  1*C*H*W
        filter: 1*c*H*W
        output: C*c
        """
        self.band_ms = band_ms
        self.band_hs = band_hs

    def forward(self, hs, ms):
        
        # assert ms.shape[2] == hs.shape[2]
        result = []

        for i in range(0, self.band_ms):
            result.append(F.conv2d(hs, torch.tile(ms[i:(i+1), :, :, :], [self.band_hs, 1, 1, 1]), stride=ms.shape[2],
                                   groups=self.band_hs))
        # print(result[0].shape)
        return torch.concat(result, 0)

class cross_scale_attention_spe(nn.Module):
    def __init__(self, band_hs, band_ms, ks=5, mid_ch=64, ratio=4, stride=4, softmax_scale=10):
        super(cross_scale_attention_spe, self).__init__()

        self.ratio = ratio
        self.stride = stride
        self.ks = ks
        self.softmax_scale = softmax_scale
        self.mid_ch = mid_ch
        self.band_hs = band_hs
        self.band_ms = band_ms
        # self.in_ch = in_ch
        self.output_pad = 1 if (ratio % 2) == 0 else 0

        # self.spe_conv = Conv_spe(band_hs, band_ms)
        # self.conv_q = nn.Sequential(
        #     BasicBlock(self.band_ms, self.band_ms, kernel_size=3, stride=ratio, bias=True, bn=False,
        #                          act=nn.LeakyReLU()), 
        #     BasicBlock(self.band_ms, self.band_ms, kernel_size=3, stride=ratio, bias=True, bn=False,
        #                          act=nn.LeakyReLU()))  # 处理多光谱
        self.conv_q = BasicBlock(self.band_ms, self.band_ms, kernel_size=4, stride=4, bias=True, bn=False,
                                 act=nn.LeakyReLU())  # kernel_size原先为3，stride=ratio**2
        self.conv_k = nn.Sequential(  # 处理高光谱
            # nn.Upsample(scale_factor=ratio, mode='bicubic'),
            BasicBlock(self.band_hs, self.band_hs, kernel_size=4, stride=4, bias=True, bn=False, act=nn.LeakyReLU())
        )  # kernel_size原先为3，stride=ratio

        self.conv_v = BasicBlock(band_ms, band_ms, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())

        self.conv_result = BasicBlock(band_hs, band_hs, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())

    def forward(self, hrms, msi, hsi):  # ms_f convolve hs_f
        N, _, _, _ = hsi.shape
        # hh = int(h / self.ratio)

        # hrms_group = torch.split(self.conv_v(hrms), N, dim=0)
        hrms_f = self.conv_v(hrms)
        msi_down = F.interpolate(msi, scale_factor=1/self.ratio, mode='bilinear')
        msi_down_f = self.conv_q(msi_down)
        # ms_f_group = torch.split(self.conv_q(msi_down), N, dim=0)

        hs_f = self.conv_k(hsi)

        k0_max = torch.max(torch.sqrt(reduce_sum(torch.pow(msi_down_f, 2), axis=[2, 3],
                                                       keepdim=True)))
        msi_down_f = msi_down_f / k0_max

        att_map = torch.einsum('ijkl, imkl -> ijm', msi_down_f, hs_f)
        att_map = F.softmax(att_map * self.softmax_scale, dim=1)
        results = torch.einsum('ijk, ijmn -> ikmn', att_map, hrms_f) / 6

        return self.conv_result(results)

class recon(nn.Module):
    def __init__(self, band_hs, ks=3, mid_ch=64):
        super(recon, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(band_hs, band_hs, 1),
            nn.Conv2d(band_hs, band_hs, ks, padding=int(ks / 2), padding_mode='circular'),
            nn.Tanh())

    def forward(self, x):
        return self.conv0(x)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class Our_net(nn.Module):
    def __init__(self, band_hs, band_ms, mid_ch=10, ratio=16):  # band is the number of hyperspectral image
        super(Our_net, self).__init__()
        self.band_hs = band_hs
        self.band_ms = band_ms
        self.ratio = ratio

        # # model = Our_net(band_ms, mid_ch=mid_ch, ratio=ratio)
        self.encoder_hs_net = encoder_hs(band_hs, ks=3, ratio=ratio, len_res=5, mid_channel=mid_ch)
        self.encoder_ms_net = encoder_ms(band_ms, ks=3, ratio=ratio, len_res=5, mid_channel=mid_ch)

        unfold_size = 2
        self.transformer = cross_scale_attention(ks=unfold_size, in_ch=mid_ch, shifts=[0, 0], output_pad=0, band_hs=band_hs, mid_ch=mid_ch, ratio=ratio,
                                                 stride=unfold_size, softmax_scale=10)
        self.transformer_red = cross_scale_attention(ks=unfold_size, in_ch=mid_ch, shifts=[0, 1],band_hs=band_hs, mid_ch=mid_ch, ratio=ratio,
                                                     stride=unfold_size, softmax_scale=10)
        self.transformer_red2 = cross_scale_attention(ks=unfold_size, in_ch=mid_ch, shifts=[1, 0],band_hs=band_hs, mid_ch=mid_ch, ratio=ratio,
                                                      stride=unfold_size, softmax_scale=10)
        self.transformer_red3 = cross_scale_attention(ks=unfold_size, in_ch=mid_ch, shifts=[1, 1],band_hs=band_hs, mid_ch=mid_ch, ratio=ratio,
                                                      stride=unfold_size, softmax_scale=10)

        self.transformer_spe = cross_scale_attention_spe(band_hs, band_ms, ks=5, mid_ch=mid_ch, ratio=ratio, softmax_scale=10)

        num = 5
        self.down_dim = nn.Sequential(
            nn.Conv2d(num * band_hs, num*band_hs, kernel_size=11, padding=5, padding_mode=padding_mode, groups=num * band_hs),
            nn.Conv2d(num * band_hs, band_hs, kernel_size=1))
            # nn.Conv2d(band_hs, band_hs, kernel_size=3, padding=1, padding_mode=padding_mode))
            # nn.Tanh())

    def forward(self, hs, ms):
        # print('00', torch.any(torch.isnan(hs)))
        # print('000', torch.any(torch.isnan(ms)))
        ratio_red = 2  # LRHS降的尺度
        # hs and ms's high level feature own the same channel number
        high_hs = self.encoder_hs_net(hs)
        high_ms = self.encoder_ms_net(ms)
        # print(high_hs)
        # print(high_hs)

        result_hs = self.transformer(hs, high_ms, high_hs)
        # print('111', torch.any(torch.isnan(result_hs)))
        result_hs_red = self.transformer_red(hs, high_ms, high_hs)
        # print('222', torch.any(torch.isnan(result_hs_red)))
        result_hs_red2 = self.transformer_red2(hs, high_ms, high_hs)
        # print('333', torch.any(torch.isnan(result_hs_red2)))
        result_hs_red3 = self.transformer_red3(hs, high_ms, high_hs)
        # print('444', torch.any(torch.isnan(result_hs_red3)))

        # print(result_hs.shape)
        result_spe = self.transformer_spe(ms, ms, hs)
        # print('555', torch.any(torch.isnan(result_spe)))
        # exit()
        result_hs = self.down_dim(torch.concat([result_hs, result_hs_red, result_hs_red2, result_hs_red3, result_spe], dim=1)) \
                    + F.upsample(hs, scale_factor=self.ratio, mode='bicubic')
        # print('666', torch.any(torch.isnan(result_hs)))
        return result_hs, high_hs, high_ms   # , hs_red, high_hs_red, hs_red2, high_hs_red2


class dis(nn.Module):
    def __init__(self, band_ms):
        super(dis, self).__init__()
        padding_mode = 'circular'
        norm = spectral_norm
        self.conv0 = norm(nn.Conv2d(band_ms, 16, kernel_size=3, padding=1, padding_mode=padding_mode))

        self.conv = nn.ModuleList([norm(nn.Conv2d(2 ** (i + 4), 2 ** (i + 5), kernel_size=3,
                                                 padding=1, padding_mode=padding_mode)) for i in range(4)])

        self.conv1 = nn.Sequential(
            norm(nn.Conv2d(256, 1, kernel_size=3, padding=1, padding_mode=padding_mode)),
            nn.Sigmoid())

    def forward(self, ms):
        f0 = self.conv0(ms)
        for i in range(4):
            f0 = self.conv[i](f0)
        return self.conv1(f0)


if __name__ == '__main__':
    a = torch.randn([10, 8, 48, 48])
    b = torch.randn([10, 3, 144, 144])
    # c = torch.randn([10, 8, 144, 144])

    transformer = cross_scale_attention_spe(8, 3, ratio=3)

    d = transformer(b, b, a)
    print(d.shape)
