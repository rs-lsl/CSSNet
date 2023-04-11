import torch


class SAMLoss(torch.nn.Module):

    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, input, label):
        return _sam(input, label)


def _sam(img1, img2):
    inner_product = torch.sum(img1 * img2, 0)
    img1_spectral_norm = torch.sqrt(torch.sum(img1 ** 2, 0))
    img2_spectral_norm = torch.sqrt(torch.sum(img2 ** 2, 0))
    # numerical stability
    cos_theta = torch.clip(inner_product / (img1_spectral_norm * img2_spectral_norm + 1e-10), min=0, max=1)
    return torch.mean(torch.acos(cos_theta))

def gaussian1d(window_size, sigma):
    ###window_size = 11
    x = torch.arange(window_size)
    x = x - window_size//2
    gauss = torch.exp(-x ** 2 / float(2 * sigma ** 2))
    # print('gauss.size():', gauss.size())
    ### torch.Size([11])
    return gauss / gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian1d(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    # print('2d',_2D_window.shape)
    # print(window_size, sigma, channel)
    return _2D_window.expand([channel,1,window_size,window_size])

def _ssim(img1, img2, window, window_size, channel=126 ,data_range = 255.,size_average=True,C=None):
    # size_average for different channel

    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    # print(mu1.shape)
    # print(mu1[0,0])
    # print(mu1.mean())
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    if C ==None:
        C1 = (0.01*data_range) ** 2
        C2 = (0.03*data_range) ** 2
    else:
        C1 = (C[0]*data_range) ** 2
        C2 = (C[1]*data_range) ** 2
    # l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    sc = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    lsc = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1))*sc

    if size_average:
        ### ssim_map.mean()是对这个tensor里面的所有的数值求平均
        return lsc.mean()
    else:
        # ## 返回各个channel的值
        return lsc.flatten(2).mean(-1),sc.flatten(2).mean(-1)

class SSIMLoss(torch.nn.Module):
   """
   1. 继承paddle.nn.Layer
   """
   def __init__(self, window_size=11, channel=3, data_range=255., sigma=1.5):
       """
       2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
       """
       super(SSIMLoss, self).__init__()
       self.data_range = data_range
       self.C = [0.01, 0.03]
       self.window_size = window_size
       self.channel = channel
       self.sigma = sigma
       self.window = create_window(self.window_size, self.sigma, self.channel)
       # print(self.window_size,self.window.shape)
   def forward(self, input, label):
       """
       3. 实现forward函数，forward在调用时会传递两个参数：input和label
           - input：单个或批次训练数据经过模型前向计算输出结果
           - label：单个或批次训练数据对应的标签数据
           接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
       """
       # 使用Paddle中相关API自定义的计算逻辑
       # output = xxxxx
       # return output
       return 1-_ssim(input, label,data_range = self.data_range,
                      window = self.window, window_size=self.window_size, channel=self.channel,
                      size_average=True,C=self.C)

def cc_loss(x, y):
   """输入两个向量计算相关系数"""

   x_reducemean = x - torch.mean(x, dim=0)
   y_reducemean = y - torch.mean(y, dim=0)

   fenzi = torch.mean(x_reducemean * y_reducemean)

   return 1 - fenzi / (torch.std(x_reducemean) * torch.std(y_reducemean))