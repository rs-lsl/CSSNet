from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import rotate

class Mydata(Dataset):
    def __init__(self, lrhs, pan, label):
        super(Mydata, self).__init__()
        self.lrhs = lrhs
        self.pan = pan
        self.label = label

    def __getitem__(self, idx):
        assert idx < self.pan.shape[0]

        return self.lrhs[idx, :, :, :], self.pan[idx, :, :, :], self.label[idx, :, :, :]
        # self.spec_graident_weight, self.pan_weightfactor

    def __len__(self):
        return self.pan.shape[0]

def resize(image, size):
    size = check_size(size)
    # image = imresize(image, size)
    image = np.array(Image.fromarray(image).resize(size))
    return image

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size



