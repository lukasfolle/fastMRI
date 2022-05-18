import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import fastmri
from fastmri.data import transforms


class HammingWindowNetwork(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.hamming_window_layer = HammingWindowLayer(shape)
    
    def forward(self, data):
        data = data.squeeze()
        output = []
        data = self.hamming_window_layer(data)
        for offset in range(data.shape[1]):
            image = fastmri.ifft3c(data[:, offset])
            image = fastmri.complex_abs(image)
            image = fastmri.rss(image, dim=0).squeeze()
            # image = transforms.complex_center_crop_3d(image, (image.shape[0], 128, 128))
            output.append(image)
        output = torch.stack(output, 0)
        return output


class HammingWindowLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        _weight_init = hamming_window_init(torch.ones(shape))
        self.weight = nn.Parameter(_weight_init, requires_grad=True)
    
    def forward(self, data):
        data = data * self.weight.reshape((1, 1, 20, 128, 256, 1))
        return data


def hamming_window_init(data=torch.ones((20, 128, 128))):
    for axis, axis_size in enumerate(data.shape):
        filter_shape = [1, ] * data.ndim
        filter_shape[axis] = axis_size
        window =  torch.hamming_window(axis_size).reshape(filter_shape)
        window = window ** (1.0 / data.ndim)
        data *= window
    return data


class HammingWindowParametrized(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.54))
        self.beta = torch.nn.Parameter(torch.tensor(0.46))
        self.device = device
    
    def hamming_function(self, data):
        window_0 = self.alpha - self.beta * torch.cos(torch.pi * 2 * torch.linspace(0, data.shape[2], data.shape[2], device=self.device) / data.shape[2])
        data = data * window_0.reshape((1, 1, -1, 1, 1, 1))
        window_1 = self.alpha - self.beta * torch.cos(torch.pi * 2 * torch.linspace(0, data.shape[3], data.shape[3], device=self.device) / data.shape[3])
        data = data * window_1.reshape((1, 1, 1, -1, 1, 1))
        window_2 = self.alpha - self.beta * torch.cos(torch.pi * 2 * torch.linspace(0, data.shape[4], data.shape[4], device=self.device) / data.shape[4])
        data = data * window_2.reshape((1, 1, 1, 1, -1, 1))
        return data
    
    def forward(self, data):
        shape = data.shape
        data = data.squeeze()
        data = self.hamming_function(data)
        data = data.reshape(shape)
        return data    
    
        

def main():
    # hw = hamming_window()
    # for k in range(10):
    #     plt.subplot(2, 5, k + 1)
    #     plt.imshow(hw[k * 2], vmin=0, vmax=1)
    # plt.show()
    # hwn = HammingWindowNetwork((20, 128, 256))
    # pred = hwn(torch.rand((28, 8, 20, 128, 256, 2)))
    # print(pred.shape)
    data = torch.ones((1, 28, 8, 20, 100, 100, 2))
    hwp = HammingWindowParametrized()
    res = hwp(data)
    plt.imshow(res[0, 0, 4, 10, ..., 0].detach())#, vmin=-1, vmax=1)
    plt.show()
    


if __name__ == "__main__":
    main()
    