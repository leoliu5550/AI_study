from numpy import outer
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch
import time
def timer(func):
    def warp(*args):
        time_start = time.process_time()
        result = func(*args)
        time_end = time.process_time()
        time_c= time_end - time_start
        print(f"{func.__name__} result {result} cost is {time_c}")
        return 

    return warp


class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
            )
        self.sig = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2
            )
        self.fc = nn.Linear(7*7*1,10)

    
    def forward(self, x):
        
        out = self.cnn1(x)
        out = self.sig(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.sig2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        return out


# With square kernels and equal stride
m = nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = 3, stride=2)
# non-square kernels and unequal stride and with padding
# m = nn.Conv2d(in_channels = 16, out_channels = 33, kernel_size = (3, 5), stride=(2, 1), padding=(4, 2))
# # non-square kernels and unequal stride and with padding and dilation
# m = nn.Conv2d(in_channels = 16, out_channels =33, kernel_size = (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(1,3,32,32)
output = m(input)
print(output.shape)