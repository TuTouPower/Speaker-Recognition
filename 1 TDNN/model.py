import torch

class Net(torch.nn.Module):

    def __init__(self, idim, odim, hdim=192): # 128+64
    
        super().__init__()
        # help(torch.nn.Conv1d.__init__)
        # torch.nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # in_channels  输入通道数
        # out_channels 输出通道数
        # kernel_size  卷积核大小
        # padding      两边补零数
        # dilation     膨胀系数
        self.conv1 = torch.nn.Conv1d(in_channels=idim, out_channels=hdim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv1d(in_channels=hdim, out_channels=hdim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = torch.nn.Conv1d(in_channels=hdim, out_channels=hdim, kernel_size=3, padding=4, dilation=4)
        self.conv4 = torch.nn.Conv1d(in_channels=hdim, out_channels=hdim, kernel_size=3, padding=8, dilation=8)
        self.conv5 = torch.nn.Conv1d(in_channels=hdim, out_channels=odim, kernel_size=3, padding=16, dilation=16)

    def forward(self, x):

        o1 = self.conv1(x).tanh()
        o2 = self.conv2(o1).tanh()
        o3 = self.conv3(o2).tanh()
        o4 = self.conv4(o3).tanh()
        o5 = self.conv5(o4)
        o = o5.mean(dim=2)

        return o

if __name__ == "__main__":

    idim = 39 # 39维MFCC
    odim = 43 # 43位同学
    net = Net(idim, odim)
    print(net)

    x = torch.randn(10, idim, 100) # 10条语音，每条100帧，每帧idim维
    y = net(x)
    print(x.shape, y.shape)
