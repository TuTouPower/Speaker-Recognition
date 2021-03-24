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
        self.BiGRU = torch.nn.GRU(input_size=idim, hidden_size=hdim, num_layers=3, batch_first=True, dropout=0.5, bidirectional = True)
        self.conv1 = torch.nn.Conv1d(in_channels=2*hdim, out_channels=hdim, kernel_size=3, padding=1, dilation=2)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=8)
        self.conv2 = torch.nn.Conv1d(in_channels=89, out_channels=hdim, kernel_size=3, padding=2, dilation=4)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=16)
        self.lstm = torch.nn.LSTM(input_size=94, hidden_size=hdim, num_layers=2, batch_first=True, dropout=0.5)
        self.linear = torch.nn.Linear(in_features=hdim, out_features=odim)

    def forward(self, x):

        o_ = x.permute(0,2,1).contiguous()               #  100  39  S T
        o_,x = self.BiGRU(o_)
        o_BiGru = o_.tanh()                 #  100  384 T S
        #print(o_BiGru.shape)            
        o_BiGru = o_BiGru.permute(0,2,1).contiguous()    #  384  100 T S
        #print(o_BiGru.shape)

        o_conv1 = self.conv1(o_BiGru).tanh()#  192  98  T S
        #print(o_conv1.shape)
        o_conv1 = o_conv1.permute(0,2,1).contiguous()    #  98   192 S T
        o_pool1 = self.pool1(o_conv1)       #  98   89  S T
        #print(o_pool1.shape)
        o_pool1 = o_pool1.permute(0,2,1).contiguous()    #  89   98  T S

        o_conv2 = self.conv2(o_pool1).tanh()#  192  94  T S
        #print(o_conv2.shape)
        o_conv2 = o_conv2.permute(0,2,1).contiguous()    #  94   192 S T
        o_pool2 = self.pool2(o_conv2)       #  94   81  S T
        #print(o_pool2.shape)
        o_pool2 = o_pool2.permute(2,0,1).contiguous()    #  81   94  T S
        
        o_lstm,x = self.lstm(o_pool2)
        o = self.linear(o_lstm[-1])

        return o

if __name__ == "__main__":

    idim = 39 # 39维MFCC
    odim = 43 # 43位同学
    net = Net(idim, odim)
    print(net)

    x = torch.randn(43, idim, 100) # 10条语音，每条100帧，每帧idim维
    y = net(x)
    print(x.shape, y.shape)
