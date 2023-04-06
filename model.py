import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.relu(self.bn(self.conv2(out)))
        return (x + out)


class ModelLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, pool=False, dropout =0.0, bias=False):
        super(ModelLayer, self).__init__()
        
        self.pool = pool
        self.dropout= dropout
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.mp = None
        self.dropoutLayer = None
        self.init_weights()
    
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
    
    
    def forward(self, x):
        
        x = self.bn(self.conv(x))
        
        if self.pool is True:
            self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            x = self.mp(x)
            
        x = self.relu(x)
        
        if self.dropout > 0.0:
            self.dropoutLayer = nn.Dropout2d(p=self.dropout)
            x = self.dropoutLayer(x)
        
        return x
        
        
class DigitRecognizer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_classes, dropout=0.0, bias=False):
        super(DigitRecognizer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.bias = bias
        self.dropout = dropout
        self.model_list = nn.ModuleList()

        for i in range(5):
            if i % 2 == 0:
                self.pool=False
            else:
                self.pool=True
        
        
            self.model_list.append(ModelLayer(self.in_channels,
                                          self.out_channels,
                                          bias=self.bias,
                                          pool=self.pool,
                                          dropout=self.dropout))
            
            # self.model_list.append(ResidualBlock(in_channels=self.out_channels,
            #                                      out_channels=self.out_channels,
            #                                      bias=self.bias))

            self.in_channels = self.out_channels
            if i == 3:
                self.out_channels = self.num_classes
            else:
                self.out_channels = self.out_channels * 2
        

        
        self.model_list.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()))

    
    def init_weights(self, m):
    
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
    
        
    def forward(self, x):
        
        for seq in self.model_list:
            # seq.apply(self.init_weights)
            x = seq(x)
        return x