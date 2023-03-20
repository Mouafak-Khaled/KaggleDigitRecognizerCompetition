import torch
import torch.nn as nn

class DigitRecognizer(nn.Module):
    
    def __init__(self, in_channels, num_classes, bias=False):
        super(DigitRecognizer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bias = bias
        
        self.relu = nn.ReLU()
        
        self.conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, bias=self.bias)
        self.bn = nn.BatchNorm2d(32)
        self.mb = nn.MaxPool2d(kernel_size=3, stride=(2,2))

        self.flatten = nn.Flatten()
        
        self.cl = nn.Linear(4608, 2048, bias=self.bias)
        self.ln1 = torch.nn.LayerNorm(2048)
        
        self.cl2 = nn.Linear(2048, self.num_classes)
    
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1) 

        
    def forward(self, x):
        x = self.relu(self.mb(self.bn(self.conv(x))))
        x = self.flatten(x)
        x = self.relu(self.ln1(self.cl(x)))
        x = self.cl2(x)
        return x