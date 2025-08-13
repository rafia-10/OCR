import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        assert imgH % 16 == 0, 'imgH must be multiple of 16'

        self.cnn = nn.Sequential(
            nn.Conv2d(nc,64,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
            nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True)
        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True),
            nn.LSTM(nh*2, nh, bidirectional=True)
        )
        self.fc = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b,c,h,w = conv.size()
        conv = conv.squeeze(2).permute(2,0,1)
        recurrent,_ = self.rnn(conv)
        output = self.fc(recurrent)
        return output.permute(1,0,2)
