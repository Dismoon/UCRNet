import torch.nn as nn
import torch
import torch.nn.functional as F


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RDBs_1(nn.Module):
    def __init__(self, C, ks, G):
        super(RDBs_1, self).__init__()
        self.rdbs_1 = nn.ModuleList([])
        for j in range(1, C + 1):
            self.rdbs_1.append(nn.Conv3d(G * j, G, ks, padding=int((ks - 1) / 2)))

    def forward(self, x):
        for layers in self.rdbs_1:
            tmp = layers(x)
            tmp = F.relu(tmp, True)
            x = torch.cat([x, tmp], 1)

        return x

    def initialize_weight(self, w_mean, w_std):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') == 0:
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class RDBs(nn.Module):
    def __init__(self, C, D, ks, G):
        super(RDBs, self).__init__()
        self.rdbs = nn.ModuleList([])
        for i in range(1, D + 1):
            self.rdbs.append(RDBs_1(C, ks, G))
            self.rdbs.append(nn.Conv3d(G * (C + 1), G, 1))

    def forward(self, input):
        rdb_in = input
        rdb_concat = list()
        for i, layers in enumerate(self.rdbs):
            if i % 2 == 0:
                x = rdb_in
            x = layers(x)
            if str(layers).find('RDBs_1') != 0:
                rdb_in = torch.add(x, rdb_in)
                rdb_concat.append(rdb_in)

        return torch.cat(rdb_concat, 1)

    def initialize_weight(self, w_mean, w_std):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') == 0:
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class DRDN(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 D,
                 C,
                 G,
                 G0,
                 ks
                 ):
        super(DRDN, self).__init__()
        self.body = nn.ModuleDict({
            'conv1': nn.Conv3d(input_channel, G0, ks, padding=int((ks - 1) / 2)),
            'conv2': nn.Conv3d(G0, G, ks, padding=int((ks - 1) / 2)),
            'RDB': RDBs(C, D, ks, G),
            'conv3': nn.Conv3d(G * D, G0, 1, padding=0),
            'conv4': nn.Conv3d(G0, G, ks, padding=int((ks - 1) / 2)),
            'conv5': nn.Conv3d(G, output_channel, ks, padding=int((ks - 1) / 2))
        }
        )

    def forward(self, x):
        F_1 = self.body['conv1'](x)
        F0 = self.body['conv2'](F_1)
        FD = self.body['RDB'](F0)
        FGF1 = self.body['conv3'](FD)
        FGF2 = self.body['conv4'](FGF1)
        out = self.body['conv5'](FGF2)

        return out

    def initialize_weight(self, w_mean=0, w_std=0.01):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') == 0:
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
