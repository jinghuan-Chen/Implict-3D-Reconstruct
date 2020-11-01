import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.z_dim, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs, is_training=False):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.z_dim)
        d_5 = torch.sigmoid(d_5)

        return d_5