import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim, gf_split):
        '''
        :param z_dim: feature [1, 256]
        :param point_dim: coordinate of n point [n, 3]
        :param gf_dim:
        :param gf_split: the part number
        '''

        super(generator, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.linear_1 = nn.Linear(self.z_dim + self.point_dim, self.gf_dim * 4, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim * 4, self.gf_dim, bias=True)
        l3_layer_weights = torch.zeros((self.gf_dim, self.gf_split), requires_grad=True)
        self.part_layer_weights = nn.Parameter(l3_layer_weights)

        # self.linear_3 = nn.Linear(self.gf_split, 1, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(l3_layer_weights, mean=0.0, std=0.02)



    def forward(self, points, z, is_training=False):

        zs1 = z.view(-1, 1, self.z_dim)
        zs = zs1.repeat(1, points.shape[1], 1)
        pointz = torch.cat([points, zs], len(points.shape) - 1)

        d_1 = self.linear_1(pointz)
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.linear_2(d_1)
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = torch.matmul(d_2, self.part_layer_weights)
        d_3 = torch.sigmoid(d_3)
        d_3_max = torch.max(d_3, 2, keepdim=True)

        return d_3, d_3_max[0]