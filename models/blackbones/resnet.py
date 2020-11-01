import torch.nn as nn
import torch.nn.functional as F
import torch
class resnet_block(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(resnet_block, self).__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		if self.dim_in == self.dim_out:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.bn_1 = nn.BatchNorm2d(self.dim_out)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.bn_2 = nn.BatchNorm2d(self.dim_out)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
		else:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
			self.bn_1 = nn.BatchNorm2d(self.dim_out)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.bn_2 = nn.BatchNorm2d(self.dim_out)
			self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
			self.bn_s = nn.BatchNorm2d(self.dim_out)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
			nn.init.xavier_uniform_(self.conv_s.weight)

	def forward(self, input, is_training=False):
		if self.dim_in == self.dim_out:
			output = self.bn_1(self.conv_1(input))
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
			output = self.bn_2(self.conv_2(output))
			output = output+input
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
		else:
			output = self.bn_1(self.conv_1(input))
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
			output = self.bn_2(self.conv_2(output))
			input_ = self.bn_s(self.conv_s(input))
			output = output+input_
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
		return output

class img_encoder(nn.Module):
	def __init__(self, img_ef_dim, z_dim):
		super(img_encoder, self).__init__()
		self.img_ef_dim = img_ef_dim
		self.z_dim = z_dim
		self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
		self.bn_0 = nn.BatchNorm2d(self.img_ef_dim)
		self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
		self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
		self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim*2)
		self.res_4 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*2)
		self.res_5 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*4)
		self.res_6 = resnet_block(self.img_ef_dim*4, self.img_ef_dim*4)
		self.res_7 = resnet_block(self.img_ef_dim*4, self.img_ef_dim*8)
		self.res_8 = resnet_block(self.img_ef_dim*8, self.img_ef_dim*8)
		self.conv_9 = nn.Conv2d(self.img_ef_dim*8, self.img_ef_dim*8, 4, stride=2, padding=1, bias=False)
		self.bn_9 = nn.BatchNorm2d(self.img_ef_dim*8)
		self.conv_10 = nn.Conv2d(self.img_ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_0.weight)
		nn.init.xavier_uniform_(self.conv_9.weight)
		nn.init.xavier_uniform_(self.conv_10.weight)

	def forward(self, view, is_training=False):
		layer_0 = self.bn_0(self.conv_0(1-view))
		layer_0 = F.leaky_relu(layer_0, negative_slope=0.02, inplace=True)

		layer_1 = self.res_1(layer_0, is_training=is_training)
		layer_2 = self.res_2(layer_1, is_training=is_training)

		layer_3 = self.res_3(layer_2, is_training=is_training)
		layer_4 = self.res_4(layer_3, is_training=is_training)

		layer_5 = self.res_5(layer_4, is_training=is_training)
		layer_6 = self.res_6(layer_5, is_training=is_training)

		layer_7 = self.res_7(layer_6, is_training=is_training)
		layer_8 = self.res_8(layer_7, is_training=is_training)

		layer_9 = self.bn_9(self.conv_9(layer_8))
		layer_9 = F.leaky_relu(layer_9, negative_slope=0.02, inplace=True)

		layer_10 = self.conv_10(layer_9)
		layer_10 = layer_10.view(-1,self.z_dim)
		layer_10 = torch.sigmoid(layer_10)

		return layer_10
