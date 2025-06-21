import os,sys

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..block import *
from ..utils import *

class BilinearUp(nn.Module):
	def __init__(self, in_channels, out_channels, factor=(1,2,2)):
		super(BilinearUp, self).__init__()
		assert in_channels==out_channels
		self.groups = in_channels
		self.factor = factor
		self.kernel_size = [(2 * f) - (f % 2) for f in self.factor]
		self.padding = [int(math.ceil((f - 1) / 2.0)) for f in factor]
		self.init_weights()

	def init_weights(self):
		weight = torch.Tensor(self.groups, 1, *self.kernel_size)
		width = weight.size(-1)
		hight = weight.size(-2)
		assert width==hight
		f = float(math.ceil(width / 2.0))
		c = float(width - 1) / (2.0 * f)
		for w in range(width):
			for h in range(hight):
				weight[...,h,w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c))
		self.register_buffer('weight', weight) # fixed

	def forward(self, x):
		return F.conv_transpose3d(x, self.weight, stride=self.factor, padding=self.padding, groups=self.groups)

class ShapeAwareLayer(nn.Module):
    def __init__(self, in_channels):
        super(ShapeAwareLayer, self).__init__()
        # # 3D卷积，提取深度信息
        # self.conv3d = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        # # 2D卷积，提取平面内信息
        # self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3d = nn.Conv3d(in_channels, in_channels, kernel_size=(3,3,1), padding=(1,1,0))  # 各向异性 3D 卷积
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 用于 xy 平面的 2D 卷积

    def forward(self, x):
        x_3d = self.conv3d(x)  # 3D 卷积增强 z 方向特征
        x_2d = torch.mean(x_3d, dim=2)  # 平均池化 z 方向，保留 xy 平面
        x_2d = self.conv2d(x_2d)  # 2D 卷积
        x_2d_expanded = x_2d.unsqueeze(2).expand_as(x_3d)  # 将 xy 平面扩展回 z 方向
        return x_3d + x_2d_expanded  # 融合 3D 和 2D 特征


class CrossSliceAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, embed_dim=64):
        super(CrossSliceAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.query = nn.Linear(in_channels, embed_dim * num_heads)
        self.key = nn.Linear(in_channels, embed_dim * num_heads)
        self.value = nn.Linear(in_channels, embed_dim * num_heads)
        self.out = nn.Linear(embed_dim * num_heads, in_channels)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        x_reshaped = x.permute(0, 3, 4, 2, 1).reshape(-1, depth, channels)
        Q = self.query(x_reshaped).view(-1, depth, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)
        K = self.key(x_reshaped).view(-1, depth, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)
        V = self.value(x_reshaped).view(-1, depth, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(-1, depth, self.embed_dim * self.num_heads)
        output = self.out(attention_output)
        return output.view(batch_size, height, width, depth, channels).permute(0, 4, 3, 1, 2)
 

class resunet_csa_saa_v2(nn.Module):

    """Lightweight 3D U-net with residual blocks (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.
        
    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=3, filters=[28, 36, 48, 64, 80], pad_mode='rep', norm_mode='bn', act_mode='elu', 
                 do_embedding=True, head_depth=1, output_act='sigmoid'):
        super().__init__()

        self.depth = len(filters)-2
        self.do_embedding = do_embedding
        self.output_act = output_act # activation function for the output layer

        # encoding path
        if self.do_embedding: 
            num_out = filters[1]
            self.downE = nn.Sequential(
                # anisotropic embedding
                conv3d_norm_act(in_planes=in_channel, out_planes=filters[0], 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                # 2d residual module
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            )
        else:
            filters[0] = in_channel
            num_out = out_channel
        
        self.downC = nn.ModuleList(
            [nn.Sequential(
            conv3d_norm_act(in_planes=filters[x], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # pooling downsample
        # self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) for x in range(self.depth+1)])
        self.downS = nn.ModuleList(  # 用conv3d 下采样
            [conv3d_norm_act(in_planes=filters[x], out_planes=filters[x], kernel_size=(1,3,3), stride=(1, 2, 2), padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            for x in range(self.depth+1)])

        # center block
        self.center = nn.Sequential(conv3d_norm_act(in_planes=filters[-2], out_planes=filters[-1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[-1], filters[-1], projection=True)
        )
        self.middle = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x], out_planes=filters[x],
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth+1)])
            
        self.upC = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x+1], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[x+1], filters[x+1], projection=False)
            ) for x in range(self.depth)])

        if self.do_embedding: 
            # decoding path
            self.upE = nn.Sequential(
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                conv3d_norm_act(in_planes=filters[0], out_planes=out_channel, 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode)
            )
            # conv + upsample
            self.upS = nn.ModuleList([nn.Sequential(
                            conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                            BilinearUp(filters[x], filters[x], factor=(1,2,2))) for x in range(self.depth+1)])
        else:
            # new
            head_pred = [residual_block_3d(filters[1], filters[1], projection=False)
                                    for x in range(head_depth-1)] + \
                              [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)]
            self.upS = nn.ModuleList( [nn.Sequential(*head_pred)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                                     BilinearUp(filters[x], filters[x], factor=(1, 2, 2))) for x in range(1,self.depth+1)])
            """
            # old
            self.upS = nn.ModuleList( [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                        nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(1,self.depth+1)])
            """
        
        self.shape_aware_layers = nn.ModuleList([ShapeAwareLayer(filters[x+1]) for x in range(self.depth)])
        self.cross_slice_attention = nn.ModuleList([CrossSliceAttention(filters[x+1], num_heads=4, embed_dim=64) for x in range(self.depth)])

        #initialization
        ortho_init(self)

    def forward(self, x):
        # encoding path
        if self.do_embedding:
            z = self.downE(x) # torch.Size([4, 1, 32, 256, 256])
            x = self.downS[0](z) # downsample

        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            down_u[i] = self.shape_aware_layers[i](down_u[i])  # 形状感知层
            down_u[i] = self.cross_slice_attention[i](down_u[i])  # 跨切片注意力
            x = self.downS[i+1](down_u[i]) # downsample

        x = self.center(x)

        z = self.middle[0](z)
        layer = []


        for i in range(len(down_u)-1):
            layer.append(self.middle[i+1](down_u[i]))
            # print(i)

        # decoding path
        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i+1](x)
            x = self.shape_aware_layers[i](x)  # 添加形状感知层
            x = self.cross_slice_attention[i](x)  # 添加跨切片注意力
            x = self.upC[i](x)

        if self.do_embedding: 
            x = z + self.upS[0](x)
            x = self.upE(x)
        else:
            x = self.upS[0](x)

        x = get_functional_act(self.output_act)(x)
        return x


