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


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim=256, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 使用 1x1 卷积代替线性层，将输入的通道数转换为 embed_dim
        self.embedding_layer = nn.Conv3d(in_channels, embed_dim, kernel_size=1)

        # 初始化注意力层和前馈层
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # 输入形状: (batch_size, channels, depth, height, width)
        batch_size, channels, depth, height, width = x.shape

        # 用 1x1 卷积将通道数转换为 embed_dim
        x = self.embedding_layer(x)  # 输出形状为 (batch_size, embed_dim, depth, height, width)

        # 将输入重塑为 (batch_size * height * width, depth, embed_dim)
        x = x.view(batch_size * height * width, depth, self.embed_dim).permute(1, 0, 2)

        # 通过 Multihead Attention 层
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # 残差连接
        x = self.norm1(x)

        # 通过前馈神经网络 (FFN) 和残差连接
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        # 恢复为原始形状 (batch_size, embed_dim, depth, height, width)
        x = x.permute(1, 0, 2).contiguous().view(batch_size, self.embed_dim, depth, height, width)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Conv3d(dim, dim, kernel_size=1)
        self.key = nn.Conv3d(context_dim, dim, kernel_size=1)  # 调整上一层通道数到当前层通道数
        self.value = nn.Conv3d(context_dim, dim, kernel_size=1)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        if context.shape[2:] != x.shape[2:]:  # 仅在空间维度不一致时调整
            context = F.interpolate(context, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        # 查询和键值特征图
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        # 计算注意力
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = F.softmax(attention, dim=-1)
        out = (attention @ value).reshape_as(x)
        return out + x  # 残差连接


class BoundaryAwareAttention(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryAwareAttention, self).__init__()
        self.attention = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        attention_weights = torch.sigmoid(self.attention(x))
        return x * attention_weights  # 对边界位置加权

        


class CrossSliceAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, embed_dim=64):
        """
        初始化 Cross-Slice Attention 模块
        Args:
            in_channels (int): 输入通道数，即来自编码器的特征通道数
            num_heads (int): 注意力头的数量
            embed_dim (int): 每个注意力头的嵌入维度
        """
        super(CrossSliceAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # 使用线性层生成 Query, Key, Value
        self.query = nn.Linear(in_channels, embed_dim * num_heads)
        self.key = nn.Linear(in_channels, embed_dim * num_heads)
        self.value = nn.Linear(in_channels, embed_dim * num_heads)
        
        # 输出线性层
        self.out = nn.Linear(embed_dim * num_heads, in_channels)

    def forward(self, x):
        """
        x: 输入特征，形状为 (batch_size, in_channels, depth, height, width)
        """
        # 获取输入特征维度
        batch_size, channels, depth, height, width = x.shape

        # 将输入重塑为 (batch_size * height * width, depth, in_channels)
        x_view = x.permute(0, 3, 4, 2, 1).reshape(-1, depth, channels)

        # 计算 Query, Key, Value，输出维度为 (batch_size * height * width, depth, embed_dim * num_heads)
        Q = self.query(x_view)
        K = self.key(x_view)
        V = self.value(x_view)

        # 将 Q, K, V 进行多头分割，维度为 (batch_size * height * width, num_heads, depth, embed_dim)
        Q = Q.reshape(-1, depth, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)
        K = K.reshape(-1, depth, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)
        V = V.reshape(-1, depth, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)

        # 计算注意力得分，形状为 (batch_size * height * width, num_heads, depth, depth)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # 将注意力得分应用于 V，形状为 (batch_size * height * width, num_heads, depth, embed_dim)
        attention_output = torch.matmul(attention_probs, V)

        # 将多头结果合并回去，形状为 (batch_size * height * width, depth, embed_dim * num_heads)
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(-1, depth, self.embed_dim * self.num_heads)

        # 使用输出层，将形状变回 (batch_size * height * width, depth, in_channels)
        output = self.out(attention_output)

        # 重塑回原始输入的形状 (batch_size, in_channels, depth, height, width)
        output = output.view(batch_size, height, width, depth, channels).permute(0, 4, 3, 1, 2)

        return output

class unet_res_ba(nn.Module):
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

        self.depth = len(filters)-2         # 3
        self.do_embedding = do_embedding    # True
        self.output_act = output_act # activation function for the output layer  sigmoid

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

        
        # Cross-Slice Attention 模块列表

        self.cross_slice_attention = nn.ModuleList([CrossSliceAttention(filters[x+1], num_heads=4, embed_dim=64) for x in range(self.depth)])

        self.boundary_attention = nn.ModuleList([BoundaryAwareAttention(filters[x+1]) for x in range(self.depth)])


        #initialization
        ortho_init(self)


    def forward(self, x):
        # encoding path
        if self.do_embedding:
            z = self.downE(x) # torch.Size([1, 28, 16, 256, 256])
            x = self.downS[0](z) # downsample   torch.Size([1, 28, 16, 128, 128])

        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            # down_u[i] = self.cross_slice_attention[i](down_u[i])  # 跨切片注意力
            x = self.downS[i+1](down_u[i])

        x = self.center(x)
        z = self.middle[0](z)
        layer = []

        for i in range(len(down_u)-1):
            layer.append(self.middle[i+1](down_u[i]))
            # print(i)


        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i+1](x)
            x = self.boundary_attention[i](x)  # 加入边界注意力
            x = self.upC[i](x)


        if self.do_embedding: 
            x = z + self.upS[0](x)
            x = self.upE(x)
        else:
            x = self.upS[0](x)

        x = get_functional_act(self.output_act)(x)
        return x