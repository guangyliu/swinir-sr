# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import mmsr.models.archs.arch_util as arch_util

torch.set_printoptions(profile="full")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, nf=180, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH, i.e., (2*window_size-1*2*window_size-1, numHeads)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # torch.Size([25, 64, 180]) -> torch.Size([25, 64, 180*3])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # numWindows*B, window_size * window_size, C. i.e., torch.Size([759, 64, 60])
        # print('x: ', x.shape)   #x:  torch.Size([25, 64, 180])
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                         4)  # 3, numWindows*B, num_heads, window_size*window_size, c//num_heads
        # print('qkv: ', qkv.shape)  # qkv:  torch.Size([3, 25, 2, 64, 90])
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print(self.scale) #0.10540925533894598
        q = q * self.scale

        attn = (q @ k.transpose(-2,
                                -1))  # Matrix multiplication, torch.Size([759, 6, 64, 10]) @ torch.Size([759, 6, 10, 64]) -> torch.Size([759, 6, 64, 64])
        # attn shape: (numWindows*B, num_heads, window_size*window_size, window_size*window_size)

        # print(self.relative_position_index.view(-1).shape)  # torch.Size([64, 64]) -> torch.Size([4096])
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # torch.Size([759, 64, 60])

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


'''
Ref Window Attention by zwb
'''


class RefWindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH, i.e., (2*window_size-1*2*window_size-1, numHeads)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #TODO check whether this implement of qkv functions is correct
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lr_up_windows, ref_down_up_windows, ref_windows, mask=None):
        """
        Args:
            input: lr_up_windows, ref_down_up_windows, ref_windows. input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = lr_up_windows.shape  # numWindows*B, window_size * window_size, C. i.e., torch.Size([759, 64, 60])
        # print('lr_up_windows shape: ', lr_up_windows.shape)   torch.Size([400, 64, 180])
        # print('q(lr_up_windows) shape: ', self.q(lr_up_windows).shape)   torch.Size([400, 64, 180])
        q = self.q(lr_up_windows).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                 4)  # TODO check whether have this 1 dim
        k = self.k(ref_down_up_windows).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(ref_windows).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, numWindows*B, num_heads, window_size*window_size, c//num_heads
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        attn = (q @ k.transpose(-2,
                                -1))  # Matrix multiplication, torch.Size([759, 6, 64, 10]) @ torch.Size([759, 6, 10, 64]) -> torch.Size([759, 6, 64, 64])
        # attn shape: (numWindows*B, num_heads, window_size*window_size, window_size*window_size)

        # print(self.relative_position_index.view(-1).shape)  # torch.Size([64, 64]) -> torch.Size([4096])
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # torch.Size([759, 64, 60])

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


'''
RefSwinTransformerBlock by zwb
'''


class RefSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

         represents one STL

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = RefWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))  # class slice(start, stop[, step])
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # print(img_mask.shape) #torch.Size([1, 264, 184, 1])
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        # print(mask_windows.shape)  #torch.Size([759, 8, 8, 1])
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # nW, window_size*window_size
        # print(mask_windows.shape) #torch.Size([759, 64])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # print(mask_windows.unsqueeze(1).shape, (mask_windows.unsqueeze(2).shape)) # torch.Size([759, 1, 64]) torch.Size([759, 64, 1])
        # print(attn_mask.shape) # torch.Size([759, 64, 64])
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, lr_up, ref_down_up, ref, lr_up_size):
        '''
        input: lr_up, ref_down_up, ref, lr_up_size
        '''
        H, W = lr_up_size  # 264 184
        B, L, C = lr_up.shape  # 1 48576 60
        assert L == H * W, "input feature has wrong size"

        shortcut = lr_up
        lr_up = self.norm1(lr_up).view(B, H, W, C)
        ref_down_up = self.norm1(ref_down_up).view(B, H, W, C)
        ref = self.norm1(ref).view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_lr_up = torch.roll(lr_up, shifts=(-self.shift_size, -self.shift_size),
                                       dims=(1, 2))  # torch.Size([1, 264, 184, 60])
            shifted_ref_down_up = torch.roll(ref_down_up, shifts=(-self.shift_size, -self.shift_size),
                                             dims=(1, 2))
            shifted_ref = torch.roll(ref, shifts=(-self.shift_size, -self.shift_size),
                                     dims=(1, 2))

        else:
            shifted_lr_up = lr_up
            shifted_ref_down_up = ref_down_up
            shifted_ref = ref

            # partition windows
        lr_up_windows = window_partition(shifted_lr_up, self.window_size)  # nW*B, window_size, window_size, C
        lr_up_windows = lr_up_windows.view(-1, self.window_size * self.window_size,
                                           C)  # nW*B, window_size*window_size, C. torch.Size([759, 64, 60])
        ref_down_up_windows = window_partition(shifted_ref_down_up, self.window_size)
        ref_down_up_windows = ref_down_up_windows.view(-1, self.window_size * self.window_size, C)
        ref_windows = window_partition(shifted_ref, self.window_size)
        ref_windows = ref_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == lr_up_size:
            attn_windows = self.attn(lr_up_windows, ref_down_up_windows, ref_windows,
                                     mask=self.attn_mask)  # nW*B, window_size*window_size, C. perform self-attention here to update features
        else:
            attn_windows = self.attn(lr_up_windows, ref_down_up_windows, ref_windows,
                                     mask=self.calculate_mask(lr_up_size).to(lr_up.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

         represents one STL

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))  # class slice(start, stop[, step])
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # print(img_mask.shape) #torch.Size([1, 264, 184, 1])
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        # print(mask_windows.shape)  #torch.Size([759, 8, 8, 1])
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # nW, window_size*window_size
        # print(mask_windows.shape) #torch.Size([759, 64])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # print(mask_windows.unsqueeze(1).shape, (mask_windows.unsqueeze(2).shape)) # torch.Size([759, 1, 64]) torch.Size([759, 64, 1])
        # print(attn_mask.shape) # torch.Size([759, 64, 64])
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size  # 264 184
        B, L, C = x.shape  # 1 48576 60
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))  # torch.Size([1, 160, 160, 180])
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C. torch.Size([759, 64, 60])

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows,
                                     mask=self.attn_mask)  # nW*B, window_size*window_size, C. perform self-attention here to update features
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])  # blocks == one RSTB

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


'''RefBasicLayer by zwb'''


class RefBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            RefSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
            for i in range(depth)])  # blocks == one RSTB

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, lr_up, ref_down_up, ref, lr_up_size):
        '''
        input: lr_up, ref_down_up, ref, lr_up_size
        '''
        for blk in self.blocks:
            if self.use_checkpoint:
                lr_up = checkpoint.checkpoint(blk, lr_up, ref_down_up, ref)
            else:
                lr_up = blk(lr_up, ref_down_up, ref, lr_up_size)  # update lr_up
        if self.downsample is not None:
            lr_up = self.downsample(lr_up)
        return lr_up

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


'''
Reference based RSTB by zwb
'''


class Ref_RSTB(nn.Module):
    """Reference based Residual Swin Transformer Block (Ref_RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):  # TODO change the img_size

        super(Ref_RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.ref_residual_group = RefBasicLayer(dim=dim,
                                                input_resolution=input_resolution,
                                                depth=depth,
                                                num_heads=num_heads,
                                                window_size=window_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=drop_path,
                                                norm_layer=norm_layer,
                                                downsample=downsample,
                                                use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, lr_up, ref_down_up, ref, lr_up_size):
        '''
        input: lr_up, ref_down_up, ref, lr_up_size
        '''
        return self.patch_embed(
            self.conv(self.patch_unembed(self.ref_residual_group(lr_up, ref_down_up, ref, lr_up_size),
                                         lr_up_size))) + lr_up  # TODO whether plus lr_up in the end?

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=160, patch_size=1, in_chans=3, embed_dim=180, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # print('before PatchEmbed 1: ', x.shape)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        # print('before PatchEmbed 2 : ', x.shape)
        if self.norm is not None:
            x = self.norm(x)
        # print('After PatchEmbed: ', x.shape)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinRefSR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 160
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each Swin Transformer layer. [6,6,6,6,6,6]
        num_heads (tuple(int)): Number of attention heads in different layers. [6,6,6,6,6,6]
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4 # TODO Try?
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=160, patch_size=1, in_chans=3,
                 embed_dim=180, depths=[3, 3, 3, 3], num_heads=[3, 3, 3, 3],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='pixelshuffle', resi_connection='1conv',
                 **kwargs):
        super(SwinRefSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        n_blocks = 16  # For aggregation
        self.img_range = torch.as_tensor(img_range).cuda()
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1).cuda()
        else:
            self.mean = torch.zeros(1, 1, 1, 1).cuda()
        self.upscale = upscale
        self.upsampler = upsampler

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first_backbone = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)  # 3 --> 180
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)  # 3 --> 180
        self.content_extractor = ContentExtractor(in_nc=3, nf=embed_dim, n_blocks=n_blocks)  # 3 --> 180

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)  # number of RSTB
        self.embed_dim = embed_dim  # 180
        self.ape = ape  # True: add absolute position embedding to the patch embedding
        self.patch_norm = patch_norm  # True: add normalization after patch embedding
        self.num_features = embed_dim  # 180
        self.mlp_ratio = mlp_ratio  # 4

        ##### For backone model with 40x40 input images ######
        self.patch_embed_small = PatchEmbed(
            img_size=img_size // upscale, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches_small = self.patch_embed_small.num_patches
        patches_resolution_small = self.patch_embed_small.patches_resolution

        # self.patch_unembed_small = PatchUnEmbed(
        #     img_size=img_size // upscale, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        #### End For backone model######

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_small = nn.Parameter(torch.zeros(1, num_patches_small, embed_dim))
            trunc_normal_(self.absolute_pos_embed_small, std=.02)

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB) For backbone model
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # how many layers of RSTB 4
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution_small[0],
                                           patches_resolution_small[1]),
                         depth=depths[i_layer],  # depths: [3,3,3,3]
                         num_heads=num_heads[i_layer],  # num_heads: [3,3,3,3]
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size // 4,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)

        '''
        Cross Attetion Module
        New kqv by zwb, For similiar feature searching module
        '''
        # build Residual Swin Transformer blocks (RSTB)
        self.ref_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):  # how many layers of RSTB
            ref_layer = Ref_RSTB(dim=embed_dim,
                                 input_resolution=(patches_resolution[0],
                                                   patches_resolution[1]),
                                 depth=depths[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                 # no impact on SR results
                                 norm_layer=norm_layer,
                                 downsample=None,
                                 use_checkpoint=use_checkpoint,
                                 img_size=img_size,
                                 patch_size=patch_size,
                                 resi_connection=resi_connection
                                 )
            self.ref_layers.append(ref_layer)

        self.norm = norm_layer(self.num_features)  # 180

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body_backbone = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            '''
            self-attetion backbone + cross attention searching module
            '''
            self.conv_before_upsample_backbone = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                               nn.LeakyReLU(inplace=True))
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(
                                                          inplace=True))  # channel 180 --> 64, remain the same shape
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Sequential(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv2d(num_feat, num_out_ch, 3, 1,
                                                     1))  # 64*2->64->3 #TODO change the LeakyRelU, and check whether it is reasonable
            # self.conv_last = nn.Conv2d(num_feat * 2, num_out_ch, 3, 1, 1)

            # Aggregation module from C2-matching
            self.head_large = nn.Sequential(
                nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True))
            self.body_large = arch_util.make_layer(
                arch_util.ResidualBlockNoBN, n_blocks, nf=num_feat)
            self.tail_large = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // 2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(num_feat // 2, 3, kernel_size=3, stride=1, padding=1))


            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        elif self.upsampler == 'mypixelshuffle':
            '''
            Without backbone, Swin-Transformer directly generate the sr output
            '''
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))  # 180 --> 64
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Sequential(nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))  # *2 added by me

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def backbone_forward_features(self, x):
        '''
        input x: torch.Size([1, 180, 40, 40])
        '''
        x_size = (x.shape[2], x.shape[3])  # (40, 40)
        x = self.patch_embed_small(x)  # torch.Size([1, 180, 40, 40]) -> torch.Size([1, 1600, 180])

        if self.ape:
            x += self.absolute_pos_embed_small  # torch.Size([1, 1600, 180])
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)  # delete _small

        return x

    def ref_forward_features(self, lr_up, ref_down_up, ref):
        '''
        input: lr_up, ref_down_up, ref
        '''
        lr_up_size = (lr_up.shape[2], lr_up.shape[3])  # [160, 160]

        lr_up = self.patch_embed(lr_up)  # torch.Size([1, 180, 160, 160]) -> torch.Size([1, 25600, 180])
        ref_down_up = self.patch_embed(ref_down_up)
        ref = self.patch_embed(ref)

        if self.ape:
            lr_up += self.absolute_pos_embed
            ref_down_up += self.absolute_pos_embed
            ref += self.absolute_pos_embed
        lr_up = self.pos_drop(lr_up)
        ref_down_up = self.pos_drop(ref_down_up)
        ref = self.pos_drop(ref)

        for ref_layer in self.ref_layers:
            lr_up = ref_layer(lr_up, ref_down_up, ref, lr_up_size)  # used to write wrongly: ref_features -> lr_up

        lr_up = self.norm(lr_up)  # B L C
        lr_up = self.patch_unembed(lr_up, lr_up_size)

        return lr_up

    def aggregate_features(self, backbone_feature, ref_feature):
        '''
        input: lr feature extracted by backbone, and ref_feature generated by cross-attention net
        '''

        concat_features = torch.cat([backbone_feature, ref_feature],
                                    1)  # check the channel #torch.Size([B, 64*2, 160, 160])
        concat_features = self.head_large(concat_features)
        concat_features = self.body_large(concat_features) + backbone_feature
        backbone_feature = self.tail_large(concat_features)
        return backbone_feature

    def forward(self, lr, ref, ref_down_up, lr_up):
        '''
        input: LR, ref, ref_down_up, lr_up
        '''
        self.mean = self.mean.type_as(lr)
        lr = (lr - self.mean) * self.img_range  # 1
        ref = (ref - self.mean) * self.img_range
        ref_down_up = (ref_down_up - self.mean) * self.img_range
        lr_up = (lr_up - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            '''
            SwinIR backbone
            '''
            base = F.interpolate(lr, None, 4, 'bilinear', False)  # x4
            lr = self.conv_first(lr)  # torch.Size([B, 180, 40, 40])
            #lr = self.content_extractor(lr)  # dim: 3 -> 180
            lr = self.conv_after_body(self.backbone_forward_features(lr)) + lr  # torch.Size([1, 60, 264, 184]) dim: 180 -> 180
            lr = self.conv_before_upsample(lr)  # torch.Size([B, 64, 40, 40]) dim: embed_dim -> num_feat
            backbone_feature = self.upsample(
                lr)  # torch.Size([B, 64, 160, 160]) to be concatenated with lr_up_feature then output the SR result
            '''
            reference matching via cross attention.
            '''
            lr_up = self.conv_first(lr_up)  # For Q
            ref_down_up = self.conv_first(ref_down_up)  # For K
            ref = self.conv_first(ref)  # For V

            # lr_up = self.content_extractor(lr_up)  # For Q
            # ref_down_up = self.content_extractor(ref_down_up)  # For K
            # ref = self.content_extractor(ref)  # For V

            lr_up = self.conv_after_body(self.ref_forward_features(lr_up, ref_down_up, ref)) + lr_up
            lr_up = self.conv_before_upsample(lr_up)  # torch.Size([2, 64, 160, 160])
            '''
            final aggregation. 
            '''
            # TODO: To improve the aggregation module

            # sr = self.conv_last(concat_features)  # torch.Size([1, 3, 1056, 736])

            #Concate:
            sr = self.aggregate_features(backbone_feature,
                                         lr_up) + base  # TODO: Try to add the backbone_feature, or base, or delete?

            # #Add:
            # sr = backbone_feature + lr_up + backbone_feature  # TODO: Try to add the backbone_feature, or base

            # sr = self.tail_large(sr)

        elif self.upsampler == 'pixelshuffle_not_share_weights':
            # for classical SR
            '''
            SwinIR backbone
            '''
            lr = self.conv_first_backbone(lr)  # torch.Size([1, 60, 264, 184])
            lr = self.conv_after_body_backbone(self.backbone_forward_features(lr)) + lr  # torch.Size([1, 60, 264, 184])
            lr = self.conv_before_upsample_backbone(lr)  # torch.Size([1, 64, 264, 184])
            backbone_feature = self.upsample(
                lr)  # torch.Size([B, 64, 160, 160]) to be concatenated with lr_up_feature then output the SR result
            '''
            reference matching via cross attention.
            '''
            lr_up = self.conv_first(lr_up)  # For Q
            ref_down_up = self.conv_first(ref_down_up)  # For K
            ref = self.conv_first(ref)  # For V

            lr_up = self.conv_after_body(self.ref_forward_features(lr_up, ref_down_up, ref)) + lr_up
            lr_up = self.conv_before_upsample(lr_up)  # torch.Size([2, 64, 160, 160])
            '''
            final aggregation. 
            '''
            # TODO To improve the aggregation module
            concat_features = torch.cat([backbone_feature, lr_up],
                                        1)  # check the channel #torch.Size([B, 64*2, 160, 160])

            sr = self.conv_last(concat_features)  # torch.Size([1, 3, 1056, 736])

        elif self.upsampler == 'mypixelshuffle':
            '''
            cross-attention module directly output the sr result
            '''
            lr_up = self.conv_first(lr_up)  # For Q
            ref_down_up = self.conv_first(ref_down_up)  # For K
            ref = self.conv_first(ref)  # For V

            lr_up = self.conv_after_body(self.ref_forward_features(lr_up, ref_down_up, ref)) + lr_up
            lr_up = self.conv_before_upsample(lr_up)  # torch.Size([2, 64, 160, 160])
            sr = self.conv_last(lr_up)  # torch.Size([1, 3, 1056, 736])

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            lr = self.conv_first(lr)
            lr = self.conv_after_body(self.forward_features(lr)) + lr
            sr = self.upsample(lr)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(lr)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            sr = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(lr)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            sr = lr + self.conv_last(res)
        sr = sr / self.img_range + self.mean

        return sr

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinRefSR(upscale=4, img_size=(height, width),
                      window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                      embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
    # print(model)
    # print(height, width, model.flops() / 1e9)
    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
