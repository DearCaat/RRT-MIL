import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, reduce


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PPEG(nn.Module):
    def __init__(self, dim=512, k=7, conv_1d=False, bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k, 1), 1, (k // 2, 0), groups=dim, bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5, 1), 1, (5 // 2, 0), groups=dim, bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3, 1), 1, (3 // 2, 0), groups=dim, bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))

        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        # 避免最后补出来的特征图小于卷积核，这里只能用zero padding
        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x = torch.cat([x, torch.zeros((B, zero_pad, C), device=x.device)], dim=1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length > 0:
            x = x[:, :-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class PEG(nn.Module):
    def __init__(self, dim=512, k=7, bias=True, conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k, 1), 1, (k // 2, 0), groups=dim, bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length > 0:
            x = x[:, :-add_length]

        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Attention(nn.Module):
    def __init__(self, input_dim=512, act="relu", bias=False, dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D, bias=bias)]

        if act == "gelu":
            self.attention += [nn.GELU()]
        elif act == "relu":
            self.attention += [nn.ReLU()]
        elif act == "tanh":
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K, bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self, x, no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        else:
            return x, A


class DAttention(nn.Module):
    def __init__(self, input_dim=512, act="relu", mask_ratio=0.0, gated=False, bias=False, dropout=False):
        super(DAttention, self).__init__()
        self.attention = Attention(input_dim, act, bias, dropout)

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, no_norm=False, mask_enable=False, classifier=None, **kwags):
        x, attn = self.attention(x, no_norm)

        if return_attn:
            return x.squeeze(1), attn.squeeze(1)
        else:
            return x.squeeze(1)


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


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_landmarks=256, pinv_iterations=6, residual=True, residual_conv_kernel=33, eps=1e-8, dropout=0.0):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = math.ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = "... i d, ... j d -> ... i j"
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:, :, 0].unsqueeze(-2) @ attn2
            attn1 = attn1 @ attn3

            return out, attn1[:, :, 0, -n + 1 :]

        return out


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(
        self, dim, head_dim=None, window_size=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, conv=True, epeg_k=15, conv_2d=False, conv_bias=True, conv_type="attn"
    ):
        super().__init__()
        self.dim = dim
        self.window_size = [window_size, window_size] if window_size is not None else None  # Wh, Ww
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        if window_size is not None:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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
            trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv_2d = conv_2d
        self.conv_type = conv_type
        if conv:
            kernel_size = epeg_k
            padding = kernel_size // 2

            if conv_2d:
                if conv_type == "attn":
                    self.pe = nn.Conv2d(num_heads, num_heads, kernel_size, padding=padding, groups=num_heads, bias=conv_bias)
            else:
                if conv_type == "attn":
                    self.pe = nn.Conv2d(num_heads, num_heads, (kernel_size, 1), padding=(padding, 0), groups=num_heads, bias=conv_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, return_attn=False, no_pe=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        L = int(np.ceil(np.sqrt(N)))
        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.pe is not None and not no_pe:
            if self.conv_type == "attn":
                pe = self.pe(attn)
                attn = attn + pe

        if self.window_size is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )  # Wh*Ww,Wh*Ww,nH
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

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn
        else:
            return x


class RRTAttntion(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution=None,
        head_dim=None,
        num_heads=8,
        window_size=0,
        shift_size=False,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        window_num=8,
        conv=False,
        rpe=False,
        min_win_num=0,
        min_win_ratio=0.0,
        glob_pe="none",
        win_attn="native",
        moe_enable=False,
        fl=False,
        crmsa_k=1,
        wandb=None,
        l1_shortcut=True,
        no_weight_to_all=False,
        minmax_weight=True,
        moe_mask_diag=False,
        mask_diag=False,
        moe_mlp=False,
        **kawrgs
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size if window_size > 0 else None
        self.shift_size = shift_size
        self.window_num = window_num
        self.min_win_num = min_win_num
        self.min_win_ratio = min_win_ratio
        self.rpe = rpe

        self.pe = nn.Identity()

        if self.window_size is not None:
            self.window_num = None
        self.fused_window_process = False

        if win_attn == "native":
            self.attn = WindowAttention(
                dim,
                head_dim=head_dim,
                num_heads=num_heads,
                window_size=self.window_size if rpe else None,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                conv=conv,
                **kawrgs
            )
        elif win_attn == "ntrans":
            self.attn = NystromAttention(dim=dim, dim_head=head_dim, heads=num_heads, dropout=drop)

        self.wandb = wandb
        self.fl = fl
        self.l1_shortcut = l1_shortcut
        self.no_weight_to_all = no_weight_to_all
        self.minmax_weight = minmax_weight
        self.moe_mask_diag = moe_mask_diag
        self.mask_diag = mask_diag
        self.moe_mlp = moe_mlp
        if moe_enable:
            if moe_mlp:
                self.phi = [nn.Linear(self.dim, self.dim // 4, bias=False)]
                self.phi += [nn.Tanh()]
                self.phi += [nn.Linear(self.dim // 4, crmsa_k, bias=False)]
                self.phi = nn.Sequential(*self.phi)
            else:
                self.phi = nn.Parameter(
                    torch.empty(
                        (self.dim, crmsa_k),
                    )
                )
                nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))
        else:
            self.phi = None
        self.attn_mask = None

    def padding(self, x):
        B, L, C = x.shape
        if self.window_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.window_size
            H, W = H + _n, W + _n
            window_num = int(H // self.window_size)
            window_size = self.window_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.window_num
            H, W = H + _n, W + _n
            window_size = int(H // self.window_num)
            window_num = self.window_num

        add_length = H * W - L
        if (add_length > L / (self.min_win_ratio + 1e-8) or L < self.min_win_num) and not self.rpe:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            window_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, window_num, window_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        shift_size = 0

        # padding
        x, H, W, add_length, window_num, window_size = self.padding(x)

        x = x.view(B, H, W, C)

        # cyclic shift
        if shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C

        x_windows = self.pe(x_windows)

        x_windows = x_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.phi is None:
            attn_mask = torch.diag(torch.tensor([float(-100.0) for i in range(x_windows.size(-2))], device=x_windows.device)).unsqueeze(0).repeat((x_windows.size(0), 1, 1)) if self.mask_diag else None
            if return_attn:
                attn_windows, _attn = self.attn(x_windows, attn_mask, return_attn)  # nW*B, window_size*window_size, C
                dispatch_weights, combine_weights, dispatch_weights_1 = None, None, None
            else:
                attn_windows = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_mask = torch.diag(torch.tensor([float(-100.0) for i in range(x_windows.size(0))], device=x_windows.device)).unsqueeze(0) if self.moe_mask_diag else None
            if self.moe_mlp:
                logits = self.phi(x_windows).transpose(1, 2)  # W*B, sW, window_size*window_size
            else:
                logits = torch.einsum("w p c, c n -> w p n", x_windows, self.phi).transpose(1, 2)  # nW*B, sW, window_size*window_size
            # sW = logits.size(1)
            dispatch_weights = logits.softmax(dim=-1)
            combine_weights = logits.softmax(dim=1)
            if self.minmax_weight:
                logits_min, _ = logits.min(dim=-1)
                logits_max, _ = logits.max(dim=-1)
                dispatch_weights_1 = (logits - logits_min.unsqueeze(-1)) / (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)
            else:
                dispatch_weights_1 = dispatch_weights
            if self.wandb is not None:
                self.wandb.log(
                    {
                        "dispatch_max": dispatch_weights.max(),
                        "patch_num": L,
                        "add_num": add_length,
                        "combine_max": combine_weights.max(),
                        "combine_min": combine_weights.min(),
                    },
                    commit=False,
                )
            attn_windows = torch.einsum("w p c, w n p -> w n p c", x_windows, dispatch_weights).sum(dim=-2).transpose(0, 1)  # sW, nW, C
            if return_attn:
                attn_windows, _attn = self.attn(attn_windows, attn_mask, return_attn)  # sW, nW, C
                attn_windows = attn_windows.transpose(0, 1)  # nW, sW, C
            else:
                attn_windows = self.attn(attn_windows, attn_mask).transpose(0, 1)  # nW, sW, C
            if self.no_weight_to_all:
                attn_windows = attn_windows.unsqueeze(-2).repeat((1, 1, window_size * window_size, 1))  # nW, sW, window_size*window_size, C
            else:
                attn_windows = torch.einsum("w n c, w n p -> w n p c", attn_windows, dispatch_weights_1)  # nW, sW, window_size*window_size, C
            attn_windows = torch.einsum("w n p c, w n p -> w n p c", attn_windows, combine_weights).sum(dim=1)  # nW, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)

        # pe
        # attn_windows = self.pe(attn_windows)

        # reverse cyclic shift
        if shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        if return_attn:
            return x, (_attn, dispatch_weights, combine_weights, dispatch_weights_1)
        else:
            return x


class TransLayer1(nn.Module):
    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        head=8,
        drop_out=0.1,
        drop_path=0.0,
        need_down=False,
        need_reduce=False,
        down_ratio=2,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        trans_dim=64,
        n_cycle=1,
        attn="ntrans",
        n_window=8,
        trans_conv=False,
        shift_size=False,
        window_size=0,
        rpe=False,
        min_win_num=0,
        min_win_ratio=0,
        qkv_bias=True,
        shortcut=True,
        sc_ratio="1",
        **kwargs
    ):
        super().__init__()

        if need_reduce:
            self.reduction = nn.Linear(dim, dim // down_ratio, bias=False)
            dim = dim // down_ratio
        else:
            self.reduction = nn.Identity()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()

        if attn == "ntrans":
            self.attn = NystromAttention(
                dim=dim,
                dim_head=trans_dim,  # dim // 8
                heads=head,
                num_landmarks=256,  # number of landmarks dim // 2
                pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out,
            )
        elif attn == "rrt":
            self.attn = RRTAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                window_num=n_window,
                head_dim=trans_dim,
                conv=trans_conv,
                shift_size=shift_size,
                window_size=window_size,
                rpe=rpe,
                min_win_num=min_win_num,
                min_win_ratio=min_win_ratio,
                qkv_bias=qkv_bias,
                **kwargs
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, need_attn=False):
        attn = None

        x, attn = self.forward_trans(x, need_attn=need_attn)

        if need_attn:
            return x, attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None

        x = self.reduction(x)
        B, L, C = x.shape

        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x + self.drop_path(z)

        return x, attn


class RRTEncoder(nn.Module):
    """
    learnable mask token
    """

    def __init__(
        self,
        mlp_dim=512,
        pos_pos=0,
        pos="none",
        peg_k=7,
        attn="ntrans",
        rrt_window_num=8,
        drop_out=0.1,
        n_layers=2,
        n_heads=8,
        multi_scale=False,
        drop_path=0.1,
        pool="attn",
        da_act="tanh",
        reduce_ratio=0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        da_gated=False,
        da_bias=False,
        da_dropout=False,
        trans_dim=64,
        n_cycle=1,
        trans_conv=True,
        rpe=False,
        window_size=0,
        min_win_num=0,
        min_win_ratio=0,
        qkv_bias=True,
        shift_size=False,
        peg_bias=True,
        peg_1d=False,
        init=False,
        l2_n_heads=8,
        **kwargs
    ):
        super(RRTEncoder, self).__init__()

        # 不需要降维
        if reduce_ratio == 0:
            pass
        # 根据多尺度自动降维，维度前后不变
        elif reduce_ratio == -1:
            reduce_ratio = n_layers - 1
        # 指定降维维度 (2的n次)
        else:
            pass

        self.final_dim = mlp_dim // (2**reduce_ratio) if reduce_ratio > 0 else mlp_dim
        if multi_scale:
            self.final_dim = self.final_dim * (2 ** (n_layers - 1))

        self.pool = pool
        if pool == "attn":
            self.pool_fn = DAttention(self.final_dim, da_act, gated=da_gated, bias=da_bias, dropout=da_dropout)

        self.norm = nn.LayerNorm(self.final_dim)

        l2_shortcut = True
        self.all_shortcut = False
        l2_sc_ratio = 0
        l2_conv = 0
        if l2_conv == 0:
            l2_conv_k = 0
            l2_conv = False
        else:
            l2_conv_k = l2_conv
            l2_conv = True

        self.layer1 = TransLayer1(
            dim=mlp_dim,
            head=n_heads,
            drop_out=drop_out,
            drop_path=drop_path,
            need_down=multi_scale,
            need_reduce=reduce_ratio != 0,
            down_ratio=2**reduce_ratio,
            ffn=ffn,
            ffn_act=ffn_act,
            mlp_ratio=mlp_ratio,
            trans_dim=trans_dim,
            n_cycle=n_cycle,
            attn=attn,
            n_window=rrt_window_num,
            trans_conv=trans_conv,
            rpe=rpe,
            window_size=window_size,
            min_win_num=min_win_num,
            min_win_ratio=min_win_ratio,
            qkv_bias=qkv_bias,
            shift_size=shift_size,
            moe_enable=False,
            fl=True,
            shortcut=True,
            **kwargs
        )

        if reduce_ratio > 0:
            mlp_dim = mlp_dim // (2**reduce_ratio)

        if multi_scale:
            mlp_dim = mlp_dim * 2

        if n_layers >= 2:
            self.layers = []
            self.layers += [
                TransLayer1(
                    dim=mlp_dim,
                    head=l2_n_heads,
                    drop_out=drop_out,
                    drop_path=drop_path,
                    need_down=multi_scale,
                    need_reduce=reduce_ratio != 0,
                    down_ratio=2**reduce_ratio,
                    ffn=ffn,
                    ffn_act=ffn_act,
                    mlp_ratio=mlp_ratio,
                    trans_dim=trans_dim,
                    n_cycle=n_cycle,
                    attn=attn,
                    n_window=rrt_window_num,
                    rpe=rpe,
                    window_size=window_size,
                    min_win_num=min_win_num,
                    min_win_ratio=min_win_ratio,
                    qkv_bias=qkv_bias,
                    shift_size=shift_size,
                    moe_enable=True,
                    trans_conv=False,
                    shortcut=l2_shortcut,
                    sc_ratio=l2_sc_ratio,
                    **kwargs
                )
            ]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = nn.Identity()

        if pos == "ppeg":
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        elif pos == "peg":
            self.pos_embedding = PEG(mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if init:
            self.apply(initialize_weights)

    def forward(self, x, no_pool=False, return_trans_attn=False, return_attn=False, no_norm=False):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            shape_len = 4
        batch, num_patches, C = x.shape  # 直接是特征

        # translayer1
        x_shortcut = x
        if return_trans_attn:
            x, trans_attn = self.layer1(x, True)
        else:
            x = self.layer1(x)
            trans_attn = None

        x = self.pos_embedding(x)

        # translayer2
        for i, layer in enumerate(self.layers.children()):
            if return_trans_attn:
                x, trans_attn = layer(x, return_trans_attn)
            else:
                x = layer(x)
                trans_attn = None

        if self.all_shortcut:
            x = x + x_shortcut

        # ---->cls_token
        x = self.norm(x)

        if no_pool:
            if shape_len == 2:
                x = x.squeeze(0)
            elif shape_len == 4:
                x = x.transpose(1, 2)
                x = x.reshape(batch, C, int(num_patches**0.5), int(num_patches**0.5))
            return x

        if self.pool == "attn":
            if return_attn:
                logits, a = self.pool_fn(x, return_attn=True, no_norm=no_norm)
            else:
                logits = self.pool_fn(x)
        else:
            logits = x

        if shape_len == 2:
            logits = logits.squeeze(0)
        elif shape_len == 4:
            logits = logits.transpose(1, 2)
            logits = logits.reshape(batch, C, int(num_patches**0.5), int(num_patches**0.5))

        if return_attn:
            return logits, a, trans_attn

        else:
            return logits, trans_attn


class RRT(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        mlp_dim=512,
        act="relu",
        n_classes=2,
        dropout=0.25,
        pos_pos=0,
        pos="none",
        peg_k=7,
        attn="rrt",
        pool="attn",
        rrt_window_num=8,
        n_layers=2,
        n_heads=8,
        multi_scale=False,
        drop_path=0.0,
        da_act="tanh",
        trans_dropout=0.1,
        reduce_ratio=0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        da_gated=False,
        da_bias=False,
        da_dropout=False,
        trans_dim=64,
        n_cycle=1,
        trans_conv=False,
        rpe=False,
        window_size=0,
        min_win_num=0,
        min_win_ratio=0,
        qkv_bias=True,
        shift_size=False,
        **kwargs
    ):
        super(RRT, self).__init__()

        self.patch_to_emb = [nn.Linear(input_dim, mlp_dim)]

        if act.lower() == "relu":
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == "gelu":
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.rrt_window_num = rrt_window_num

        self.online_encoder = RRTEncoder(
            mlp_dim=mlp_dim,
            pos_pos=pos_pos,
            pos=pos,
            peg_k=peg_k,
            attn=attn,
            rrt_window_num=rrt_window_num,
            n_layers=n_layers,
            n_heads=n_heads,
            multi_scale=multi_scale,
            drop_path=drop_path,
            pool=pool,
            da_act=da_act,
            drop_out=trans_dropout,
            reduce_ratio=reduce_ratio,
            ffn=ffn,
            ffn_act=ffn_act,
            mlp_ratio=mlp_ratio,
            da_gated=da_gated,
            da_bias=da_bias,
            da_dropout=da_dropout,
            trans_dim=trans_dim,
            n_cycle=n_cycle,
            trans_conv=trans_conv,
            rpe=rpe,
            window_size=window_size,
            min_win_num=min_win_num,
            min_win_ratio=min_win_ratio,
            qkv_bias=qkv_bias,
            shift_size=shift_size,
            **kwargs
        )

        self.predictor = nn.Linear(self.online_encoder.final_dim, n_classes)

        self.apply(initialize_weights)

    def forward(self, x, return_attn=False, no_norm=False, return_trans_attn=False):
        x = self.patch_to_emb(x)  # n*512
        x = self.dp(x)

        # forward online network
        if return_attn:
            x, a, t_a = self.online_encoder(x, return_attn=True, no_norm=no_norm, return_trans_attn=return_trans_attn)
        else:
            x, t_a = self.online_encoder(x, return_trans_attn=return_trans_attn)

        # prediction
        logits = self.predictor(x)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S
