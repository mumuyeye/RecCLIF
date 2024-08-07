import math
import torch.nn as nn
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.init as init
import torch.nn.functional as F


class FeedForward_Conv(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)  # 假设DWConv的维度与hidden_dim一致
        self.act = nn.GELU()  # 可以替换成nn.ReLU()或任何其他激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming Uniform initialization
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Apply truncated normal initialization
            # Truncated normal is not directly available in PyTorch, but we can mimic its behavior
            # Here, we'll initialize normally and then clamp the weights
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # self.dwconv = DWConv(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # 可以换RELU?
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        return self.net(x)

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming Uniform initialization
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Apply truncated normal initialization
            # Truncated normal is not directly available in PyTorch, but we can mimic its behavior
            # Here, we'll initialize normally and then clamp the weights
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming Uniform initialization
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Apply truncated normal initialization
            # Truncated normal is not directly available in PyTorch, but we can mimic its behavior
            # Here, we'll initialize normally and then clamp the weights
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Ef_Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        # Assuming H and W are the same and N is a perfect square
        assert math.sqrt(N) % 1 == 0, f"N must be a perfect square, got N={N}"
        H = W = int(math.sqrt(N))

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )
        self.apply(self._init_weights)

    def _check_nan(self, tensor, message):
        if torch.isnan(tensor).any():
            raise RuntimeError(f"NaN detected in {message}")

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            self._check_nan(x, f"after attention residual connection")
            x = ff(x) + x
            self._check_nan(x, f"after feedforward residual connection")

        return self.norm(x)

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming Uniform initialization
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Apply truncated normal initialization
            # Truncated normal is not directly available in PyTorch, but we can mimic its behavior
            # Here, we'll initialize normally and then clamp the weights
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Transformer_Conv(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward_Conv(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )
        self.apply(self._init_weights)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming Uniform initialization
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Apply truncated normal initialization
            # Truncated normal is not directly available in PyTorch, but we can mimic its behavior
            # Here, we'll initialize normally and then clamp the weights
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()


class Ef_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Ef_Attention(
                            dim, heads=heads, dim_head=dim_head, dropout=dropout
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )
        self.apply(self._init_weights)

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x_orig = x
            x = attn(x) + x
            if torch.isnan(x).any():
                print(f"NaN detected after attention layer {i}")

            x = ff(x) + x
            if torch.isnan(x).any():
                print(f"NaN detected after feedforward layer {i}")

            # 额外检查：确保残差连接后不引入NaN
            if torch.isnan(x_orig).any():
                print(f"NaN detected in original input to layer {i}")
            if torch.isnan(x - x_orig).any():
                print(f"NaN introduced by layer {i}")

        x = self.norm(x)
        if torch.isnan(x).any():
            print("NaN detected after applying LayerNorm")

        return x

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming Uniform initialization
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Apply truncated normal initialization
            # Truncated normal is not directly available in PyTorch, but we can mimic its behavior
            # Here, we'll initialize normally and then clamp the weights
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()


class Pre_LN_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)  # 初始层归一化，可选
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),  # Pre-LN，用于Attention前的归一化
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        nn.LayerNorm(dim),  # Pre-LN，用于FeedForward前的归一化
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )
        self.final_norm = nn.LayerNorm(dim)  # 在整个Transformer最后的归一化
        self.apply(self._init_weights)

    def _check_nan(self, tensor, message):
        if torch.isnan(tensor).any():
            raise RuntimeError(f"NaN detected in {message}")

    def forward(self, x):
        # x = self.norm(x)  # 初始层归一化，可选
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x  # 应用Pre-LN然后执行Attention操作和残差连接
            self._check_nan(x, "after attention residual connection")
            x = ff(norm2(x)) + x  # 应用Pre-LN然后执行FeedForward操作和残差连接
            self._check_nan(x, "after feedforward residual connection")

        return self.final_norm(x)  # 在整个Transformer最后应用层归一化

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        # 计算H和W，假设N是完全平方数
        sqrt_N = int(math.sqrt(N))
        if sqrt_N * sqrt_N == N:
            # 如果N是完全平方数，则执行DWConv操作
            H = W = sqrt_N
            x = x.transpose(1, 2).view(B, C, H, W)
            x = self.dwconv(x)
            x = x.flatten(2).transpose(1, 2)
        return x


def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=25, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.embed_dim = embed_dim

        # Assuming patch_size > stride to allow overlapping
        assert (
            max(self.patch_size) > self.stride
        ), "Patch size must be greater than stride."

        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=(self.patch_size[0] // 2, self.patch_size[1] // 2),
        )  # Adjust padding as needed
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming Uniform initialization
            init.kaiming_uniform_(
                module.weight, a=math.sqrt(5)
            )  # a is the negative slope of the rectifier
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            # Apply truncated normal initialization
            # Truncated normal is not directly available in PyTorch, but we can mimic its behavior
            # Here, we'll initialize normally and then clamp the weights
            mean = 0.0
            std = 0.02
            module.weight.data.normal_(mean, std)
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(
                    min=mean - 2 * std, max=mean + 2 * std
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        # Ensure input has a channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Shape becomes (N, 1, H, W)

        x = self.proj(x)  # Apply convolution, output shape (N, embed_dim, H', W')
        N, C, H, W = x.shape

        # Calculate number of patches
        # For simplicity, here we directly use H and W from the output shape,
        # which are effectively the number of patches in each dimension
        num_patches = H * W

        # Reshape to (N, num_patches, embed_dim)
        x = x.permute(0, 2, 3, 1).reshape(N, num_patches, C)
        x = self.norm(x)

        return x


def calculate_num_patches(img_size, patch_size, stride, padding):
    # Ensure img_size and patch_size are 2-tuple for H and W
    img_size = to_2tuple(img_size)
    patch_size = to_2tuple(patch_size)
    padding = to_2tuple(padding)

    num_patches_h = ((img_size[0] + 2 * padding[0] - patch_size[0]) // stride) + 1
    num_patches_w = ((img_size[1] + 2 * padding[1] - patch_size[1]) // stride) + 1

    total_num_patches = num_patches_h * num_patches_w
    return total_num_patches


class UserEmbeddingModel(nn.Module):
    def __init__(self, num_items, dim):
        super(UserEmbeddingModel, self).__init__()
        # 分解为两个连续的层，中间维度假设为dim的一半
        self.intermediate_dim = dim // 2  # 中间层的维度
        self.user_embed_mapping_part1 = nn.Linear(num_items, self.intermediate_dim)
        self.activation = nn.ReLU()  # 中间的激活函数
        self.user_embed_mapping_part2 = nn.Linear(self.intermediate_dim, dim)

    def forward(self, x):
        x = self.user_embed_mapping_part1(x)
        x = self.activation(x)
        x = self.user_embed_mapping_part2(x)
        return x


class User_cls_add(nn.Module):
    def __init__(self, num_items, cls_dim):
        super(User_cls_add, self).__init__()
        self.mapping = nn.Linear(num_items, cls_dim)  # 将user_rows映射到cls的维度

    def forward(self, user_rows, cls):
        # 映射user_rows到cls维度
        user_rows_mapped = self.mapping(user_rows)  # [batch_size, cls_dim]

        # 扩展cls以匹配user_rows的批次大小
        cls_expanded = cls.repeat(user_rows.size(0), 1, 1).squeeze(
            1
        )  # [batch_size, cls_dim]

        # 将映射后的user_rows与扩展的cls相结合
        combined_feature = cls_expanded + user_rows_mapped  # [batch_size, cls_dim]

        # 调整输出维度
        combined_feature = combined_feature.unsqueeze(1)  # [batch_size, 1, cls_dim]

        return combined_feature
