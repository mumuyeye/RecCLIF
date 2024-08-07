import sys
import torch
import torch.nn as nn
from .modules import (
    Transformer,
    Transformer_Conv,
    OverlapPatchEmbed,
    FeedForward_Conv,
    Ef_Transformer,
    calculate_num_patches,
    UserEmbeddingModel,
    Pre_LN_Transformer,
    User_cls_add,
)
from einops.layers.torch import Rearrange
from einops import repeat
import torch.nn.init as init
import math


class subvcf(nn.Module):
    def __init__(self, args):
        super(subvcf, self).__init__()
        self.args = args
        dim_head = 64
        sub_size = args.block_size
        patch_size = args.patch_size
        # assert sub_size % patch_size == 0 ,"sub_size should be divided by patch_size"
        dim = args.sub_dim
        dropout = args.sub_dropout
        depth = args.sub_depth
        heads = args.sub_heads
        mlp_dim = args.sub_mlp_dim
        num_patches = (sub_size // patch_size) ** 2
        stride = args.stride
        num_patches = calculate_num_patches(
            sub_size, patch_size, stride, patch_size // 2
        )
        patch_dim = patch_size**2
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b (h p1) (w p2) -> b (h w) (p1 p2)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # self.to_overlap_patch_embedding = OverlapPatchEmbed(img_size=sub_size, stride=stride, patch_size=patch_size, embed_dim=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = Pre_LN_Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.to_latent = nn.Identity()

    def forward(self, sub):
        sub = self.to_patch_embedding(sub)
        assert not torch.isnan(
            sub
        ).any(), "NaN detected after to_overlap_patch_embedding"

        b, n, _ = sub.shape
        sub += self.pos_embedding[:, :(n)]
        assert not torch.isnan(
            sub
        ).any(), "NaN detected after adding positional embeddings"

        sub = self.dropout(sub)
        assert not torch.isnan(sub).any(), "NaN detected after dropout"

        sub = self.transformer(sub)
        assert not torch.isnan(sub).any(), "NaN detected after transformer"

        sub = sub.mean(dim=1)
        sub = self.to_latent(sub)
        assert not torch.isnan(sub).any(), "NaN detected after to_latent"

        return sub


class mlp_subvcf(nn.Module):
    def __init__(self, args):
        super(mlp_subvcf, self).__init__()
        # 初始化参数
        self.args = args
        dim = args.sub_dim
        dropout = args.sub_dropout
        mlp_layers = args.mlp_layers  # 假设args中有一个mlp_layers参数定义MLP的层数
        mlp_dim = args.sub_mlp_dim
        patch_size = args.patch_size
        sub_size = args.block_size
        stride = args.stride
        # num_patches = (sub_size // patch_size) ** 2
        num_patches = calculate_num_patches(
            sub_size, patch_size, stride, patch_size // 2
        )
        patch_dim = patch_size**2

        # 补丁嵌入
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_size, p2=patch_size),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.to_overlap_patch_embedding = OverlapPatchEmbed(
            img_size=sub_size, stride=stride, patch_size=patch_size, embed_dim=dim
        )
        # MLP替代Transformer卷积
        mlp_modules = [
            FeedForward_Conv(dim, mlp_dim, dropout) for _ in range(mlp_layers)
        ]
        self.mlp = nn.Sequential(*mlp_modules)

        self.dropout = nn.Dropout(dropout)
        self.to_latent = nn.Identity()

    def forward(self, sub):
        sub = self.to_overlap_patch_embedding(sub)
        b, n, _ = sub.shape
        sub += self.pos_embedding[:, :(n)]
        sub = self.dropout(sub)
        sub = self.mlp(sub)
        sub = sub.mean(dim=1)
        sub = self.to_latent(sub)
        return sub


class vcf(nn.Module):
    def __init__(self, args):
        super(vcf, self).__init__()
        self.args = args
        dim_head = 64
        image_height, image_width = args.image_height, args.image_width
        block_size = args.block_size
        assert (
            image_height % block_size == 0 and image_width % block_size == 0
        ), "image_size should be divided by block_size"
        dim = args.dim
        dropout = args.dropout
        depth = args.depth
        heads = args.heads
        mlp_dim = args.mlp_dim
        num_blocks = (image_height // block_size) * (image_width // block_size)
        self.to_block_embedding = nn.Sequential(
            Rearrange(
                "b (h1 p1) (w1 p2) -> b (h1 w1) p1 p2", p1=block_size, p2=block_size
            ),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_blocks + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_user_cal = nn.Sequential(
            nn.Linear(in_features=dim + dim, out_features=dim),
            nn.LayerNorm(dim),
        )
        # 可以选用Pre_in_transformer
        self.transformer = Pre_LN_Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)
        # self.user_embedding = nn.Embedding(args.num_users, dim)
        # self.user_embed_mapping = nn.Linear(args.num_items, args.dim)
        self.user_embed_mapping = UserEmbeddingModel(args.num_items, args.dim)
        self.prediction_layer = nn.Linear(dim, args.num_items)
        self.to_latent = nn.Identity()

    def check_for_nan(self, x, message="NaN detected"):
        if torch.isnan(x).any():
            print(f"Error: {message}")
            sys.exit(1)

    def forward(self, img, user_id):
        # self.check_for_nan(img, 'NaN detected in input image')

        user_ids_expanded = user_id.unsqueeze(-1).expand(-1, self.args.num_items)
        user_rows = torch.gather(img, 1, user_ids_expanded.unsqueeze(1)).squeeze(1)
        # self.check_for_nan(user_rows, 'NaN detected after gathering user rows')

        user_embed = self.user_embed_mapping(user_rows)
        # self.check_for_nan(user_embed, 'NaN detected in user embeddings')

        x = self.to_block_embedding(img)
        # self.check_for_nan(x, 'NaN detected after block embedding')

        b, n, _, _ = x.shape
        device = img.device
        sub_model = subvcf(self.args).to(device)
        block_list = torch.unbind(x, dim=1)

        block_list_checked = []
        for i, block in enumerate(block_list):
            block_out = sub_model(block)
            block_list_checked.append(block_out)

        x = torch.stack(block_list_checked, dim=1)
        # self.check_for_nan(x, 'NaN detected after sub_model embedding')

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        cls_tokens = torch.cat([cls_tokens, user_embed.unsqueeze(1)], dim=-1)
        cls_tokens = self.cls_user_cal(cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        self.check_for_nan(x, "NaN detected after transformer embedding")
        x = x[:, 0]
        x = self.to_latent(x)
        x = self.prediction_layer(x)
        # self.check_for_nan(x, 'NaN detected after prediction embedding')
        return x


class MF(nn.Module):
    def __init__(self, args):
        super(MF, self).__init__()
        self.args = args
        self.user_embedding = nn.Embedding(args.num_users, args.factor)
        self.item_embedding = nn.Embedding(args.num_items, args.factor)

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, user_id):
        user_embedding = self.user_embedding(user_id)
        all_item_embedding = self.item_embedding.weight
        return torch.matmul(user_embedding, all_item_embedding.t())


class ViTURC(nn.Module):
    def __init__(self, args):
        super(ViTURC, self).__init__()
        self.args = args
        dim_head = 64
        image_height, image_width = args.image_height, args.image_width
        patch_size = args.patch_size
        assert (
            image_height % patch_size == 0 and image_width % patch_size == 0
        ), "image_size should be divided by patch_size"
        dim = args.dim
        dropout = args.dropout
        depth = args.depth
        heads = args.heads
        mlp_dim = args.mlp_dim
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        block_dim = patch_size**2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 可以选用Pre_in_transformer
        self.transformer = Pre_LN_Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b (h p1) (w p2) -> b (h w) (p1 p2)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(block_dim),
            nn.Linear(block_dim, dim),
            nn.LayerNorm(dim),
        )
        # self.prediction_layer = nn.Linear(dim, args.num_items)
        self.to_latent = nn.Identity()
        self.user_cls_add = User_cls_add(args.num_items, args.dim)
        self.num_items = args.num_items

    def forward(self, matrix):
        # 取出user对应的行
        batch_size = matrix.size(0)
        user_id = torch.ones(batch_size, dtype=torch.int64).to(matrix.device)
        user_ids_expanded = user_id.view(batch_size, 1).expand(-1, self.num_items)
        user_rows = torch.gather(matrix, 1, user_ids_expanded.unsqueeze(1)).squeeze(1)

        cls_tokens = self.user_cls_add(user_rows, self.cls_token)
        x = self.to_patch_embedding(matrix)
        b, n, _ = x.shape
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.to_latent(x)
        return x


class Item_Encoder(nn.Module):
    def __init__(self, args):
        super(Item_Encoder, self).__init__()
        self.args = args
        self.item_mapping = nn.Linear(args.num_users, args.dim)

    def forward(self, matrix, item_ids):
        batch_size, num_users, num_items = matrix.shape
        item_ids_expanded = item_ids.unsqueeze(1).expand(-1, num_users, -1)
        item_columns = torch.gather(matrix, 2, item_ids_expanded)
        item_columns = item_columns.permute(0, 2, 1)
        item_embeddings = self.item_mapping(item_columns)
        return item_embeddings


# class Config:
#     def __init__(self):
#         self.image_size = 256  # 假设的图像尺寸
#         self.block_size = 32  # 块的尺寸
#         self.dim = 512  # 特征维度
#         self.dropout = 0.1  # dropout 比率
#         self.depth = 6  # transformer 的深度
#         self.heads = 8  # 注意力头的数量
#         self.mlp_dim = 1024  # MLP的维度
#         self.num_users = 1000  # 用户总数
#         self.num_items = 10000  # 物品总数
#         self.sub_size = 64  # subvcf中子块的尺寸
#         self.patch_size = 16  # 子块中的补丁尺寸
#         self.sub_dim = 512  # subvcf中的维度
#         self.sub_dropout = 0.1  # subvcf中的dropout比率
#         self.sub_depth = 4  # subvcf中transformer的深度
#         self.sub_heads = 4  # subvcf中的注意力头数量
#         self.sub_mlp_dim = 1024  # subvcf中MLP的维度

# # 创建一个 args 实例
# args = Config()

# # 实例化模型
# model = vcf(args)

# def print_model_parameters(model):
#     total_params = 0
#     print("Model's parameters and their shapes:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.size()}")
#         total_params += param.numel()
#     total_billions = total_params / 1e9  # Convert number of parameters to billions

#     print(f"\nTotal parameters: {total_billions:.3f} Billion")
# print_model_parameters(model)

# # 模拟数据
# batch_size = 8
# img = torch.randn(batch_size, args.image_size, args.image_size)  # 假设输入图像为3通道
# user_ids = torch.randint(0, args.num_users, (batch_size,))  # 随机生成用户ID

# # 将模型和数据移至合适的设备
# device = torch.device("cpu")
# model = model.to(device)
# img = img.to(device)
# user_ids = user_ids.to(device)

# # 前向传播
# output = model(img, user_ids)

# # 输出结果
# print(output.shape)  # 应该是 [batch_size, num_items]
