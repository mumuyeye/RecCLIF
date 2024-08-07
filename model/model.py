import torch
from .encoders import vcf, MF, ViTURC, Item_Encoder
from . import longclip
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist


class Model2(nn.Module):
    def __init__(self, args):
        super(Model2, self).__init__()
        self.args = args
        self.encoder = vcf(args)
        self.temperature = 0.1

    def forward(self, img, user_id, ignore_mask, y):
        # 负对数似然损失
        logits = self.encoder(img, user_id)

        ignore_mask = ignore_mask[:, : logits.size(1)]
        batch_size, num_items = logits.shape
        # Apply temperature scaling
        logits = logits / self.temperature

        # Apply ignore mask by setting the ignored positions to a very negative value
        # logits = logits.masked_fill(ignore_mask.bool(), float('-inf'))

        # Compute softmax probabilities
        softmax_probs = F.softmax(logits, dim=1)

        # Ensure y is long type for gather
        y = y.long()

        # Select the probabilities of the positive items using gather
        positive_probs = torch.gather(softmax_probs, 1, y)

        loss = -torch.log(positive_probs + 1e-9)
        return loss.mean()


class MF_model(nn.Module):
    def __init__(self, args):
        super(MF_model, self).__init__()
        self.args = args
        self.encoder = MF(args)
        self.crieterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, user_id, pos_items, neg_items):
        # TODO 需要构建user-item对
        pass

    def evaluate(self, user_id, pos_items, candidate):
        # TODO 评估时候可能需要提供给我候选集的序号
        pass


class InBatchSampledModel(nn.Module):
    def __init__(self, args):
        super(InBatchSampledModel, self).__init__()
        self.args = args
        self.encoder = vcf(args)  # 假设vcf是你的编码器
        self.temperature = 0.1

    def forward(self, img, user_id, ignore_mask, y):
        # Encoder generates logits
        logits = self.encoder(
            img, user_id
        )  # 假设编码器返回的logits维度为 [batch_size, num_items]

        ignore_mask = ignore_mask[:, : logits.size(1)]
        # Apply temperature scaling
        logits = logits / self.temperature

        # Apply ignore mask if necessary
        # Here we assume ignore_mask indicates items that should not be considered as negative samples
        # logits = logits.masked_fill(ignore_mask.bool(), float('-inf'))

        # Compute softmax probabilities across items for each user
        softmax_probs = F.softmax(logits, dim=1)

        # In-batch negatives for contrastive learning
        # Assuming y are indices of positive items, reshape for broadcasting
        y_reshaped = y.view(-1, 1)
        pos_mask = torch.eq(
            y_reshaped, torch.arange(logits.size(1)).to(y.device)
        ).float()

        # Subtract positive item probabilities to obtain negatives only
        neg_probs = softmax_probs * (1 - pos_mask)
        neg_probs_sum = neg_probs.sum(dim=1, keepdim=True)

        # Select the probabilities of the positive items
        positive_probs = torch.gather(softmax_probs, 1, y_reshaped)

        # Contrastive loss as negative log likelihood of positive item probability over sum of negatives
        loss = -torch.log(positive_probs / (neg_probs_sum + 1e-9))

        return loss.mean()


class ViTURC_MODEL(nn.Module):
    def __init__(self, args):
        super(ViTURC_MODEL, self).__init__()  # 添加了对父类构造函数的调用
        self.args = args
        self.user_encoder = ViTURC(args)
        self.item_encoder = Item_Encoder(args)
        self.criterion = nn.BCEWithLogitsLoss()

    # 
    def forward(self, matrix, pos_items, neg_items):
        user_features = self.user_encoder(matrix)

        pos_item_features = self.item_encoder(matrix, pos_items)
        neg_item_features = self.item_encoder(matrix, neg_items)

        # 扩展user_features以匹配pos_item_features和neg_item_features的维度
        expanded_user_features_pos = user_features.unsqueeze(1).expand(
            -1, pos_item_features.size(1), -1
        )
        expanded_user_features_neg = user_features.unsqueeze(1).expand(
            -1, neg_item_features.size(1), -1
        )

        # 计算得分
        pos_scores = (expanded_user_features_pos * pos_item_features).sum(
            dim=2
        )  # 修改求和维度
        neg_scores = (expanded_user_features_neg * neg_item_features).sum(
            dim=2
        )  # 修改求和维度

        # 因为有多个负样本，需要取平均或者总和
        pos_scores = pos_scores.squeeze(1)  # 假设每个用户只有一个正样本
        neg_scores = neg_scores.mean(dim=1)  # 取负样本得分的平均值

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        pos_loss = self.criterion(pos_scores, pos_labels)
        neg_loss = self.criterion(neg_scores, neg_labels)

        loss = pos_loss + neg_loss

        return loss

    def evaluate(self, matrix):
        """
        Evaluate the model by returning the globally sorted list of candidate items for each batch,
        based on the candidate items indicated by the first row of each batch in the matrix.
        """
        batch_size = matrix.size(0)
        num_items = matrix.size(2)
        sorted_indices_list = []

        for i in range(batch_size):
            # 提取每个批次的第一个用户的候选集items的全局序号
            candidate_item_indices = torch.where(matrix[i, 0, :] == 1)[0]
            # 获取当前批次第一个用户的特征
            user_features = self.user_encoder(matrix[i].unsqueeze(0))
            # 编码当前批次的候选集items的特征
            candidate_item_features = self.item_encoder(
                matrix[i].unsqueeze(0), candidate_item_indices.unsqueeze(0)
            )

            # 计算得分
            scores = torch.matmul(
                user_features, candidate_item_features.transpose(1, 2)
            ).squeeze(0)

            # 对当前批次的候选集得分进行排序，获取排序后的局部索引
            _, sorted_local_indices = scores.sort(descending=True)

            # 将局部索引映射回全局索引
            sorted_global_indices = candidate_item_indices[sorted_local_indices]

            # 添加到结果列表
            sorted_indices_list.append(sorted_global_indices)

        return torch.stack(sorted_indices_list).squeeze()


class Rec_Long_Clip(nn.Module):
    def __init__(self, args):
        super(Rec_Long_Clip, self).__init__()
        self.args = args
        self.base_model = args.base_model
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)  # 设置当前进程的默认设备
        device = torch.device(f"cuda:{local_rank}")

        self.model, _ = longclip.load_from_clip(self.base_model, device="cpu")
        # self.long_clip_model = args.long_clip_path
        # self.model, _ = longclip.load(self.long_clip_model, device="cpu")
        self.model = self.model.float().to(device)
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)

    def PCA(self, input_tensor, PCA_dim):
        mean = torch.mean(input_tensor, dim=0)
        X_centered = input_tensor - mean.unsqueeze(0)
        X_centered = X_centered.float()
        cov_matrix = torch.mm(X_centered.T, X_centered)
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        principal_components = eigenvectors[:, :PCA_dim]
        X_transformed = torch.mm(X_centered, principal_components)
        X_reversed = torch.mm(X_transformed, principal_components.T)
        X_reversed += mean
        return X_reversed

    def forward(self, images, texts):
        # 编码图像特征并进行归一化
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 编码文本特征并进行归一化
        text_features = self.model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算图像到文本的相似度
        sim_i2t = torch.matmul(image_features, text_features.T)
        # 计算文本到图像的相似度
        sim_t2i = torch.matmul(image_features, text_features.T)
        sim_t2i = sim_t2i.T

        # 应用logit scale
        sim_i2t = self.model.logit_scale.exp() * sim_i2t
        sim_t2i = self.model.logit_scale.exp() * sim_t2i

        # 生成目标索引
        bs = images.size(0)
        targets = torch.arange(bs, dtype=torch.long).to(images.device)

        # 计算交叉熵损失，并取平均
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        return loss_itc

    def forward_short(self, images, texts):
        # 编码图像特征并进行归一化
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = self.PCA(image_features, 32)

        # 编码文本特征并进行归一化
        text_features = self.model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算图像到文本的相似度
        sim_i2t = torch.matmul(image_features, text_features.T)
        # 计算文本到图像的相似度
        sim_t2i = torch.matmul(text_features, image_features.T)
        sim_t2i = sim_t2i.T

        # 应用logit scale
        sim_i2t = self.model.logit_scale.exp() * sim_i2t
        sim_t2i = self.model.logit_scale.exp() * sim_t2i

        # 生成目标索引
        bs = images.size(0)
        targets = torch.arange(bs, dtype=torch.int).to(images.device)

        # 计算交叉熵损失，并取平均
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        return loss_itc


class Clip4Rec(nn.Module):
    def __init__(self, args):
        super(Clip4Rec, self).__init__()
        self.args = args
        self.base_model = args.base_model
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        self.model, _ = longclip.load_from_clip(self.base_model, device="cpu")
        # self.long_clip_model = args.long_clip_path
        # self.model, _ = longclip.load(self.long_clip_model, device="cpu")
        self.model = self.model.float().to(device)
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
        self.criterion = nn.BCEWithLogitsLoss()

    def PCA(self, input_tensor, PCA_dim):
        mean = torch.mean(input_tensor, dim=0)
        X_centered = input_tensor - mean.unsqueeze(0)
        X_centered = X_centered.float()
        cov_matrix = torch.mm(X_centered.T, X_centered)
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        principal_components = eigenvectors[:, :PCA_dim]
        X_transformed = torch.mm(X_centered, principal_components)
        X_reversed = torch.mm(X_transformed, principal_components.T)
        X_reversed += mean
        return X_reversed

    def forward(
        self, images, tokenized_batches, positive_items, batch_pos_indices_list
    ):
        # 编码图像特征并进行归一化
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # image_features = self.PCA(image_features, 32)

        batch_size = images.size(0)
        context_length = tokenized_batches[0].size(1)
        text_features_list = []

        # 编码每个batch的候选文本特征并进行归一化
        for i in range(batch_size):
            text_features = self.model.encode_text(tokenized_batches[i])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features)

        # 初始化正样本和负样本的损失
        pos_loss = 0.0
        neg_loss = 0.0

        for i in range(batch_size):
            pos_item = positive_items[i].item()
            pos_index = batch_pos_indices_list[i].index(pos_item)

            # 获取正样本的文本特征
            pos_text_feature = text_features_list[i][pos_index].unsqueeze(0)

            # 计算图像特征与正样本文本特征的相似度
            expanded_image_features_pos = (
                image_features[i].unsqueeze(0).expand_as(pos_text_feature)
            )
            pos_scores = (expanded_image_features_pos * pos_text_feature).sum(dim=1)

            # 获取所有负样本的文本特征
            neg_text_features = torch.cat(
                [
                    text_features_list[i][:pos_index],
                    text_features_list[i][pos_index + 1 :],
                ],
                dim=0,
            )

            # 计算图像特征与负样本文本特征的相似度
            expanded_image_features_neg = (
                image_features[i].unsqueeze(0).expand_as(neg_text_features)
            )
            neg_scores = (expanded_image_features_neg * neg_text_features).sum(dim=1)

            # 使用 logit_scale 缩放相似度
            pos_scores = pos_scores * self.model.logit_scale.exp()
            neg_scores = neg_scores * self.model.logit_scale.exp()

            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            pos_loss += self.criterion(pos_scores, pos_labels)
            neg_loss += self.criterion(neg_scores, neg_labels)

        # 平均正样本和负样本的损失
        pos_loss /= batch_size
        neg_loss /= batch_size

        # 总损失是正样本和负样本损失的总和
        total_loss = pos_loss + neg_loss

        return total_loss
