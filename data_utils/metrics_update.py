import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import (
    movielens_dataset,
    SequentialDistributedSampler,
    TestDataset,
    ClipTestDataset,
)
from .dataset_gl import Vit_dataset
from .dataset_czy import ClipRecDataset
from model import longclip
import torch.distributed as dist
import re


import re


def generate_texts(batch_initial_texts):
    all_generated_texts = []
    all_pos_indices = []

    for initial_text in batch_initial_texts:
        batch_generated_texts = []
        batch_pos_indices = []

        # 使用正则表达式提取 candidate_set 和 pos_index
        candidate_set_match = re.search(
            r"Candidate items for the predicted user are in row 1, columns ([\d, ]+).",
            initial_text,
        )
        pos_index_match = re.search(
            r"Next item for the predicted user is in column (\d+).", initial_text
        )

        if candidate_set_match and pos_index_match:
            candidate_set_str = candidate_set_match.group(1)
            candidate_set = [int(x) for x in candidate_set_str.split(", ")]
            pos_index = int(pos_index_match.group(1))

            # 循环生成每个替换后的文本
            for candidate in candidate_set:
                new_text = re.sub(
                    r"Next item for the predicted user is in column \d+",
                    f"Next item for the predicted user is in column {candidate}",
                    initial_text,
                )
                batch_generated_texts.append(new_text)
                batch_pos_indices.append(candidate)
        else:
            print("Failed to extract candidate_set or pos_index from the text.")

        all_generated_texts.append(batch_generated_texts)
        all_pos_indices.append(batch_pos_indices)

    return all_generated_texts, all_pos_indices


def id_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(
        "valid_results   {}".format("\t".join(["{:0.5f}".format(i * 100) for i in x]))
    )


def get_mean(arr):
    return [i.mean() for i in arr]


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = (
            distributed_concat(eval_m, len(test_sampler.dataset))
            .to(torch.device("cpu"))
            .numpy()
        )
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def calculate_recall(rank, ground_truth):
    # Recall is equal to HR when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / len(ground_truth)
    return result


def calculate_ndcg(rank, ground_truth):
    len_rank = len(rank)
    idcg_len = min(len(ground_truth), len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]

    dcg = np.cumsum(
        [
            1.0 / np.log2(idx + 2) if item in ground_truth else 0.0
            for idx, item in enumerate(rank)
        ]
    )
    result = dcg / idcg
    return result


def eval_model(model, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = ClipRecDataset(args, mode="val")
    test_sampler = SequentialDistributedSampler(
        eval_dataset, batch_size=test_batch_size
    )
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )
    model.eval()
    topK = 5
    recalls, ndcgs = [], []
    Log_file.info(
        "valid_methods  {}".format(
            "\t".join(["Recall{}".format(topK), "nDCG{}".format(topK)])
        )
    )
    with torch.no_grad():
        for data in eval_dl:
            matrixs, positive_items, _ = (
                data["matrix"],
                data["pos_items"],
                data["neg_items"],
            )
            matrixs, positive_items = (
                matrixs.to(local_rank),
                positive_items.to(local_rank),
            )

            preds = model.module.evaluate(
                matrixs
            )  # Assuming this returns [batch_size, num_candidate_items]

            # 分别对每个样本计算评估指标
            batch_recalls = []
            batch_ndcgs = []
            for i in range(matrixs.size(0)):
                single_pred = preds[i]
                single_positive_item = positive_items[i]

                recall = calculate_recall(single_pred[:5], single_positive_item)
                ndcg = calculate_ndcg(single_pred[:5], single_positive_item)

                # 存储结果
                batch_recalls.append(torch.tensor(recall[-1], device=local_rank))
                batch_ndcgs.append(torch.tensor(ndcg[-1], device=local_rank))

            # 将本批次的结果合并到总结果中
            recalls.append(torch.stack(batch_recalls))
            ndcgs.append(torch.stack(batch_ndcgs))

        # 合并所有批次的结果
        recalls = torch.cat(recalls, dim=0)
        ndcgs = torch.cat(ndcgs, dim=0)
        mean_eval = eval_concat([recalls, ndcgs], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)

    return mean_eval[0]


def eval_model_clip(
    model, test_batch_size, args, item_num, Log_file, v_or_t, local_rank
):
    eval_dataset = ClipRecDataset(args, mode="val")  # 根据实际情况选择合适的数据集
    test_sampler = SequentialDistributedSampler(
        eval_dataset, batch_size=test_batch_size
    )
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )
    model.eval()
    topK = 5
    ndcgKs = [10, 20]  # 新增NDCG的K值
    recalls, ndcgs_5, ndcgs_10, ndcgs_20 = [], [], [], []  # 分别存储各个K值的NDCG

    Log_file.info(
        "valid_methods  {:<10}{:<10}{:<10}{:<10}".format(
            "Recall5", "nDCG5", "nDCG10", "nDCG20"
        )
    )
    with torch.no_grad():
        for data in eval_dl:
            initial_texts = data["texts"]
            images, positive_items = data["images"].to(local_rank), data[
                "pos_items"
            ].to(local_rank)

            # 生成文本和对应的 pos_index
            batch_texts_list, batch_pos_indices_list = generate_texts(initial_texts)

            all_sorted_indices = []
            # 遍历每个图像及其对应的文本候选集
            for i, (texts, pos_indices) in enumerate(
                zip(batch_texts_list, batch_pos_indices_list)
            ):
                tokenized_texts = longclip.tokenize(texts, truncate=True).to(local_rank)
                text_features = model.module.model.encode_text(tokenized_texts)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                image_feature = images[i].unsqueeze(0)  # 添加批次维度以进行矩阵乘法
                image_features = model.module.model.encode_image(image_feature)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                sim = image_features @ text_features.T
                sorted_indices = torch.argsort(sim, dim=1, descending=True)
                # 映射排序索引到原始位置
                pos_indices_tensor = torch.tensor(
                    pos_indices, dtype=torch.long, device=local_rank
                )
                mapped_indices = torch.gather(
                    pos_indices_tensor, 0, sorted_indices.squeeze(0)
                )
                all_sorted_indices.append(mapped_indices)

            # 计算评价指标
            batch_recalls = []
            batch_ndcgs_5, batch_ndcgs_10, batch_ndcgs_20 = [], [], []
            for i in range(images.size(0)):
                single_pred_5 = all_sorted_indices[i][:5]  # 获取前5个预测结果
                single_pred_10 = all_sorted_indices[i][:10]  # 获取前10个预测结果
                single_pred_20 = all_sorted_indices[i][:20]  # 获取前20个预测结果
                single_positive_item = positive_items[i]

                recall = calculate_recall(single_pred_5, single_positive_item)
                ndcg_5 = calculate_ndcg(single_pred_5, single_positive_item)
                ndcg_10 = calculate_ndcg(single_pred_10, single_positive_item)
                ndcg_20 = calculate_ndcg(single_pred_20, single_positive_item)

                batch_recalls.append(torch.tensor(recall[-1], device=local_rank))
                batch_ndcgs_5.append(torch.tensor(ndcg_5[-1], device=local_rank))
                batch_ndcgs_10.append(torch.tensor(ndcg_10[-1], device=local_rank))
                batch_ndcgs_20.append(torch.tensor(ndcg_20[-1], device=local_rank))

            # 将本批次的结果合并到总结果中
            recalls.append(torch.stack(batch_recalls))
            ndcgs_5.append(torch.stack(batch_ndcgs_5))
            ndcgs_10.append(torch.stack(batch_ndcgs_10))
            ndcgs_20.append(torch.stack(batch_ndcgs_20))

        # 合并所有批次的结果
        recalls = torch.cat(recalls, dim=0)
        ndcgs_5 = torch.cat(ndcgs_5, dim=0)
        ndcgs_10 = torch.cat(ndcgs_10, dim=0)
        ndcgs_20 = torch.cat(ndcgs_20, dim=0)
        mean_eval = eval_concat([recalls, ndcgs_5, ndcgs_10, ndcgs_20], test_sampler)

        # 使用相同的格式记录评估结果
        Log_file.info(
            "valid_results  {:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}".format(
                mean_eval[0].item() * 100,
                mean_eval[1].item() * 100,
                mean_eval[2].item() * 100,
                mean_eval[3].item() * 100,
            )
        )

    return mean_eval[1]


def compute_sorted_indices(initial_texts, images, model, local_rank):
    batch_texts_list, batch_pos_indices_list = generate_texts(initial_texts)
    all_sorted_indices = []

    for i, (texts, pos_indices) in enumerate(
        zip(batch_texts_list, batch_pos_indices_list)
    ):
        tokenized_texts = longclip.tokenize(texts, truncate=True).to(local_rank)
        text_features = model.module.model.encode_text(tokenized_texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_feature = images[i].unsqueeze(
            0
        )  # Add batch dimension for matrix multiplication
        image_features = model.module.model.encode_image(image_feature)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        sim = image_features @ text_features.T
        sorted_indices = torch.argsort(sim, dim=1, descending=True)

        # Map sorted indices to original positions
        pos_indices_tensor = torch.tensor(
            pos_indices, dtype=torch.long, device=local_rank
        )
        mapped_indices = torch.gather(pos_indices_tensor, 0, sorted_indices.squeeze(0))
        all_sorted_indices.append(mapped_indices)

    all_sorted_indices = torch.stack(all_sorted_indices, dim=0)
    return all_sorted_indices


def train_eval(preds, pos_items, item_num, local_rank):
    # 初始化存储每个批次结果的列表
    recalls = []
    ndcgs = []
    # 遍历每个批次，计算recall和ndcg
    for i in range(preds.size(0)):  # 假设preds的第一个维度是batch_size
        recall = calculate_recall(preds[i, :5], pos_items[i])
        ndcg = calculate_ndcg(preds[i, :5], pos_items[i])

        recalls.append(recall[-1])
        ndcgs.append(ndcg[-1])

    # 计算平均recall和ndcg
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0

    return avg_recall, avg_ndcg
