import math
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from scipy.sparse import csr_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
import time
import psutil
import os
import pickle


def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


def dataset_prepare(dataset_name):
    path = f"/home/lyn/resys_baselines/new_vcf/data_utils/LLMRec/dataset/{dataset_name}/{dataset_name}.inter"
    df = pd.read_csv(
        path, sep="\t", usecols=["user_id:token", "item_id:token", "timestamp:float"]
    )

    # 如果数据集名称是 'ml-1m-full'，进行筛选
    if dataset_name == "ml-1m-full":
        df = df[df["timestamp:float"] > 3.5]

    # 去除重复的交互记录
    df = df.drop_duplicates(subset=["user_id:token", "item_id:token"])

    # 计算每个用户和物品的交互次数
    user_interaction_counts = df["user_id:token"].value_counts()
    item_interaction_counts = df["item_id:token"].value_counts()

    # 过滤掉交互次数少于5次的用户和物品
    df = df[
        df["user_id:token"].isin(
            user_interaction_counts[user_interaction_counts >= 5].index
        )
    ]
    df = df[
        df["item_id:token"].isin(
            item_interaction_counts[item_interaction_counts >= 5].index
        )
    ]

    data_array = df.to_numpy()
    processed_data = {}
    for row in data_array:
        user_id, item_id, timestamp = row[0], row[1], row[2]
        if user_id not in processed_data:
            processed_data[user_id] = []
        processed_data[user_id].append((item_id, timestamp))

    train_data, val_data, test_data = [], [], []
    val_count, test_count = 0, 0
    for user_id, user_data in processed_data.items():
        sorted_data = sorted(user_data, key=lambda x: x[1])
        if dataset_name == "Games-6k":
            if len(sorted_data) > 2:
                train_data.append((user_id, sorted_data[0][0]))
                if val_count < 1000:
                    val_data.append((user_id, sorted_data[-2][0]))
                    val_count += 1
                if test_count < 1000:
                    test_data.append((user_id, sorted_data[-1][0]))
                    test_count += 1
            elif len(sorted_data) == 2:
                train_data.append((user_id, sorted_data[0][0]))
                if val_count < 1000:
                    val_data.append((user_id, sorted_data[1][0]))
                    val_count += 1
        else:
            if len(sorted_data) > 2:
                train_data.extend(
                    [(user_id, item_id) for item_id, _ in sorted_data[:-2]]
                )
                val_data.append((user_id, sorted_data[-2][0]))
                test_data.append((user_id, sorted_data[-1][0]))
            elif len(sorted_data) == 2:
                train_data.append((user_id, sorted_data[-2][0]))
                val_data.append((user_id, sorted_data[-1][0]))

    num_users = len(processed_data)
    num_interactions = len(data_array)
    num_train = len(train_data)
    num_val = len(val_data)
    num_test = len(test_data)

    print(f"Dataset: {dataset_name}")
    print(f"Number of users: {num_users}")
    print(f"Number of interactions: {num_interactions}")
    print(f"Training set size: {num_train}")
    print(f"Validation set size: {num_val}")
    print(f"Test set size: {num_test}")

    return processed_data, train_data, val_data, test_data


def precompute_mappings(interactions):
    print("Precomputing mappings...")
    all_items = set(
        item for user_data in interactions.values() for item, _ in user_data
    )
    item_to_index = {item: idx for idx, item in enumerate(all_items)}
    user_ids = list(interactions.keys())
    user_to_index = {user: idx for idx, user in enumerate(user_ids)}
    print(f"Total items: {len(all_items)}, Total users: {len(user_ids)}")
    return user_to_index, item_to_index, user_ids


def build_interaction_matrix(
    interactions, user_to_index, item_to_index, num_users, num_items
):
    print("Building interaction matrix...")
    rows, cols, data = [], [], []
    for user, user_data in interactions.items():
        user_idx = user_to_index[user]
        for item, _ in user_data:
            item_idx = item_to_index[item]
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(1)
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    gc.collect()  # 强制垃圾回收
    return interaction_matrix


def compute_similarity_matrix(interaction_matrix):
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(interaction_matrix)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix


def construct_matrix(
    selected_user,
    interactions,
    user_to_index,
    item_to_index,
    user_ids,
    similarity_matrix,
    interaction_matrix,  # 新增参数
    num_samples=3,
):
    num_users = len(user_to_index)
    num_items = len(item_to_index)

    # 获取目标用户的索引
    user_index = user_to_index[selected_user]

    # 获取按相似度排序的用户索引
    sorted_indices = np.argsort(-similarity_matrix[user_index])[1:]

    # 获取目标用户的交互信息
    selected_user_interactions = (
        interaction_matrix.getrow(user_index).toarray().flatten()
    )

    results = []
    attempts = 0

    while len(results) < num_samples and attempts < num_samples * 2:
        # 初始化矩阵
        matrix = np.zeros((224, len(selected_user_interactions)))
        matrix[0] = selected_user_interactions.copy()
        matrix[1] = selected_user_interactions

        # 填充相似用户的交互信息
        row_idx = 2
        for idx in sorted_indices[:222]:
            matrix[row_idx] = interaction_matrix.getrow(idx).toarray().flatten()
            row_idx += 1
            if row_idx == 224:
                break

        # 计算每列的重叠计数并选择重叠最多的列
        overlap_counts = np.sum(matrix, axis=0)
        selected_columns = np.argsort(-overlap_counts)[:224]
        matrix = matrix[:, selected_columns]

        # 获取目标用户的交互信息
        candidate_row = matrix[0].copy()
        interacted_indices = np.where(candidate_row == 1)[0]
        non_interacted_indices = np.where(candidate_row == 0)[0]

        # 检查交互和非交互物品的数量
        if len(interacted_indices) == 0 or len(non_interacted_indices) < 19:
            attempts += 1
            continue

        # 选择正样本和负样本
        if len(interacted_indices) == 1:
            positive_sample_indices = interacted_indices
        else:
            positive_sample_indices = np.random.choice(interacted_indices, 1)
            interacted_indices = np.setdiff1d(
                interacted_indices, positive_sample_indices
            )
            candidate_row[interacted_indices] = 0

        negative_sample_indices = np.random.choice(
            non_interacted_indices, 19, replace=False
        )
        candidate_row[negative_sample_indices] = 1

        # 标注正负样本
        annotations = np.array([""] * len(candidate_row))
        annotations[positive_sample_indices[0]] = "Positive"
        annotations[negative_sample_indices] = "Negative"

        # 更新矩阵中的候选行
        matrix[0] = candidate_row

        # 创建行标签
        row_labels = [
            f"Candidate (User {selected_user})",
            f"History (User {selected_user})",
        ] + [f"User {user_ids[idx]}" for idx in sorted_indices[:222]]

        # 创建DataFrame并添加注释
        df = pd.DataFrame(matrix, index=row_labels)
        df.loc["Annotations"] = annotations

        # 添加结果
        results.append(df)
        attempts += 1

        # 如果交互物品数量为1，跳出循环
        if len(interacted_indices) == 1:
            break

    return results


class ClipRecDataset(Dataset):
    def __init__(self, args, mode, num_retries=10):
        self.num_retries = num_retries
        self.args = args
        self.mode = mode
        self._imdb = []
        datasetlist = list(args.dataset_name.split(" "))
        self.precomputed_data = {}

        for name in datasetlist:
            cache_file = f"{name}_precomputed.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    self.precomputed_data[name] = pickle.load(f)
            else:
                self.dataset, self.train_data, self.val_data, self.test_data = (
                    dataset_prepare(name)
                )
                print(f"Loading data from {name} in {mode} mode")
                user_to_index, item_to_index, user_ids = precompute_mappings(
                    self.dataset
                )
                num_users = len(user_to_index)
                num_items = len(item_to_index)
                interaction_matrix = build_interaction_matrix(
                    self.dataset, user_to_index, item_to_index, num_users, num_items
                )
                similarity_matrix = compute_similarity_matrix(interaction_matrix)

                self.precomputed_data[name] = {
                    "user_to_index": user_to_index,
                    "item_to_index": item_to_index,
                    "user_ids": user_ids,
                    "similarity_matrix": similarity_matrix,
                    "interaction_matrix": interaction_matrix,
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(self.precomputed_data[name], f)

            # Ensure self.train_data, self.val_data, self.test_data are set before calling _construct_imdb
            self.dataset, self.train_data, self.val_data, self.test_data = (
                dataset_prepare(name)
            )
            self._construct_imdb(name)

    def prompt_generate(self, userid, interactions, precomputed):
        user_to_index = precomputed["user_to_index"]
        item_to_index = precomputed["item_to_index"]
        user_ids = precomputed["user_ids"]
        similarity_matrix = precomputed["similarity_matrix"]
        interaction_matrix = precomputed["interaction_matrix"]

        matrices = construct_matrix(
            userid,
            interactions,
            user_to_index,
            item_to_index,
            user_ids,
            similarity_matrix,
            interaction_matrix,  # 传递缓存的交互矩阵
        )
        results = []
        for matrix in matrices:
            candidate_row = matrix.iloc[0].values
            candidate_set = np.where(candidate_row == 1)[0]
            user_history = np.where(matrix.iloc[1].values == 1)[0]
            pos_index = candidate_set[0] if len(candidate_set) > 0 else None

            description = (
                f"Rows represent users, columns represent items. Black indicates interaction, white indicates no interaction. "
                f"Interacted items for the predicted user are in row 2, columns {', '.join(str(i) for i in user_history)}. "
                f"Candidate items for the predicted user are in row 1, columns {', '.join(str(i) for i in candidate_set)}. "
                f"Next item for the predicted user is in column {pos_index}."
            )

            img = np.ones((224, 224, 3), dtype=np.uint8) * 255
            img[np.where(matrix.values == 1)] = [0, 0, 0]
            img_reshaped = img.transpose(2, 0, 1)

            result = {
                "matrix": img_reshaped,
                "texts": description,
                "pos_items": [pos_index] if pos_index is not None else [-1],
            }
            results.append(result)

        return results

    def _construct_imdb(self, name):
        user_data = (
            self.train_data
            if self.mode == "train"
            else self.val_data if self.mode == "val" else self.test_data
        )
        interactions = {
            "train": self._transform_interactions(self.train_data),
            "val": self._transform_interactions(self.val_data),
            "test": self._transform_interactions(self.test_data),
        }
        transformed_interactions = interactions[self.mode]
        unique_user_ids = list(set(user_id for user_id, _ in user_data))

        num_processes = min(cpu_count(), 2)  # 进一步减少进程数以降低内存消耗
        chunk_size = (len(unique_user_ids) + num_processes - 1) // num_processes
        user_chunks = [
            unique_user_ids[i : i + chunk_size]
            for i in range(0, len(unique_user_ids), chunk_size)
        ]

        results = []
        with Pool(num_processes) as pool:
            for chunk in user_chunks:
                results.append(
                    pool.apply_async(
                        self.process_batch, args=(chunk, transformed_interactions)
                    )
                )

            with tqdm(total=len(unique_user_ids), desc="Processing users") as pbar:
                for res in results:
                    batch_results = res.get()
                    for result_list in batch_results:
                        for result in result_list:
                            self._imdb.append(result)
                            if self.mode == "train":
                                for _ in range(self.args.sf_num):
                                    if self.args.shuffle_mode == "rows":
                                        self._imdb.append(self.shuffle_rows(result))
                                    elif self.args.shuffle_mode == "columns":
                                        self._imdb.append(self.shuffle_columns(result))
                            pbar.update(1)
                    print_memory_usage()  # 在每个批次处理后打印内存使用情况
                    gc.collect()

        print(f"Number of matrices: {len(self._imdb)}")
        print_memory_usage()  # 在所有批次处理完成后打印内存使用情况

    def process_batch(self, user_ids, interactions):
        results = []
        for user_id in user_ids:
            result = self.prompt_generate(
                user_id, interactions, self.precomputed_data[self.args.dataset_name]
            )
            results.append(result)
        print_memory_usage()  # 在每个批次处理后打印内存使用情况
        gc.collect()
        return results

    def _transform_interactions(self, data):
        interactions = {}
        for user, item in data:
            if user not in interactions:
                interactions[user] = []
            interactions[user].append(
                (item, int(time.time() * 1000) + len(interactions[user]))
            )
        return interactions

    def shuffle_rows(self, data):
        matrix = data["matrix"]
        fixed_part = matrix[:2, :]
        shuffle_part = matrix[2:, :]
        np.random.shuffle(shuffle_part)
        shuffled_matrix = np.vstack((fixed_part, shuffle_part))
        shuffled_data = {
            "matrix": shuffled_matrix,
            "texts": data["texts"],
            "pos_items": data["pos_items"],
        }
        return shuffled_data

    def shuffle_columns(self, data):
        matrix = data["matrix"]
        shuffled_matrix = matrix[:, np.random.permutation(matrix.shape[1])]
        shuffled_data = {
            "matrix": shuffled_matrix,
            "texts": data["texts"],
            "pos_items": data["pos_items"],
        }
        return shuffled_data

    def __len__(self):
        return len(self._imdb)

    def __getitem__(self, idx):
        pos_items = self._imdb[idx]["pos_items"]
        pos_items_tensor = torch.tensor(pos_items, dtype=torch.int64)
        return {
            "images": torch.tensor(self._imdb[idx]["matrix"]),
            "texts": self._imdb[idx]["texts"],
            "pos_items": pos_items_tensor,
        }


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = (
            int(
                math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)
            )
            * self.batch_size
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        return iter(indices)

    def __len__(self):
        return self.num_samples


# # 参数设置
# class Args:
#     dataset_name = "lastfm"
#     shuffle_mode = "rows"
#     sf_num = 1


# args = Args()

# start_time = time.time()
# # 创建数据集实例
# dataset = ClipRecDataset(args, mode="train")

# end_time = time.time()
# print(f"数据集创建时间: {end_time - start_time}")
# # 创建DataLoader
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# def print_sample_data(data_loader):
#     for i, batch in enumerate(data_loader):
#         if i == 0:  # 只打印第一个batch的数据
#             print("\n示例数据 (第一个batch):")
#             print(f"Images shape: {batch['images'].shape}")
#             print(f"Texts: {batch['texts']}")
#             print(f"Positive items: {batch['pos_items']}")
#             break


# # 运行测试脚本
# if __name__ == "__main__":
#     print("测试开始...\n")
#     print_sample_data(data_loader)
#     print("\n测试结束")

# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import random
# from scipy.sparse import csr_matrix
# from joblib import Memory, Parallel, delayed

# # 创建缓存目录
# memory = Memory(location='cache_directory', verbose=0)

# # 缓存相似度计算
# @memory.cache
# def compute_similarity_matrix(interaction_matrix):
#     return cosine_similarity(interaction_matrix)

# def construct_matrix(selected_user, interactions, num_samples=3):
#     # 构建用户-物品交互矩阵
#     all_items = set(
#         item for user_data in interactions.values() for item, _ in user_data
#     )
#     item_to_index = {item: idx for idx, item in enumerate(all_items)}
#     num_items = len(all_items)

#     user_ids = list(interactions.keys())
#     user_to_index = {user: idx for idx, user in enumerate(user_ids)}
#     num_users = len(user_ids)

#     row_ind = []
#     col_ind = []
#     for user, user_data in interactions.items():
#         user_idx = user_to_index[user]
#         for item, _ in user_data:
#             item_idx = item_to_index[item]
#             row_ind.append(user_idx)
#             col_ind.append(item_idx)

#     interaction_matrix = csr_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(num_users, num_items))

#     # 计算相似度
#     similarity_matrix = compute_similarity_matrix(interaction_matrix)
#     user_index = user_to_index[selected_user]
#     sorted_indices = np.argsort(-similarity_matrix[user_index])[1:]  # 按相似度排序

#     # 构建候选集和历史交互矩阵
#     selected_user_interactions = interaction_matrix.getrow(user_index).toarray().flatten()

#     # 初始化结果列表
#     results = []

#     def process_sample():
#         matrix = []
#         matrix.append(
#             selected_user_interactions.copy()
#         )  # 第一行，候选集（暂时使用历史交互代替）
#         matrix.append(selected_user_interactions)  # 第二行，历史交互

#         for idx in sorted_indices:
#             matrix.append(interaction_matrix.getrow(idx).toarray().flatten())
#             if len(matrix) == 224:  # 5的时候也还可以
#                 break

#         matrix = np.array(matrix)

#         # 选择224列
#         overlap_counts = np.sum(matrix, axis=0)
#         selected_columns = np.argsort(-overlap_counts)[:224]
#         matrix = matrix[:, selected_columns]

#         # 处理第一行
#         candidate_row = matrix[0].copy()
#         interacted_indices = np.where(candidate_row == 1)[0]
#         non_interacted_indices = np.where(candidate_row == 0)[0]

#         # 如果没有正样本或负样本则跳过本次尝试
#         if len(interacted_indices) == 0 or len(non_interacted_indices) < 19:
#             return None

#         # 如果正样本只有一个，则只尝试一次
#         if len(interacted_indices) == 1:
#             positive_sample_indices = interacted_indices
#         else:
#             # 随机选择一个历史交互作为正样本
#             positive_sample_indices = random.sample(list(interacted_indices), 1)
#             interacted_indices = interacted_indices[
#                 ~np.isin(interacted_indices, positive_sample_indices)
#             ]
#             candidate_row[interacted_indices] = 0  # 隐去其他历史交互

#         # 选择负样本
#         negative_sample_indices = random.sample(list(non_interacted_indices), 19)
#         candidate_row[negative_sample_indices] = 1  # 将负样本所在位置置为1

#         # 标注正样本和负样本
#         annotations = [""] * len(candidate_row)
#         for idx in positive_sample_indices:
#             annotations[idx] = "Positive"
#         for idx in negative_sample_indices:
#             annotations[idx] = "Negative"

#         # 将第一行更新为新的候选集
#         matrix[0] = candidate_row

#         # 为矩阵添加行标注
#         row_labels = [
#             f"Candidate (User {selected_user})",
#             f"History (User {selected_user})",
#         ] + [
#             f"User {user_ids[idx]}" for idx in sorted_indices[:222]
#         ]  # 224 - 2

#         # 将矩阵转换为DataFrame并添加注释信息
#         df = pd.DataFrame(matrix, index=row_labels)
#         df.loc["Annotations"] = annotations

#         return df

#     results = Parallel(n_jobs=-1)(delayed(process_sample)() for _ in range(num_samples * 2))
#     results = [res for res in results if res is not None][:num_samples]

#     return results
