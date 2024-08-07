import sys
import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import time

from parameters import parse_args
from model import (
    Model2,
    MF_model,
    InBatchSampledModel,
    ViTURC_MODEL,
    Rec_Long_Clip,
    longclip,
    Clip4Rec,
)
from data_utils import (
    eval_model,
    train_eval,
    eval_model_clip,
    compute_sorted_indices,
    generate_texts,
)
from data_utils import (
    movielens_dataset,
    TestDataset,
    Vit_dataset,
    ClipTestDataset,
    ClipRecDataset,
)
from data_utils.utils import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_dict = {
    "ml-1m-full": Vit_dataset,
    "debug": TestDataset,
    "long_clip_debug": ClipTestDataset,
    # "long_clip": Vit_dataset,
    "long_clip": ClipRecDataset,
}

model_dict = {
    "ViTURC": ViTURC_MODEL,
    "MF": MF_model,
    "Long_Clip": Rec_Long_Clip,
    "Clip4Rec": Clip4Rec,
}


def get_dataset(args, mode):
    dataset_class = dataset_dict.get("long_clip")  # 修改以调用不同数据集类
    if dataset_class:
        return dataset_class(args, mode)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_name}")


def train(args, local_rank):
    original_dataset_name = args.dataset_name  # 保存原始数据集名称
    Log_file.info(f"sf_num: {args.sf_num}, shuffle_mode: {args.shuffle_mode}")

    if args.is_boost_experiment == True:
        Log_file.info("build boost dataset for boost experiment...")
        args.dataset_name = args.boost_dataset_name  # 修改为 boost 数据集名称
        train_dataset = get_dataset(args=args, mode="train")
    else:
        Log_file.info("build dataset...")
        train_dataset = get_dataset(args=args, mode="train")

    subset_size = int(len(train_dataset) * args.dataset_percentage)

    dataset_subset, _ = torch.utils.data.random_split(
        train_dataset, [subset_size, len(train_dataset) - subset_size]
    )

    Log_file.info("build DDP sampler for dataset...")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_subset)
    item_num = args.num_items

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2**31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    Log_file.info("build dataloader for dataset...")
    train_dl = DataLoader(
        dataset_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_reset_seed,
        pin_memory=True,
        sampler=sampler,
    )

    Log_file.info("build model...")
    if args.model_name in model_dict:
        model = model_dict[args.model_name](args).to(local_rank)
    else:
        raise ValueError(f"Model {args.model_name} not recognized")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    model_structure = str(model)
    Log_file.info(f"Model structure: {model_structure}")

    # 根据是否是boost实验决定加载权重的逻辑
    if "None" not in args.load_ckpt_name:
        if args.is_boost_experiment:
            Log_file.info("load ckpt for boost experiment...")
            args.dataset_name = original_dataset_name  # 使用原始数据集名称加载权重
            ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
            Log_file.info(f"load checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            Log_file.info("load checkpoint...")
            model.load_state_dict(checkpoint["model_state_dict"])
            Log_file.info(f"Model loaded from {ckpt_path}")
            args.dataset_name = args.boost_dataset_name  # 重新设置为boost数据集
        else:
            Log_file.info("load ckpt if not None...")
            ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            Log_file.info("load checkpoint...")
            model.load_state_dict(checkpoint["model_state_dict"])
            Log_file.info(f"Model loaded from {ckpt_path}")

        # 使用正则表达式从 load_ckpt_name 提取 start_epoch
        start_epoch_match = re.search(r"epoch-(\d+)", args.load_ckpt_name)
        if start_epoch_match:
            start_epoch = int(start_epoch_match.group(1))
        else:
            raise ValueError(
                f"Failed to extract start_epoch from {args.load_ckpt_name}"
            )

        torch.set_rng_state(checkpoint["rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        is_early_stop = True
    else:
        checkpoint = None
        ckpt_path = None
        start_epoch = 0
        is_early_stop = True

    Log_file.info("model.cuda()...")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(
        model.module.parameters(), lr=args.lr, weight_decay=args.l2_weight
    )

    if "None" not in args.load_ckpt_name:
        optimizer.load_state_dict(checkpoint["optimizer"])
        Log_file.info(f"optimizer loaded from {ckpt_path}")

    total_num = sum(p.numel() for p in model.module.parameters())
    trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info("\n")
    Log_file.info("Training...")
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    steps_for_log, _ = para_and_log(
        model,
        len(dataset_subset),
        args.batch_size,
        Log_file,
        logging_num=args.logging_num,
        testing_num=args.testing_num,
    )

    scaler = torch.cuda.amp.GradScaler()
    if "None" not in args.load_ckpt_name:
        scaler.load_state_dict(checkpoint["scaler_state"])
        Log_file.info(f"scaler loaded from {ckpt_path}")
    Log_screen.info("{} train start".format(args.label_screen))

    writer = SummaryWriter(f"runs/{args.model_name}")

    accumulation_steps = (
        args.accumulation_steps if hasattr(args, "accumulation_steps") else 1
    )

    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        Log_file.info("\n")
        Log_file.info("epoch {} start".format(now_epoch))
        Log_file.info("")
        loss, batch_index, need_break = 0.0, 1, False
        model.train()
        train_dl.sampler.set_epoch(now_epoch)
        batch_recall = []
        batch_ndcg = []
        for data in train_dl:
            if args.model_name == "ViTURC":
                matrix, pos_items, neg_items = (
                    data["matrix"],
                    data["pos_items"],
                    data["neg_items"],
                )
                matrix, pos_items, neg_items = (
                    matrix.to(local_rank),
                    pos_items.to(local_rank),
                    neg_items.to(local_rank),
                )
            elif args.model_name == "Long_Clip" or args.model_name == "Clip4Rec":
                images, texts, positive_items = (
                    data["images"],
                    data["texts"],
                    data["pos_items"],
                )
                origin_texts = texts
                images = images.to(local_rank)
                positive_items = positive_items.to(local_rank)
                batch_texts_list, batch_pos_indices_list = generate_texts(origin_texts)
                texts = longclip.tokenize(texts, truncate=True).to(local_rank)
                tokenized_batches = []
                for batch in batch_texts_list:
                    tokenized_texts = longclip.tokenize(batch, truncate=True)
                    tokenized_batches.append(tokenized_texts.to(local_rank))

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if args.model_name == "ViTURC":
                    bz_loss = model(matrix, pos_items, neg_items)
                elif args.model_name == "Long_Clip":
                    bz_loss = model(images, texts)
                elif args.model_name == "Clip4Rec":
                    bz_loss = model(
                        images,
                        tokenized_batches,
                        positive_items,
                        batch_pos_indices_list,
                    )
                loss += bz_loss.data.float() / accumulation_steps
            scaler.scale(bz_loss).backward()

            if (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(
                train_dl
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            writer.add_scalar(
                "Loss/train", bz_loss.data.float(), global_step=batch_index
            )

            if torch.isnan(loss.data):
                Log_file.info("Error: Loss NaN detected!")
                sys.exit(1)
            if batch_index % steps_for_log == 0:
                Log_file.info(
                    "cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}".format(
                        batch_index,
                        batch_index * args.batch_size,
                        loss.data / batch_index,
                        loss.data,
                    )
                )
            batch_index += 1
            # 关于训练集指标的评估
            model.eval()
            with torch.no_grad():
                preds = compute_sorted_indices(origin_texts, images, model, local_rank)
                iter_recall, iter_ndcg = train_eval(
                    preds, positive_items, item_num, local_rank
                )
                batch_recall.append(iter_recall)
                batch_ndcg.append(iter_ndcg)

            model.train()

        if not need_break:
            Log_file.info("")
            (
                max_eval_value,
                max_epoch,
                early_stop_epoch,
                early_stop_count,
                need_break,
                need_save,
            ) = run_eval(
                args,
                now_epoch,
                max_epoch,
                early_stop_epoch,
                max_eval_value,
                early_stop_count,
                model,
                args.val_batch_size,
                item_num,
                args.mode,
                is_early_stop,
                local_rank,
                writer,
            )
            model.train()
            if need_save and dist.get_rank() == 0:
                save_model(
                    now_epoch,
                    model,
                    model_dir,
                    optimizer,
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(),
                    scaler,
                    Log_file,
                )

        avg_recall = sum(batch_recall) / len(batch_recall) if batch_recall else 0.0
        avg_ndcg = sum(batch_ndcg) / len(batch_ndcg) if batch_ndcg else 0.0

        writer.add_scalar("Recall/train", avg_recall, global_step=now_epoch)
        writer.add_scalar("NDCG/train", avg_ndcg, global_step=now_epoch)
        metrics_to_log = [avg_recall, avg_ndcg]
        Log_file.info(
            "Training Epoch {} methods  {}".format(
                now_epoch, "\t".join(["Recall{}".format(5), "nDCG{}".format(5)])
            )
        )
        Log_file.info(
            "Training Epoch {} results  {}".format(
                now_epoch,
                "\t".join(["{:0.5f}".format(i * 100) for i in metrics_to_log]),
            )
        )

        Log_file.info("")
        next_set_start_time = report_time_train(
            batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file
        )
        Log_screen.info(
            "{} training: epoch {}/{}".format(args.label_screen, now_epoch, args.epoch)
        )

    if dist.get_rank() == 0:
        save_model(
            now_epoch,
            model,
            model_dir,
            optimizer,
            torch.get_rng_state(),
            torch.cuda.get_rng_state(),
            scaler,
            Log_file,
        )
    Log_file.info("\n")
    Log_file.info("%" * 90)
    Log_file.info(
        " max eval Hit10 {:0.5f}  in epoch {}".format(max_eval_value * 100, max_epoch)
    )
    Log_file.info(" early stop in epoch {}".format(early_stop_epoch))
    Log_file.info("the End")
    Log_screen.info(
        "{} train end in epoch {}".format(args.label_screen, early_stop_epoch)
    )


def run_eval(
    args,
    now_epoch,
    max_epoch,
    early_stop_epoch,
    max_eval_value,
    early_stop_count,
    model,
    batch_size,
    item_num,
    mode,
    is_early_stop,
    local_rank,
    writer,
):
    eval_start_time = time.time()
    Log_file.info("Validating...")
    if args.model_name == "ViTURC":
        valid_Recall20 = eval_model(
            model, batch_size, args, item_num, Log_file, mode, local_rank
        )
    elif args.model_name == "Long_Clip" or args.model_name == "Clip4Rec":
        valid_Recall20 = eval_model_clip(
            model, batch_size, args, item_num, Log_file, mode, local_rank
        )
    report_time_eval(eval_start_time, Log_file)
    Log_file.info("")
    writer.add_scalar("Recall/valid", valid_Recall20, global_step=now_epoch)
    need_break = False
    need_save = False
    if valid_Recall20 > max_eval_value:
        max_eval_value = valid_Recall20
        max_epoch = now_epoch
        early_stop_count = 0
        need_save = True
    else:
        early_stop_count += 1
        if early_stop_count > 6:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return (
        max_eval_value,
        max_epoch,
        early_stop_epoch,
        early_stop_count,
        need_break,
        need_save,
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    setup_seed(12345)
    gpus = torch.cuda.device_count()

    dir_label = args.dataset_name
    time_run = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    args.label_screen = args.label_screen + time_run

    # 添加新的超参数到 log_paras，并放到靠前的位置
    log_paras = (
        f"_boost_{args.is_boost_experiment}"
        f"_boost_ds_{args.boost_dataset_name}"
        f"_ds_{args.dataset_name}"
        f"_model_{args.model_name}"
        f"_bs_{args.batch_size}"
        f"_lr_{args.lr}_L2_{args.l2_weight}"
    )

    model_dir = os.path.join("./checkpoint_" + dir_label, "cpt_" + log_paras)

    Log_file, Log_screen = setuplogger(
        dir_label, log_paras, time_run, args.mode, dist.get_rank()
    )

    Log_file.info(args)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if "train" in args.mode:
        train(args, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info(
        "##### (time) all: {} hours {} minutes {} seconds #####".format(
            hour, minu, secon
        )
    )
