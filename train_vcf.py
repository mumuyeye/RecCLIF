import os

logging_num = 4
testing_num = 1

mode = "train"
model_name = "Clip4Rec"
# lastfm Games-6k
dataset_name = "lastfm"
base_model = "ViT-B/16"

epoch = 40
# epoch-49.pt
load_ckpt_name = None
# freeze_params = False

is_boost_experiment = False  # 添加这个参数
boost_dataset_name = "lastfm"  # 添加这个参数

l2_weight_list = [0.03]
batch_size_list = [16]
lr_list_ct = [1e-4]

for l2_weight in l2_weight_list:
    for batch_size in batch_size_list:
        for lr in lr_list_ct:
            label_screen = "bs{}_lr{}_L2{}".format(batch_size, lr, l2_weight)
            run_py = f"torchrun --nproc_per_node 2 --master_port 29502 run_vcf.py \
                            --mode {mode} --load_ckpt_name {load_ckpt_name} --label_screen {label_screen} \
                            --logging_num {logging_num} --testing_num {testing_num} --l2_weight {l2_weight} \
                            --batch_size {batch_size} --lr {lr} --epoch {epoch} --model_name {model_name} \
                            --base_model {base_model} --dataset_name {dataset_name} \
                            --is_boost_experiment {is_boost_experiment} --boost_dataset_name {boost_dataset_name}"
            os.system(run_py)
