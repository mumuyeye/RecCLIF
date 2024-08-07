from data_utils.utils import *

# note dataset:nums and items
# ml-1m: 6034, 3125 -> 6048,3136
# m1-100k: 601, 1955 -> 672, 2016


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument(
        "--dataset_name", type=str, default="lastfm", help="dataset name-A"
    )
    parser.add_argument("--rank_model_name", type=str, default="Rank")
    parser.add_argument("--candidate_model_name", type=str, default="random")
    parser.add_argument("--user_model_name", type=str, default="MF")
    parser.add_argument("--p", type=str, default="", help="pre-trained model path")
    parser.add_argument(
        "--long_clip_path",
        type=str,
        default="/home/lyn/resys_baselines/Long-CLIP/checkpoints/longclip-B.pt",
        help="pre-trained long-clip path",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="/home/lyn/data_generate/split1/listening_history_train.csv",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="/home/lyn/data_generate/split1/listening_history_val.csv",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/home/lyn/data_generate/split1/listening_history_test.csv",
    )
    parser.add_argument("--sf_num", type=int, default=4)
    parser.add_argument("--shuffle_mode", type=str, default="rows")
    parser.add_argument("--num_users", type=int, default=224)
    parser.add_argument("--num_items", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=1.0,
        help="Percentage of dataset to load",
    )
    parser.add_argument(
        "--base_model", type=str, default="ViT-B/16"
    )  # base_model的选择
    parser.add_argument("--log_scale", type=float, default=1)  # log_scale=4.6052

    parser.add_argument("--candidate_num", type=int, default=20)
    parser.add_argument("--similar_user_num", type=int, default=4)
    parser.add_argument("--interaction", type=int, default=50)
    parser.add_argument("--groud_truth", type=bool, default=True)

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--l2_weight", type=float, default=1e-5)
    parser.add_argument(
        "--freeze_params",
        type=bool,
        default=False,
        help="Whether to freeze specific model parameters",
    )
    parser.add_argument("--accumulation_steps", type=int, default=0.7)

    # ============== model parameters ==============
    # model choice
    parser.add_argument("--model_name", type=str, default="ViTURC")
    # vcf parameters
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=328)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=472)

    # subvcf parameters
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--sub_dim", type=int, default=328)
    parser.add_argument("--sub_dropout", type=float, default=0.5)
    parser.add_argument("--sub_depth", type=int, default=1)
    parser.add_argument("--sub_heads", type=int, default=16)
    parser.add_argument("--sub_mlp_dim", type=int, default=472)

    parser.add_argument("--mlp_layers", type=int, default=3)
    # MF parameter
    parser.add_argument("--factor", type=int, default=1024)
    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--load_ckpt_name", type=str, default="None")
    parser.add_argument("--label_screen", type=str, default="None")
    parser.add_argument("--logging_num", type=int, default=8)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument(
        "--is_boost_experiment",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to run the boost experiment, choose 'True' or 'False'",
    )

    parser.add_argument(
        "--boost_dataset_name",
        type=str,
        default="None",
        help="The dataset name for the boost experiment",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
