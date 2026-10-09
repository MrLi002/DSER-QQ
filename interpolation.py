import argparse
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from model.ourModel import Model
from predict.bsergb import predict_bsergb
from predict.gopro import predict_gopro
from predict.hsergb import predict_hsergb
from predict.sunfilm import predict_snufilm
from predict.vimeo90k import predict_vimeo90k


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DSER++ checkpoint")
    parser.add_argument(
        "--dataset",
        default="bsergb",
        choices=("vimeo90k", "gopro", "snufilm", "hsergb", "bsergb"),
    )
    parser.add_argument("--qt-epa", action="store_true", help="Use QT-EPA architecture (requires QT-EPA checkpoint)")
    parser.add_argument("--bins", default=8, type=int)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--mask_patch_size", default=32, type=int)
    parser.add_argument("--checkpoints_path", required=True)
    parser.add_argument(
        "--data_root_path",
        required=True,
        help="Dataset split root; used directly by the BS-ERGB evaluator",
    )
    parser.add_argument("--save_root_path", required=True)
    return parser.parse_args()


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("The released inference pipeline requires a CUDA GPU")
    if not os.path.isfile(args.checkpoints_path):
        raise FileNotFoundError(
            "Checkpoint does not exist: {}".format(args.checkpoints_path)
        )
    if not os.path.isdir(args.data_root_path):
        raise FileNotFoundError(
            "Dataset root does not exist: {}".format(args.data_root_path)
        )

    device = torch.device("cuda")
    model = Model(args, inference_only=True)
    model.load_checkpoint(args.checkpoints_path)
    save_path = os.path.join(args.save_root_path, args.dataset)

    if args.dataset == "vimeo90k":
        predict_vimeo90k(
            model,
            args.bins,
            device,
            save_path,
            data_root=args.data_root_path,
            isSave=False,
            isTestPer=True,
            saveIndexRange=[0, 2000],
        )
    elif args.dataset == "gopro":
        predict_gopro(
            model=model,
            bins=args.bins,
            device=device,
            save_path=save_path,
            data_root=args.data_root_path,
            multis=[7, 15],
            isSave=False,
            isTestPer=False,
            saveSpecificScene=None,
        )
    elif args.dataset == "snufilm":
        predict_snufilm(
            model,
            args.bins,
            device,
            save_path,
            data_root=args.data_root_path,
            difficulties=["extreme", "hard"],
            isSave=False,
            isTestPer=False,
            saveSpecificScene=None,
        )
    elif args.dataset == "hsergb":
        predict_hsergb(
            model,
            args.bins,
            device,
            save_path,
            data_root=args.data_root_path,
            multis=[5, 7],
            isSave=False,
            isTestPer=False,
            saveSpecificScene=None,
        )
    else:
        predict_bsergb(
            model,
            args.bins,
            device,
            save_path,
            data_root=args.data_root_path,
            multis=(1, 3),
            isSave=True,
            isTestPer=False,
            saveSpecificScene=None,
        )


if __name__ == "__main__":
    main(parse_args())
