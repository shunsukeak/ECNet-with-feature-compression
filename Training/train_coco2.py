import argparse
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from yolov3.datasets.coco import COCODataset
from yolov3.utils import utils as utils
# from yolov3.utils.model import create_model, parse_yolo_weights
from yolov3.utils.model_edge import create_model, parse_yolo_weights #edgeモデルをインポート
# from yolov3.utils.model_edge import create_model, parse_yolo_weights2 #edgeモデルをインポート
from yolov3.models.yolov3_edge import resblock #resblockインストール

#可視化ツールいんすとーる？
#from yolov3.log import NET

#引数の定義
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    #データの引数
    parser.add_argument(
        "--dataset_dir", type=Path, default="./data/COCO",
        help="directory path to coco dataset",
    )
    #アノテーションの引数
    parser.add_argument(
        "--anno_path", type=Path, default="./data/COCO/annotations_trainval2017/annotations/instances_train2017.json",
        help="json filename",
    )
    #重みの引数（ウェイトファイルか途中まで学習したチェックポイントファイルか）
    parser.add_argument(
        "--weights", type=Path, default="./weights/yolov3.weights",
        # "--weights", type=Path, default="weights/yolov3-tiny.conv.15",
        # "--weights", type=Path, default="./weights/yolov3-tiny.weights",
        # "--weights", type=Path, default="./train_output9/yolov3-edge_267000.ckpt",
        # "--weights", type=Path, default=None,

        help="path to darknet weights file (.weights) or checkpoint file (.pth)",
    )
    #コンフィギュレーション（様々な条件設定）ファイルの引数
    parser.add_argument(
        #"--config", type=Path, default="config/yoloargs.weightsv3_coco.yaml",
        "--config", type=Path, default="./config/yolov3edge_coco.yaml",
        help="path to config file",
    )
    #GPU
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id to use")
    #トレインしたデータ（重み）のアウトプット
    parser.add_argument(
        "--save_dir", type=Path, default="train_output100",
        help="directory where checkpoint files are saved",
    )
    #何回づつでトレインした重みを出力するか
    parser.add_argument(
        "--save_interval", type=int, default=500,
        help="interval between saving checkpoints",
    )
    # fmt: on
    args = parser.parse_args()

    return args


torch.backends.cudnn.benchmark = True

# batchnormをfreeze
def remove_sequential(network):
    for layer in network.children():
        # print(type(layer))
        if type(layer) == resblock or type(layer) == torch.nn.Sequential or type(layer) ==  torch.nn.modules.container.ModuleList: # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer)
        if list(layer.children()) == []: # if leaf node, add it to list
            # print("leaf---", layer)
            if isinstance(layer, torch.nn.BatchNorm2d):
                if hasattr(layer, 'weight'):
                    layer.weight.requires_grad_(False)
                if hasattr(layer, 'bias'):
                    layer.bias.requires_grad_(False)
                # layer.training = False
                # print("bnbnbnbn------------")
                layer.eval()


def build_optimizer(config, model):
    batch_size = config["train"]["batch_size"]
    subdivision = config["train"]["subdivision"]
    n_samples_per_iter = batch_size * subdivision  # 1反復あたりのサンプル数
    momentum = config["train"]["momentum"]
    decay = config["train"]["decay"]
    base_lr = config["train"]["lr"] / n_samples_per_iter

    params = []
    for key, value in model.named_parameters():
        if "edgetail" in key:
            weight_decay = decay * n_samples_per_iter if "conv.weight" in key else 0
            print(key)
            params.append({"params": value, "weight_decay": weight_decay})

    optimizer = torch.optim.SGD(
        params,
        lr=base_lr,
    momentum=momentum,
    weight_decay=decay * n_samples_per_iter,
    )

    return optimizer


def build_scheduler(config, optimizer):
    burn_in = config["train"]["burn_in"]
    steps = config["train"]["steps"]

    def schedule(i):
        if i < burn_in:
            factor = (i / burn_in) ** 4
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01

        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    return scheduler


def repeater(dataloader):
    for loader in repeat(dataloader):
        for data in loader:
            yield data


def get_train_size():
    return np.random.randint(10, 20) * 32


def print_info(info, max_iter):
    print(f"iteration: {info['iter']}/{max_iter}", end=" ")
    print(f"image size: {info['img_size']}", end=" ")
    print(f"[Loss] total: {info['loss_total']:.2f}", end=" ")
    print(f"xy: {info['loss_xy']:.2f}", end=" ")
    print(f"wh: {info['loss_wh']:.2f}", end=" ")
    print(f"object: {info['loss_obj']:.2f}", end=" ")
    print(f"class: {info['loss_cls']:.2f}")


def main():
    args = parse_args()

    # 設定ファイルを読み込む。
    config = utils.load_config(args.config)
    img_size = config["train"]["img_size"]
    batch_size = config["train"]["batch_size"]
    subdivision = config["train"]["subdivision"]
    n_samples_per_iter = batch_size * subdivision
    max_iter = config["train"]["max_iter"]
    random_size = config["augmentation"]["random_size"]

    # Device を作成する。
    device = utils.get_device(gpu_id=args.gpu_id)
    print(device)

    # --weights にチェックポイントが指定された場合は読み込む。
    state = torch.load(args.weights) if args.weights.suffix == ".ckpt" else None
    #####

    # state = None
    # モデルを作成する。
    model = create_model(config)
    # parse_yolo_weights2(model, args.weights)
    # print(f"Darknet weights file {args.weights} loaded.")

    if state:
        # --weights に指定されたファイルがチェックポイントの場合、状態を復元する。
        # model.load_state_dict(state["model"])
        model.load_state_dict(state["model"], strict=False)
        print(f"Checkpoint file {args.weights} loaded.")
    else:
        # --weights に指定されたファイルが Darknet 形式の重みの場合、重みを読み込む。
        parse_yolo_weights(model, args.weights)
        print(f"Darknet weights file {args.weights} loaded.")
    # # else:
    # #     parse_yolo_weights2(model, args.weights)
    # #     print(f"Darknet weights file {args.weights} loaded.")

    # # parse_yolo_weights2(model, args.weights)
    # # print(f"Darknet weights file {args.weights} loaded.")

    model = torch.compile(model.to(device)).train()
    # model = create_model(config)

    # Optimizer を作成する。
    # optimizer = build_optimizer(config, model)
    optimizer = build_optimizer(config, model)

    # Scheduler を作成する。
    scheduler = build_scheduler(config, optimizer)

    start_iter = 1
    history = []

    if state:
        # --weights に指定されたファイルがチェックポイントの場合、状態を復元する。
        # Optimizer の状態を読み込む。
        optimizer.load_state_dict(state["optimizer"])
        # Scheduler の状態を読み込む。
        scheduler.load_state_dict(state["scheduler"])
        # 前回のステップの次のステップから再開する
        start_iter = state["iter"] + 1
        # 学習履歴を読み込む。
        history = state["history"]

    # Dataset を作成する。
    train_dataset = COCODataset(
        args.dataset_dir,
        args.anno_path,
        img_size=img_size,
        augmentation=config["augmentation"],
    )

    # DataLoader を作成する。
    train_dataloader = repeater(
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    )

    # チェックポイントを保存するディレクトリを作成する。
    args.save_dir.mkdir(exist_ok=True)

    remove_sequential(model.module_list_share)
    # print(model)
    scaler = torch.cuda.amp.GradScaler()

    for iter_i in range(start_iter, max_iter + 1):
        optimizer.zero_grad()
        for _ in range(subdivision):
            imgs, targets, _, _ = next(train_dataloader)
            imgs = imgs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 順伝搬する。
                loss = model(imgs, targets)

            # 逆伝搬する。
            # loss.backward()
            scaler.scale(loss).backward()

        # optimizer.step()
        # scheduler.step()

        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()
        scheduler.step()


        # 学習過程を記録する。
        info = {
            "iter": iter_i,
            "lr": scheduler.get_last_lr()[0] * n_samples_per_iter,
            "img_size": train_dataset.img_size,
            "loss_total": float(loss),
            "loss_xy": float(model.loss_dict["xy"]),
            "loss_wh": float(model.loss_dict["wh"]),
            "loss_obj": float(model.loss_dict["obj"]),
            "loss_cls": float(model.loss_dict["cls"]),
        }
        history.append(info)
        print_info(info, max_iter)
        # totaliter = [item["iter"] for item in history]
        # totalloss = [item["loss_total"] for item in history]
        # fig, ax = plt.subplots()
        # ax.plot(totaliter, totalloss, marker="o", color = "blue", linestyle = "--")
        # fig.savefig('./sample.png')
        # plt.close(fig)
        #print(model)

        # モデルの入力サイズを変更する。
        if random_size and iter_i % 10 == 0:
            train_dataset.img_size = get_train_size()
            print(f"input image size changed to {train_dataset.img_size}.")

        if iter_i % args.save_interval == 0 or iter_i == max_iter:
            model_name = config["model"]["name"]
            if iter_i == max_iter:
                state_dict = {"model": model.state_dict()}
                state_dict_path = args.save_dir / f"{model_name}_final.pth"
                history_path = args.save_dir / "history_final.csv"
            else:
                state_dict = {
                    "iter": iter_i,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                }
                state_dict_path = args.save_dir / f"{model_name}_{iter_i:06d}.ckpt"
                history_path = args.save_dir / f"history_{iter_i:06d}.csv"
                # totaliter = iter_i
                # totalloss = float(loss)
                # fig, ax = plt.subplots()
                # ax.plot(totaliter, totalloss, marker="o", color = "blue", linestyle = "--")
                # fig.savefig('./loss2.png')
                # plt.close(fig)

            # チェックポイントを保存する。
            torch.save(state_dict, state_dict_path)

            # 学習経過を保存する。
            pd.DataFrame(history).to_csv(history_path, index=False)


            print(
                f"Training state saved. checkpoints: {state_dict_path}, loss history: {history_path}."
            )


if __name__ == "__main__":
    main()
