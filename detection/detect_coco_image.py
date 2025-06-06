import argparse
from pathlib import Path

import os
from PIL import Image
from yolov3.detector import Detector, Detector2, Detector3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument(
        "--input", type=Path, default="./zu5.jpg",
        help="image path or directory path which contains images to infer",
    )
    parser.add_argument(
        "--output", type=Path, default="output_detected_edge",
        help="directory path to output detection results",
    )
    parser.add_argument(
        "--weights", type=Path, 
        default="./weights/yolov3.weights",
        # default="./train_output_test/ckpt/yolov3-edge_000300.ckpt",
        help="path to weights file",
    )
    parser.add_argument(
        "--config", type=Path, 
        default="config/yolov3_coco.yaml",
        # default="config/yolov3edge_coco.yaml",

        help="path to config file",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id to use")
    # fmt: on
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    detector = Detector(args.config, args.weights, args.gpu_id)
    # detector = Detector2(args.config, args.weights, args.gpu_id)


    # 検出する。
    detections, img_paths = detector.detect_from_path(args.input)

    # feature = detector.feature(args.input, args.config)

    # 検出結果を出力する。
    args.output.mkdir(exist_ok=True)
    for detection, img_path in zip(detections, img_paths):
        print(img_path)
        for box in detection:
            print(
                # f"{box['class_name']} {box['confidence']:.0%} "
                f"({box['x1']:.0f}, {box['y1']:.0f}, {box['x2']:.0f}, {box['y2']:.0f})"
            )

        # 検出結果を画像に描画して、保存する。
        img = Image.open(img_path)
        detector.draw_boxes(img, detection)
        img.save(args.output / Path(img_path).name)

    


if __name__ == "__main__":
    main()
