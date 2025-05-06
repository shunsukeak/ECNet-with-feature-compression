import argparse
from pathlib import Path

import torch
from yolov3.utils import utils as utils
from yolov3.utils.coco_evaluator import *
from yolov3.utils.model import create_model, parse_yolo_weights, parse_yolo_weights_edge

import matplotlib.pyplot as plt
import numpy as np
#################

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=Path, required=True, help="directory path to coco dataset"
    )
    parser.add_argument("--anno_path", type=Path, required=True, help="json filename")
    parser.add_argument(
        "--weights",
        type=Path,
        help="path to weights file",
    )
    parser.add_argument(
        "--config",
        type=Path, 
        help="path to config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config_edge = utils.load_config("config/")
    config_cloud = utils.load_config("config/")
    img_size = config_edge["test"]["img_size"]
    batch_size = 1

    device = utils.get_device(gpu_id=args.gpu_id)

    model_edge = create_model(config_edge)

    state = torch.load(args.weights)

    t = 0
    for module in model_edge.module_list_dummy:
        if t >= 9:
            model_edge.module_list_edgetail.append(model_edge.module_list_dummy[t])
        t+=1 


    model_edge.load_state_dict(state["model"], strict=True)
    print(f"Checkpoint file {args.weights} loaded.")

    model_edge = model_edge.to(device).eval()

    model_cloud = create_model(config_cloud)
    parse_yolo_weights(model_cloud, Path("weights/yolov3.weights"))
    model_cloud = model_cloud.to(device).eval()


    from model import Featurecomp_FactorizedPrior
    comp = Featurecomp_FactorizedPrior()
    state = torch.load("/")
    comp.load_state_dict(state["state_dict"])
    comp = comp.to(device)

    evaluator = COCOEvaluator_edgecloud_new(
        args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    )
    
    percent_list = [0, 1, 10, 20, 30] 

    thresarea_list = [0, 250, 300, 320, 350, 1000] 






    ap50_data = []
    parameters_data =[]
    bpp_data = []

    for i in range(len(percent_list)):
        x = percent_list[i]
        ap50_values = []
        bpp_values = []
        parameters_values = []
        for k in range(len(thresarea_list)):
            z=x
            y = thresarea_list[k]
            ap50_95, ap50, bpp, parameters = evaluator.evaluate(model_edge, model_cloud, threspercent=z, thresarea=y, comp=comp)
            ap50_values.append(ap50)
            bpp_values.append(bpp)
            parameters_values.append(parameters)
        ap50_data.append(ap50_values)
        bpp_data.append(bpp_values)
        parameters_data.append(parameters_values)

    ap50_data = [torch.tensor(data) for data in ap50_data]
    bpp_data = [torch.tensor(data1) for data1 in bpp_data]
    parameters_data = [torch.tensor(data2) for data2 in parameters_data]

    plt.figure()
    for i in range(len(percent_list)):
        x = percent_list[i]
        color = cmap(i)
        
        bpp_data[i] = bpp_data[i].to('cpu').detach().numpy().copy()
        ap50_data[i] = ap50_data[i].to('cpu').detach().numpy().copy()
        plt.plot(bpp_data[i], ap50_data[i], label=f'x={x}', color = color, marker='o', linestyle='-')
        
    previous_bpp = []
    previous_map = []
    plt.plot(previous_bpp, previous_map, label = 'previous', color='blue', marker='o', linestyle='--')
    
    onlycloud_bpp = []
    onlycloud_map = []
    plt.plot(onlycloud_bpp, onlycloud_map, label = 'cloud only', color='red', marker='o', linestyle='--')

    plt.xlabel('bpp')
    plt.ylabel('mAP')
    plt.legend()

    plt.savefig("./bpp_ap50.png")

    plt.figure()
    for i in range(len(percent_list)):
        x = percent_list[i]
        color = cmap(i)        
        plt.plot(bpp_data[i], ap50_data[i], label=f'x={x}', color = color, marker='o', linestyle='-')
        
    previous_bpp = []
    previous_map = []
    plt.plot(previous_bpp, previous_map, label = 'previous', color='blue', marker='o', linestyle='--')
    
    plt.xlabel('bpp')
    plt.ylabel('mAP')
    plt.legend()

    plt.savefig("./bpp_ap50_previousandproposed.png")

    plt.figure()
    for i in range(len(percent_list)):
        x = percent_list[i]
        color = cmap(i)
        plt.plot(bpp_data[i], ap50_data[i], label=f'x={x}', color = color, marker='o', linestyle='-')
        
    onlycloud_bpp = []
    onlycloud_map = []
    plt.plot(onlycloud_bpp, onlycloud_map, label = 'only cloud', color='red', marker='o', linestyle='--')
    
    plt.xlabel('bpp')
    plt.ylabel('mAP')
    plt.legend()

    plt.savefig("./bpp_ap50_previousandonlycloud.png")

    plt.figure()
    for i in range(len(percent_list)):
        x = percent_list[i]
        color = cmap(i)
        parameters_data[i] = parameters_data[i].to('cpu').detach().numpy().copy()
        plt.plot(parameters_data[i], ap50_data[i], label=f'x={x}', color = color, marker='o', linestyle='-')
        

    plt.plot(10.0168,0.368, color='magenta', label='edge only', marker='*', markersize = 16)
    plt.plot(65.718237, 0.605, color='cyan', label='cloud only', marker='*', markersize = 16)

    plt.xlabel('parameters[M]')
    plt.ylabel('mAP')
    plt.legend()

    plt.savefig("./bpp_parameters.png")

    print("ap50_data", ap50_data)
    print("bpp_data", bpp_data)
    print("parameters_data", parameters_data)


if __name__ == "__main__":
    main()


