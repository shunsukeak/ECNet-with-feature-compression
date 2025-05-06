import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
from pathlib import Path

import torch
from yolov3.datasets.imagefolder import ImageFolder, ImageList
from yolov3.datasets.video import Video
from yolov3.utils import utils as utils1
from yolov3.utils import vis_utils as vis_utils
from yolov3.utils.model import create_model, parse_yolo_weights

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open(str('fig.jpg'))
plt.imshow(image)
config_path = 'config/yolov3_coco.yaml'
weights_path = './weights/yolov3.weights'
config_path = Path(config_path)
weights_path = Path(weights_path)

config = utils1.load_config(config_path)
class_names = utils1.load_classes(
    config_path.parent / config["model"]["class_names"]
)
device = utils1.get_device(gpu_id=0)
model = create_model(config)
parse_yolo_weights(model, weights_path)
print(f"Darknet format weights file loaded. {weights_path}")

# from torchsummary import summary
# summary(model, input_size=[[3, 416, 416]])

class SaveOutput:
    def __init__(self, model, target_layer):  
        self.model = model
        self.layer_output = []
        self.layer_grad = []
        
        self.feature_handle = target_layer.register_forward_hook(self.feature)
        self.grad_handle = target_layer.register_forward_hook(self.gradient)

    def feature(self, model, input, output):
         activation = output
         self.layer_output.append(activation.to("cpu").detach())

    def gradient(self, model, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return 

        def _hook(grad): 
            self.layer_grad.append(grad.to("cpu").detach())

        output.register_hook(_hook) 

    def release(self):
        self.feature_handle.remove()
        self.grad_handle.remove()

def find_conv_layers(module):
    conv_layers = []
    conv_weights = []
    for layer in module.children():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append(layer)
            conv_weights.append(layer.weight)
        elif isinstance(layer, torch.nn.Sequential):
            child_conv_layers, child_conv_weights = find_conv_layers(layer)
            conv_layers.append(child_conv_layers)
            conv_weights.append(child_conv_weights)
    return conv_layers, conv_weights

conv2d_layers, conv2d_weights = find_conv_layers(model.module_list)

print("Conv2d Layers:")
print(conv2d_layers)
print("Conv2d Weights:")
print(conv2d_weights)

print(len(conv2d_layers))
print(len(conv2d_weights))


model = model.to(device).eval()
#we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []# get all the model children as list
model_children = list(model.module_list.children())#counter to keep count of the conv layers
counter = 0 #append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

outputs = []
names = []
print(conv2d_layers[0:])
for layer in conv2d_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)

save = SaveOutput(model.module_list, conv2d_layers[0])
intermediate = save.layer_output[0].squeeze(0).numpy()

fig=plt.figure(figsize=(40,20))
for i,im in enumerate(intermediate):
    ax1=fig.add_subplot(4,8,i+1)
    ax1.imshow(im,'gray')

