import math

import torch
import torch.nn as nn

import torch
from torch import nn
from torchvision import models

from pathlib import Path
import sys
sys.path.append("path/to/pytorch_yolov3")
from yolov3.utils.model import create_model, parse_yolo_weights
from yolov3.utils import utils as utils

	
  
from yolov3.models.yolov3 import resblock, YOLOLayer
def remove_sequential(network):
	for layer in network.children():
		if type(layer) == torch.nn.Sequential or type(layer) ==  torch.nn.modules.container.ModuleList or type(layer)== resblock or type(layer)== YOLOLayer: # if sequential layer, apply recursively to layers in sequential layer
			remove_sequential(layer)
		if list(layer.children()) == []: # if leaf node, add it to list
			if isinstance(layer, torch.nn.BatchNorm2d):
				layer.eval()
	
	
class YOLO_feature(nn.Module):
	def __init__(self):
		super().__init__()
		config = utils.load_config(Path("/"))
		self.yolov3_model = create_model(config)		
		parse_yolo_weights(self.yolov3_model, Path(""))
		self.criterion = nn.MSELoss(reduction="mean")
		remove_sequential(self.yolov3_model)
	

	def forward(self, real, comp):
		with torch.no_grad():
			mid = self.yolov3_model(real, train_comp=True)
		
		out_net = comp(mid)

		loss = self.criterion(mid, out_net["x_hat"])

		return loss, out_net



class RateDistortionLoss(nn.Module):

	def __init__(self, lmbda=1e-2):
		super().__init__()
		self.feature_comp = YOLO_feature()
		self.lmbda = lmbda
	
	
	def forward(self, comp, target):
		N, _, H, W = target.size()
		out = {}
		num_pixels = N * H * W

		out["mse_loss"], output = self.feature_comp(target, comp)

		out["bpp_loss"] = sum(
			(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
			for likelihoods in output["likelihoods"].values()
		)

		out["loss"] = self.lmbda*500* out["mse_loss"]  + out["bpp_loss"] 
		# out["loss"] = self.lmbda*500* out["mse_loss"]
  
		
		return out
