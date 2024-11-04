# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import sys
sys.path.append("/")

from compressai.datasets import ImageFolder

import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from loss import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

import argparse
import json
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ImageDataset(Dataset):
    def __init__(self, image_dir, train=True):
        self.image_dir = image_dir
        self.mask_dir = image_dir.replace("2017", "2017_mask")
        self.image_list = os.listdir(self.image_dir)
        self.N = 256
        if train:
            transform = [
                A.RandomCrop(self.N, self.N),
                A.HorizontalFlip(),
                A.Normalize(mean=(0,0,0), std=(1,1,1)),
                ToTensorV2()
            ]
        else:
            transform = [
                A.CenterCrop(self.N, self.N),
                A.Normalize(mean=(0,0,0), std=(1,1,1)),
                ToTensorV2()
            ]
        self.transform = A.Compose(
            transform,
            additional_targets={'image0': 'image'}
                )
        self.resize =  A.Compose(
            [A.Resize(height=self.N, width=self.N)],
            additional_targets={'image0': 'image'}
                )

        
    def __len__(self):
        return len(self.image_list)

    

    def __getitem__(self, id):
        filename = self.image_list[id]
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if min(image.shape[:2]) < self.N:
            augmented = self.resize(image=image)
            image = augmented['image']
        augmented = self.transform(image=image)
        image = augmented['image']
        return image



class AverageMeter:
	"""Compute running average."""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



def configure_optimizers(net, args):
	"""Separate parameters for the main optimizer and the auxiliary optimizer.
	Return two optimizers"""
	conf = {
		"net": {"type": "Adam", "lr": args.learning_rate},
		"aux": {"type": "Adam", "lr": args.aux_learning_rate},
	}
	optimizer = net_aux_optimizer(net, conf)
	return optimizer["net"], optimizer["aux"]


def train_one_epoch(
	model, criterion, train_dataloader, optimizer, aux_optimizer, clip_max_norm):
	
	model.train()
	device = next(model.parameters()).device

	downsample = nn.Upsample(scale_factor=1/2, mode='bilinear')

	ITERS_TO_ACCUMULATE = 1

	for i, d in enumerate(tqdm(train_dataloader)):
		optimizer.zero_grad()
		aux_optimizer.zero_grad()

		d = d.to(device) 

		out_criterion =  criterion(model, d)
  
		out_criterion["loss"].sum().backward()
   
	
		if clip_max_norm > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
		optimizer.step()

			
		aux_loss = model.aux_loss()

		aux_loss.backward()
		aux_optimizer.step()
  
	


def test_epoch(epoch, test_dataloader, model, criterion, r, save):
	model.eval()
	device = next(model.parameters()).device

	loss = AverageMeter()
	bpp_loss = AverageMeter()
	mse_loss = AverageMeter()
	aux_loss = AverageMeter()

	with torch.no_grad():
		for d in tqdm(test_dataloader):
			d = d[0]
			d = d.to(device)

			out_criterion =  criterion(model, d)
   
			aux_loss.update(model.aux_loss())
			bpp_loss.update(out_criterion["bpp_loss"].mean())
			loss.update(out_criterion["loss"].mean())
			mse_loss.update(out_criterion["mse_loss"].mean())


	print(
		f"Test epoch {epoch}: Average losses:"
		f"\tLoss: {loss.avg:.5f} |"
		f"\tMSE loss: {mse_loss.avg:.5f} |"
		f"\tBpp loss: {bpp_loss.avg:.5f} |"
		f"\tAux loss: {aux_loss.avg:.5f}\n"
	)

	return loss.avg, bpp_loss.avg


def save_checkpoint(state, save, e, filename="checkpoint.pth.tar"):
	torch.save(state, save + "/" + e+"_"+filename)


def parse_args(argv):
	parser = argparse.ArgumentParser(description="Example training script.")
	parser.add_argument(
		"-m",
		"--model",
		default="bmshj2018-factorized",
		choices=image_models.keys(),
		help="Model architecture (default: %(default)s)",
	)
	parser.add_argument(
		"-d", "--dataset", type=str, default="/home/shunsukeakamatsu/Documents/pytorch_yolov3/data/COCO", help="Training dataset"
	)
	parser.add_argument(
		"-e",
		"--epochs",
		default=50,
		type=int,
		help="Number of epochs (default: %(default)s)",
	)
	parser.add_argument(
		"-lr",
		"--learning-rate",
		default=1e-4,
		type=float,
		help="Learning rate (default: %(default)s)",
	)
	parser.add_argument(
		"-n",
		"--num-workers",
		type=int,
		default=4,
		help="Dataloaders threads (default: %(default)s)",
	)
	parser.add_argument(
		"--lambda",
		dest="lmbda",
		type=float,
		default=0.01,
		help="Bit-rate distortion parameter (default: %(default)s)",
	)
	parser.add_argument(
		"--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
	)
	parser.add_argument(
		"--test-batch-size",
		type=int,
		default=8,
		help="Test batch size (default: %(default)s)",
	)
	parser.add_argument(
		"--aux-learning-rate",
		type=float,
		default=1e-3,
		help="Auxiliary loss learning rate (default: %(default)s)",
	)
	parser.add_argument(
		"--patch-size",
		type=int,
		nargs=2,
		default=(256, 256),
		help="Size of the patches to be cropped (default: %(default)s)",
	)
	parser.add_argument("--cuda", type=int, default=1, help="Set random seed for reproducibility")

	parser.add_argument("--save", type=str, default="output/feature_comp/lambda0.01", help="Path to a checkpoint")
	parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
	parser.add_argument(
		"--clip_max_norm",
		default=1.0,
		type=float,
		help="gradient clipping max norm (default: %(default)s",
	)
	parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
	args = parser.parse_args(argv)
	return args


def main(argv):
	torch.manual_seed(42)
	np.random.seed(42)
	args = parse_args(argv)

	os.makedirs(args.save, exist_ok=True)
	os.makedirs(args.save+"/images", exist_ok=True)

	
	train_dataset = ImageDataset(args.dataset + "/train2017", train=True)
	
 
	from yolov3.datasets.coco import COCODataset

	from pathlib import Path
	from yolov3.utils.coco_evaluator import COCOEvaluator, COCOEvaluator_featurecomp

	from yolov3.utils.model import create_model, parse_yolo_weights
	from yolov3.utils import utils as utils
	

	test_dataset = COCODataset(
		Path("/data/COCO"),
			Path("/annotations/instances_val5k.json"),
		   img_size=512, 
	)

	device = "cuda:" + str(args.cuda) 

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		shuffle=True,
		pin_memory=True,
	)

	test_dataloader = DataLoader(
		test_dataset,
		batch_size=1,
		num_workers=args.num_workers,
		shuffle=False,
		pin_memory=True,
	)


	evaluator = COCOEvaluator_featurecomp(
			Path("/data/COCO"),

			Path("/annotations/instances_val5k.json"),

		   img_size=512, 
			batch_size=8
		)
	config = utils.load_config(Path("/config/yolov3_coco.yaml"))
	yolov3_model = create_model(config).to(device)
	parse_yolo_weights(yolov3_model, Path("/weights/yolov3.weights"))



	from model import Featurecomp_FactorizedPrior
	import compressai 


	net = Featurecomp_FactorizedPrior()

	net = net.to(device)


	optimizer, aux_optimizer = configure_optimizers(net, args)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40,gamma=0.1)
	criterion = RateDistortionLoss(lmbda=args.lmbda).to(device)
 

	
	last_epoch = 0
	if args.checkpoint:  # load from previous checkpoint
		print("Loading", args.checkpoint)
		checkpoint = torch.load(args.checkpoint, map_location=device)
		last_epoch = checkpoint["epoch"] + 1
		net.load_state_dict(checkpoint["state_dict"],strict=False)
		optimizer.load_state_dict(checkpoint["optimizer"])
		aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
		lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


	best_loss = float("inf")
	for epoch in range(last_epoch, args.epochs):
		print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
		train_one_epoch(
				net,
				criterion,
				train_dataloader,
				optimizer,
				aux_optimizer,
				args.clip_max_norm,
		)
		loss, bpp = test_epoch(epoch, test_dataloader, net, criterion, 1, args.save)
		ap50_95, ap50_0 = evaluator.evaluate(yolov3_model, net.to(device))
		print(ap50_95, ap50_0)
		

		lr_scheduler.step()


		if args.save != "nosave":
			# if args.save:
			save_checkpoint(
				{
					"epoch": epoch,
					"state_dict": net.state_dict(),
					"loss": loss,
				
					"ap50_0": ap50_0,
					"ap50_95": ap50_95,
					"bpp": bpp,
	
					"optimizer": optimizer.state_dict(),
					"aux_optimizer": aux_optimizer.state_dict(),
					"lr_scheduler": lr_scheduler.state_dict(),
				},
				args.save,
				str(epoch)
			)


if __name__ == "__main__":
	main(sys.argv[1:])