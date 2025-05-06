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
# from compressai.losses import RateDistortionLoss
from loss import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

import argparse
import json
import os
from collections import OrderedDict

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

def parse_args(argv):
	parser = argparse.ArgumentParser(description="Example training script.")
	parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
	args = parser.parse_args(argv)
	return args


def main(argv):
	import os
	torch.manual_seed(42)
	np.random.seed(42)
	args = parse_args(argv)
 
	os.makedirs(args.checkpoint+"/record", exist_ok=True)

	
	best_loss = float("inf")

	import os, re
	dir_path = args.checkpoint
	ckpts = [
		 f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
	]

	ckpts = sorted(ckpts, key=lambda s: int(re.search(r'\d+', s).group()))

	for ckpt in ckpts:
		checkpoint = torch.load(args.checkpoint + "/" + ckpt)
  
		loss_0 = checkpoint["loss"]
		ap50_95 = checkpoint["ap50_95"]
  
		print("epoch", checkpoint["epoch"]+1, ", loss", loss_0.item(), ", mAP", ap50_95, ", BPP", checkpoint["bpp"].item())

		
if __name__ == "__main__":
	main(sys.argv[1:])