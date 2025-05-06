from compressai.models.utils import conv, deconv
from compressai.models.google import FactorizedPrior
import torch.nn as nn
import torch
from compressai.layers import *
from compressai.models import (
	Cheng2020Anchor,
	Cheng2020Attention,
	FactorizedPrior,
	FactorizedPriorReLU,
	JointAutoregressiveHierarchicalPriors,
	MeanScaleHyperprior,
	ScaleHyperprior,
)



class Featurecomp_FactorizedPrior(FactorizedPrior):
	def __init__(self, N=192, M=128):
		super().__init__(N=N, M=M)

		self.g_a = nn.Sequential(
            conv(256, N, stride=1),
            GDN(N),
            conv(N, M, stride=2),
            # GDN(N),
            # conv(N, N, stride=1),
            # GDN(N),
            # conv(N, M,stride=1),
        )

		self.g_s = nn.Sequential(
			# deconv(M, N, stride=1),
			# GDN(N, inverse=True),
			# deconv(N, N, stride=1),
			# GDN(N, inverse=True),
			deconv(M, N, stride=2),
			GDN(N, inverse=True),
			deconv(N, 256, stride=1),
		)
 
		
	def forward(self, x):
		y = self.g_a(x)
		y_hat, y_likelihoods = self.entropy_bottleneck(y)
		x_hat = self.g_s(y_hat)

		return {
			"x_hat": x_hat,
			"likelihoods": {
				"y": y_likelihoods,
			},
		}
	
model = Featurecomp_FactorizedPrior()
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of compression model parameters: {total_params}")