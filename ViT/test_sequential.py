import vdp_packed as vdp
import torch
from effnetv2 import effnetv2_s
from effnetv2_vdp import effnetv2_s as effnetv2_s_vdp

# effnetv2 = effnetv2_s()
effnetv2_vdp = effnetv2_s_vdp()

rand_input = torch.randn(1, 3, 224, 224)

# effnetv2(rand_input)
effnetv2_vdp(rand_input)