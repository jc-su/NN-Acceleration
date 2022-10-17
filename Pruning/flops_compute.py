import torch
from torchvision.models import resnet18
import numpy as np

# from pthflops import count_ops
raw_params = 11181.642
print("Round Params[K] Reduced[%]")
# Create a network and a corresponding input
for idx in range(1, 12):
    device = 'cuda:0'
    model = torch.load("l2_weights/resnet18-round{}.pth".format(idx)).to(device)
    inp = torch.rand(1,3,32,32).to(device)

    # Count the number of FLOPs
    params = sum([np.prod(p.size()) for p in model.parameters()])/ 1e3

    print(idx, params , int((raw_params-params)/raw_params*10000)/100)
    # print(count_ops(model, inp))