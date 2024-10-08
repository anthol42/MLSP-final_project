import matplotlib.pyplot as plt
import numpy as np
import torch

a = torch.tensor([1, 2, 3, 4])
print(a.unsqueeze(0).expand(10, -1).reshape(-1))