import torch
from torchvision.transforms import ToPILImage

show = ToPILImage() # 可以把Tensor转成Image，方便可视化

x = torch.randn(3, 500, 500)
show(x).show()