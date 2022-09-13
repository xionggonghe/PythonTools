import torch
import math
import torch.optim as optim
from TestLr import models
import torch.nn as nn


eval_step = 5
NUM_EPOCHS = 30

steps = eval_step * NUM_EPOCHS
T = steps
lr = 0.01

def lr_schedule_cosdecay(t,T,init_lr=lr):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr

model = models()
optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=lr, betas = (0.9, 0.999), eps=1e-08)

y = torch.zeros([4, 3, 16, 16])
loss = nn.L1Loss()

if __name__ == '__main__':
    x = torch.ones([4, 3, 16, 16])
    for index in range(NUM_EPOCHS):
        for step in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss1 = loss(out, y)
            loss1.backward()
            lr = lr_schedule_cosdecay(step, T)
            print("lr:", lr)


















