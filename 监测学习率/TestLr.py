import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class models(nn.Module):
    def __init__(self):
        super(models, self).__init__()
        self.con = nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1),
                                 nn.Conv2d(3, 3, 3, 1, 1),
                                 nn.Conv2d(3, 3, 3, 1, 1),
                                 nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        x = self.con(x)
        return x

NUM_EPOCHS = 300
# warmup_epochs = 150
model = models()



# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)

# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS - warmup_epochs, eta_min=1e-7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=10, threshold=0.3,threshold_mode='abs', eps=1e-8)

y = torch.zeros([4, 3, 16, 16])
loss = nn.L1Loss()

writer = SummaryWriter('runs/experiment_1')

if __name__ == '__main__':
    x = torch.ones([4, 3, 16, 16])
    for i in range(NUM_EPOCHS):
        optimizer.zero_grad()
        out = model(x)
        loss1 = loss(out, y)

        loss1.backward()
        p = 1 / loss1
        optimizer.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step(p)

        writer.add_scalar('lr',
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          i)





