import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

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

NUM_EPOCHS = 200
# warmup_epochs = 150
model = models()



optimizer = optim.Adam(model.parameters(), lr=1e-4)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
#
# scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates


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


