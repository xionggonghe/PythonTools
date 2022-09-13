import torch
import torch.optim as optim
import torch.nn as nn

class models(nn.Module):
    def __init__(self):
        super(models, self).__init__()
        self.con = nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        x = self.con(x)
        return x

NUM_EPOCHS = 30
warmup_epochs = 3
model = models()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
y = torch.zeros([4, 3, 16, 16])
loss = nn.L1Loss()

if __name__ == '__main__':
    x = torch.ones([4, 3, 16, 16])
    for i in range(NUM_EPOCHS):
        optimizer.zero_grad()
        out = model(x)
        loss1 = loss(out, y)
        loss1.backward()
        optimizer.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()




