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

NUM_EPOCHS = 20
warmup_epochs = 3
model = models()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True,
    threshold=0.01, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
y = torch.zeros([4, 3, 16, 16])
loss = nn.L1Loss()


if __name__ == '__main__':
    x = torch.ones([4, 3, 16, 16])
    metric = 0
    for i in range(1, NUM_EPOCHS):
        optimizer.zero_grad()
        out = model(x)
        loss1 = loss(out, y)
        loss1.backward()
        metric += pow(0.6, i)
        print("metric", metric)
        optimizer.step()
        print("lr", optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step(metric)






