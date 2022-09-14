import torch
import torch.optim as optim
import torch.nn as nn
import PIL

import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

'''##############################################搭建网络模型##########################################################################'''
class models(nn.Module):
    def __init__(self):
        super(models, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4), # in_channels, out_channels,kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 这⾥全连接层的输出个数⽐LeNet中的⼤数倍。使⽤丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3),
        )

    def forward(self, img):
        feature = self.conv(img)
        mid_fea = feature.flatten(1)
        output = self.fc(mid_fea)
        return output


'''##############################################加载数据#########################################################################'''

batch_size = 32
# 读取数据集
transformations = transforms.Compose([
    # transforms.RandomResizedCrop(320),
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.ImageFolder("../dataset/train", transform=transformations)
test_set = torchvision.datasets.ImageFolder("../dataset/test", transform=transformations)
print(train_set.classes)  #根据分的文件夹的名字来确定的类别

train_size = int(0.8 * len(train_set))
valid_size = len(train_set) - train_size
train_set, valid_dataset = torch.utils.data.random_split(train_set, [train_size, valid_size]) #划分验证数据集

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle = True, num_workers=0)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=16, num_workers=0)


NUM_EPOCHS = 30
model = models()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
loss = nn.CrossEntropyLoss()




'''
TensorBoard
运行命令:   tensorboard --logdir=runs
'''
image = torchvision.transforms.functional.to_tensor(PIL.Image.open("./win.png").convert("RGB")) # 两种方法是一样的
image = torch.unsqueeze(image, dim=0)
# 定义文件和类
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# 将图片写入文件
img_grid = torchvision.utils.make_grid(image)
writer.add_image('win', img_grid)

# 将模型写入文件
writer.add_graph(model, image)
writer.close()


'''##############################################训练网络模型##########################################################################'''
if __name__ == '__main__':
    print("Running AlexNet : ")
    for epoch in range(NUM_EPOCHS):
        acc, all = 0.0, 0
        AllLoss = 0
        i = 0
        for X, y in trainloader:
            i += 1
            optimizer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            AllLoss += l.item()
            l.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                all += y.shape[0]
                if i%5 == 0:
                    writer.add_scalar('training loss',
                                      AllLoss / 5,
                                      epoch * len(trainloader) + i)
                    AllLoss = 0
            # print(optimizer.state_dict()['param_groups'][0]['lr'])

            acc, all = 0.0, 0
            with torch.no_grad():
                for X, y in validloader:
                    acc += (model(X).argmax(dim=1) == y).float().sum().item()
                    all += y.shape[0]
                writer.add_scalar('training accuracy',
                                  all,
                                  epoch * len(trainloader) + i)
                print("acc: ", acc/all)


