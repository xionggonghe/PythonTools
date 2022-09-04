import torch
from torch import nn



class CNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(96, 48, kernel_size=5, padding=5 // 2)
        self.conv4 = nn.Conv2d(48, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print("x shape :", x.shape)     #[16, 1, 224, 224]
        x = self.relu(self.conv1(x))
        print("x conv1 shape :", x.shape)   #[16, 64, 224, 224]
        x = self.relu(self.conv2(x))
        print("x conv2 shape :", x.shape)   #[16, 32, 224, 224]
        x = self.conv3(x)
        print("x conv3 shape :", x.shape)   #[16, 32, 224, 224]
        x = self.conv4(x)
        print("x conv4 shape :", x.shape)   #[16, 1, 224, 224]
        return x

if __name__=="__main__":
    x = torch.ones(16, 1, 224, 224)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    y = model(x)





