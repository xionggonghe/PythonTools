import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
#from XGHNet import *
from PIL import Image
from PIL import ImageOps
# from model.networks import QDerainNet
from torch.autograd import Variable
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
# from utils.dataset_RGB import DataLoader_Test
import torchvision.transforms as transform
import random
# from utils.SSIM import ssim
# from utils.PSNR import torchPSNR as PSNR
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
import numpy as np
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
# # 假设你有一个大小为 16x16 的特征图
# feature_map = np.random.rand(16, 16)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fitNet(x):
    c, h, w = x.shape
    if (h%32 != 0) | (w%32 != 0):
        crop_obj = torchvision.transforms.CenterCrop([int(h/32)*32, int(w/32)*32])
        x = crop_obj(x)

    if (h > 800) | (w > 800):
        print("!!!!! so big !!!!!!")
        crop_obj = torchvision.transforms.CenterCrop([512, 512])
        x = crop_obj(x)
    # crop_obj = torchvision.transforms.CenterCrop([256, 256])
    # x = crop_obj(x)
    return x

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # # RGB转BRG
    # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def DataLoader_Test(test_dir):
    inp_files = sorted(os.listdir(os.path.join(test_dir, 'rain')))
    tar_files = sorted(os.listdir(os.path.join(test_dir, 'norain')))
    inp_filenames = [os.path.join(test_dir, 'rain', x) for x in inp_files if is_image_file(x)]
    tar_filenames = [os.path.join(test_dir, 'norain', x) for x in tar_files if is_image_file(x)]
    return inp_filenames, tar_filenames

def Datacrop(test_dir="G:/NLAQNet/data/test/Rain100L", Num=0):
    inp_fileList, tar_fileList = DataLoader_Test(test_dir)

    model.eval()

    inp_img = cv2.imread(inp_fileList[Num])
    tar_img = cv2.imread(tar_fileList[Num])

    input_ = TF.to_tensor(inp_img)
    target = TF.to_tensor(tar_img)
    # print("shape:", input_.shape, target.shape)
    input_ = fitNet(input_)
    target = fitNet(target)
    # print("fitNet  :", input_.shape, target.shape)

    with torch.no_grad():
        # input_ = input_.to(device)
        # target = target.to(device)
        input_ = torch.unsqueeze(input_, dim=0)
        target = torch.unsqueeze(target, dim=0)

    return input_, target

def plot_examples(fea, colormaps):
    data = fea
    n = 1
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=0, vmax=1)
        fig.colorbar(psm, ax=ax)
    plt.show()

if __name__ == '__main__':
    PATH = "./result/ALLmodel.pth"
    model = torch.load(PATH)
    # input_Path = "./dataset/rain/"
    target_Path = "./fea"
    derain_Path = "./fea"

    test_dir = "./Data/"
    # test_dir = "G:/NLAQNet/data/Rain100H/RainTestH/"
    num = 0
    input_, target = Datacrop(test_dir, num)

    restored, feature = model.feature_extract(input_)
    # psnr = PSNR(restored, target)
    # ssim = ssim(restored, target)
    # print("psnr: ", psnr)
    # print("ssim: ", ssim)

    save_image_tensor2cv2(input_tensor=target, filename=os.path.join(target_Path, 'target{}.png'.format(num)))
    save_image_tensor2cv2(input_tensor=restored, filename=os.path.join(derain_Path, 'restored{}.png'.format(num)))

    if not os.path.exists('./featureMap'):
        os.makedirs('./featureMap')
    file_name = ["x", "r", "xinit", "outPBnet", "outPUnet","outBnet","outQUnet", "xagg", "outr", "out"]
    for i in range(len(feature)):
        heat = feature[i].data.numpy()	     # 将tensor格式的feature map转为numpy格式
        heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
        # heatmap = np.maximum(heat, 0)        # heatmap与0比较
        heatmap = np.mean(heat, axis=0)   # 多通道时，取均值
        heatmap /= np.max(heatmap)
        #plt.matshow(heatmap)
        #plt.show()
        img = input_
        heatmap = cv2.resize(heatmap, (input_.shape[3], input_.shape[2]))  # 特征图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)              # 将特征图转换为uint8格式
        heat_img_S = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)  # 将特征图转为伪彩色图
        heat_img_W = cv2.applyColorMap(heatmap, cv2.COLORMAP_WINTER)  # 将特征图转为伪彩色图
        heat_img = np.uint8(heat_img_S)
        # 可视化特征图
        nodes = [0.5]
        colors = ["lawngreen", "lightseagreen"]
        cam = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

        # ---------------------颜色表设置规则----------------------
        # 'red': [(0.0, 0.0, 43 / 255.0),    （val0, x0, y0）
        #         (1.0, 248 / 255.0, 1.0)]   （val1, x1, y1）
        # val: 0~1.0    代表归一化数据范围
        # 上面表示的是在val0~val1的数据范围内，
        # 颜色的值由y0向x1进行渐变
        # （val0, x0, y0）可以有很多段，进行颜色锚定;
        # x0, y1可以不用管
        cdict = {'red': [(0.0, 0.0, 43/255.0),
                         # (0.5, 1.0, 1.0),
                         (1.0,  248/255.0, 1.0)],

                 'green': [(0.0, 0.0, 180/255.0),
                           # (0.75, 1.0, 1.0),
                           (1.0, 253/255.0, 1.0)],

                 'blue': [(0.0, 0.0, 45/255.0),
                          # (0.25, 0.0, 0.0),
                          # (0.5, 0.0, 0.0),
                          (1.0, 43/255.0, 1.0)]}

        cmp = mpl.colors.LinearSegmentedColormap('lsc', segmentdata=cdict)

        plt.imshow(heatmap, cmap=cmp)
        # plt.imsave('./fea/' + str(num) + 'Rain12_pltBuGn{}_{}.png'.format(i, file_name[i]),heatmap, cmap='BuGn')
        plt.axis('off')
        plt.show()


        # plot_examples(heatmap, [cam])
        # heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_PARULA)  # 将特征图转为伪彩色图
        # heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET )  # 将特征图转为伪彩色图
        # COLORMAP_JET COLORMAP_BONE
        # heat_img = cv2.addWeighted(img, 1, heat_img, 0.5, 0)     # 将伪彩色图与原始图片融合
        # heat_img = heat_img * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
        # cv2.imwrite('./fea/' + str(num) + 'Rain12_SW{}_{}.png'.format(i, file_name[i]), heat_img)  # 将图像保存


#
# if __name__ == '__main__':
#     """**************************************************************"""
#     save_infor = torch.load('../result/model_latest.pth',  map_location=torch.device('cpu'))
#     model = QDerainNet().to(device)
#     model.load_state_dict(save_infor['state_dict'])
#     print(model)
#
#     """**************************************************************"""
#
#     Num = 0
#     PatchSize = 256
#
#     img_path = "C:/Users/Xiong/Documents/paper/Derain/dataset/rain_data_train_Light/" + "rain/rain-{}.png".format(Num)
#     img_org = cv2.imread(img_path)
#
#     """**************************************************************"""
#     img_tensor = Datacrop(Num, PatchSize)
#     inp_img = img_tensor[0]
#     inp_img = torch.unsqueeze(inp_img, dim=0)
#     out, feature = model.feature_extract(inp_img)
#     show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
#     show(torch.squeeze(inp_img, dim=0)).show()
#     show(torch.squeeze(out, dim=0)).show()
#
#     """**************************************************************"""
#     if not os.path.exists('./featureMap'):
#         os.makedirs('./featureMap')
#     file_name = ["x_init", "xDEC", "x_v", "x"]
#     for i in range(len(feature)):
#
#         heat = feature[i].data.numpy()	     # 将tensor格式的feature map转为numpy格式
#         heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
#         heatmap = np.maximum(heat, 0)        # heatmap与0比较
#         heatmap = np.mean(heatmap, axis=0)   # 多通道时，取均值
#         heatmap /= np.max(heatmap)
#         #plt.matshow(heatmap)
#         #plt.show()
#         img = img_org
#         heatmap = cv2.resize(heatmap, (PatchSize, PatchSize))  # 特征图的大小调整为与原始图像相同
#         heatmap = np.uint8(255 * heatmap)              # 将特征图转换为uint8格式
#         heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)  # 将特征图转为伪彩色图
#         # COLORMAP_JET
#         # heat_img = cv2.addWeighted(img_org, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
#         #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
#
#         save_image_tensor2cv2(input_tensor=target, filename=os.path.join(target_Path, '{}.png'.format(i)))
#         save_image_tensor2cv2(input_tensor=restored, filename=os.path.join(derain_Path, '{}.png'.format(i)))





















