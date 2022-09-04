from PSNR import torchPSNR
from SSIM import SSIM
from torchvision import transforms
from PIL import Image
import os

if __name__ == '__main__':
    # 加载图像
    transformations = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    filepath = "./pic/"
    filename = ["pic1_0.png", "pic1_1.png", "pic1_2.png", "pic2_0.png", "pic2_1.png", "pic2_2.png"]

    image, img_PIL, img_tensor = [], [], []
    for i in range(len(filename)):
        # x = os.path.join(filepath, filename[i])
        image.append(Image.open(os.path.join(filepath, filename[i])))
        img_PIL.append(image[i].convert('RGB'))
        img_tensor.append(transformations(img_PIL[i]))
        img_tensor[i] = img_tensor[i].unsqueeze(0)

# ***************************** PSNR **************************
    loss1 = torchPSNR(img_tensor[0], img_tensor[1])
    loss2 = torchPSNR(img_tensor[0], img_tensor[2])
    loss3 = torchPSNR(img_tensor[0], img_tensor[3])
    loss4 = torchPSNR(img_tensor[1], img_tensor[2])
    loss5 = torchPSNR(img_tensor[3], img_tensor[4])

    print("PSNR_loss01: ", loss1)
    print("PSNR_loss02: ", loss2)
    print("PSNR_loss03: ", loss3)
    print("PSNR_loss04: ", loss4)
    print("PSNR_loss05: ", loss5)

# ***************************** SSIM **************************
    SSIM_loss = SSIM()
    loss1 = SSIM_loss(img_tensor[0], img_tensor[1])
    loss2 = SSIM_loss(img_tensor[0], img_tensor[2])
    loss3 = SSIM_loss(img_tensor[0], img_tensor[3])
    loss4 = SSIM_loss(img_tensor[1], img_tensor[2])
    loss5 = SSIM_loss(img_tensor[3], img_tensor[4])

    print("SSIM_loss01: ", loss1)
    print("SSIM_loss02: ", loss2)
    print("SSIM_loss03: ", loss3)
    print("SSIM_loss04: ", loss4)
    print("SSIM_loss05: ", loss5)




