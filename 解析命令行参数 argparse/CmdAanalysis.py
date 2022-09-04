import argparse

# 创建命令行解析对象
parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument("-lr", "--LearningRate", type=float, default=0.0001, help="The rate of gradient descend")
parser.add_argument("-ep", "--epoch", type=float, default=100, help="The amount of training epoch")
parser.add_argument("-bsz", "--batch_size", type=int, default=64, help="size of the batches")

# 将参数解析给opt对象，可使用类似调opt对象的属性方式调用参数
opt = parser.parse_args()

# 调用测试
if __name__ == '__main__':
    print(opt.epoch)
    print(opt)



