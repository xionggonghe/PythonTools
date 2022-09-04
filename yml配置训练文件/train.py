import os
from config import Config 

# 解析参数，生成opt对象，可用字典的调用形式读取参数
opt = Config('./training.yml')


if __name__ == '__main__':
    # 字典的调用形式
    print(opt["BATCH_SIZE"])
    new_lr = opt["VERBOSE"]
    print("learning rate", new_lr)

    # 对象调用形式（会有警告）
    print(opt.SESSION)

