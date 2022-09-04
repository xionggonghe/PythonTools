import datetime
import os


curr_time = datetime.datetime.now()
print(curr_time)

# 输出格式：年-月-日 时:分:秒
timestamp=datetime.datetime.strftime(curr_time,'%Y-%m-%d %H:%M:%S')
print(timestamp)

# 输出格式：年-月-日
timestamp=curr_time.date()
print(timestamp)


print("简化时间：")
timestamp=datetime.datetime.strftime(curr_time,'%Y.%m.%d %H_%M_%S')
print(timestamp)

# if not os.path.exists("./res"):
#     print("built res")
#     os.makedirs("./res")
# if not os.path.exists('./model2/'+timestamp):
#     print("built model2")
#     os.makedirs('./model2/'+timestamp)
# torch.save(net, './model/GoogLeNet_3.pth')


