import glob
import os
filePath = r"C:\Users\XGH\OneDrive\桌面\pro\Gan\dataset\rain_data_train_Heavy\input" # 搜索文件夹名

# 搜索所有文件
file_list = glob.glob('{}/*'.format(filePath))      # * 类似正则表达式，表示所有内容都行
# print("all file:", file_list)

path = 'C:\\Users\\XGH\\OneDrive\\桌面\\pro\\Gan\\dataset\\rain_data_train_Heavy\\input\\'
for _, i in enumerate(file_list):
    print(i)
    init = i
    init_name = init[init.find("norain"):init.find(".png")]
    num = init_name[init_name.find("-")+1:init_name.find("x2")]
    renameStr = "rain-"+num+".png"
    renamePath = os.path.join(path, renameStr)
    print(renamePath)
    print(num)
    os.rename(i, renamePath)

# 搜索所有文件
file_list = glob.glob('{}/*'.format(filePath))      # * 类似正则表达式，表示所有内容都行
for i in range(10):
    print(file_list[i])

"""******************************通配符*****************************"""
# *：匹配零个或者多个字符
# ?：匹配一个字符
# []：匹配指定集合中的任意单个字符，比如[abc]表示匹配单个字符a或者b或者c
# {a,b}：匹配a或者b，a与b也是通配符，可以由其他通配符组成
# !：表示非，比如!1.txt表示排除文件1.txt
# [0-9]：匹配单个数字
# [[:upper:]]：匹配任意单个大写字母
# [[:lower:]]：匹配任意单个小写字母
# [[:digit:]]：匹配任意单个数字，等价于[0-9]
# [[:alpha:]]：匹配任意单个字母，包括大写字母与小写字母
# [[:alnum:]]：匹配任意单个字母与数字
# [[:space:]]：匹配单个空白字符
# [[:punctl:]]：匹配单个标点符号
# [^]:匹配指定集合之外的其他任意单个字符，比如[^abc]表示匹配除了a、b、c以外的其他任意字符



