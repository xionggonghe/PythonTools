import os

if __name__ == '__main__':
    file_dir = "F:/project/pycharm/dataset/Rain1400/norain/"
    endStr = ".jpg"

    type = "rain" if file_dir.find("norain")<0 else "norain"

    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root)  # 当前路径
        print('sub_dirs:', dirs)  # 子文件夹
        print('files:', files)  # 文件名称，返回list类型
    for i in range(len(files)):
        print("files[{}]".format(i), files[i])

        num = files[i][:files[i].find(endStr)]
        print("num:", num)
        # if len(num)<4:
        #     if len(num)<3:
        #         if len(num)<2:
        #             des = root + type +"-000" + num + endStr
        #         else:
        #             des = root + type +"-00" + num + endStr
        #     else:
        #         des = root + type +"-0" + num + endStr
        # else:
        des = root + type + "-" +num+endStr
        print("des", des)
        os.rename(root+files[i], des)
        print(" ")




