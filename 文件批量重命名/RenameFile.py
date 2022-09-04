import os

if __name__ == '__main__':
    file_dir = "F:/project/pycharm/dataset/Rain100/rain_data_test_Heavy/rain_heavy_test/rain/"
    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root)  # 当前路径
        print('sub_dirs:', dirs)  # 子文件夹
        print('files:', files)  # 文件名称，返回list类型
    for i in range(len(files)):
        print("files[i]", files[i])
        num = files[i][files[i].find("-")+1:files[i].find("x2")]
        print("num:", num)

        des = root+"rain-"+num+".png"
        print("des", des)
        os.rename(root+files[i], des)
        print(" ")




