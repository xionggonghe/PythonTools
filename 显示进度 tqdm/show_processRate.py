from tqdm import tqdm
import time
epoch = 0
total = 200
epoch_nums = 5
"""最小单元"""
# pbar = tqdm(range(total))  # pbar类似与一个进度列表
# for i in pbar:        # pbar类似与一个进度列表
#     time.sleep(0.1)
#     pbar.set_description("processing: {}/{}".format(i, total)) # 在进度条前的文字描述

""" 单周期 """
# with tqdm(total=total) as pbar:  # tqdm 生成进度条
#     pbar.set_description('epoch:{}/{}'.format(epoch, total - 1))
#     for i in range(100):
#         time.sleep(0.1)
#         pbar.set_postfix(data='{:.6f}'.format(i)) # 在进度条后的文字描述
#         pbar.update(1)  # 更新进度条


"""多周期"""
while epoch <epoch_nums:
    epoch += 1
    time.sleep(0.1)
    total = epoch_nums*20
    with tqdm(total=total) as t:  # tqdm 生成进度条
        t.set_description('epoch:{}/{}'.format(epoch, total - 1))   # 在进度条前的文字描述
        """ 进度条内数值 <=total """
        for i in range(total):
            time.sleep(0.1)
            t.set_postfix(data='{:.6f} total: {}'.format(i, total))  # 在进度条后的文字描述
            t.update(1) # 更新进度条


"""*************************************** 实际应用***********************************************
注： from tqdm import tqdm
需更改数据:  train_set，epoch， num_epochs, epoch_loss, inputs
**********************************************************************************************"""
for epoch in epoch_nums:

    with tqdm(total=(len(train_set) - len(train_set) % batch_size)) as t:  # tqdm 生成进度条
        t.set_description('epoch:{}/{}'.format(epoch, num_epochs - 1))

        for i, data in enumerate(trainloader):
            inputs, y = data
            inputs = inputs.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = net(inputs)
            epoch_loss = loss(y_hat, y)
            epoch_loss.backward()
            with torch.no_grad():
                optimizer.step()
                """**********************"""
                t.set_postfix(loss='{:.6f}'.format(epoch_loss))
                t.update(len(X))