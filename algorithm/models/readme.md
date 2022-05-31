1. models/dataset.py 为自定义dataset文件  
2. models/train_and_val.py 为训练和验证代码  
3. models/network.py 定义网络结构  
    使用的是Restormer网络  
    Zamir S W, Arora A, Khan S, et al. Restormer: Efficient Transformer for High-Resolution Image Restoration[J]. arXiv preprint arXiv:2111.09881, 2021.
4. models/utils.py 里面为一些帮助函数
5. models/main.py 主函数  
6. models/predict.py 噪声图片去噪代码
7. models/preprocess.py 裁剪大图，生成小图将其保存为npy数据文件

执行步骤：  
1. 先将官方给定数据集移至models/dataset文件夹，再执行python preprocess.py生成4\*200\*200分辨率的数据集，
保存至models/dataset200文件夹下
2. 执行python main.py进行训练
3. 将待预测图片放至models/testdata文件夹下，执行python predict.py进行预测，预测后的图片保存至data文件夹下

训练环境：Tesla V100
1. 加载200\*200大小的图片，训练时随机裁剪为192\*192大小，进行一系列的图像增强（包括随机水平垂直翻转，随机旋转90°、180°、270°），
代码仿照 https://github.com/FanChiMao/SRMNet/blob/main/dataset_RGB.py。
2. 先使用l1损失，训练40个epoch，然后更换为mse损失，接着训练。
3. 使用AdamW优化器，设置训练400个epoch，batch_size设置为32
4. 虽然设置了lr_scheduler.ReduceLROnPlateau策略，在loss不降时动态调整学习率。
4. 预测需要32G内存，最后一张图的噪声较多，所以我将其连续预测5次得到最终图像。