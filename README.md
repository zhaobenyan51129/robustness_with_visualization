# robustness_with_visualization
本代码库使用经典FGSM（单步）和I-FGSM(多步）算法进行对抗鲁棒性攻击，并利用grad-cam可视化方法可视化对抗样本，探究攻击不同位置时对抗鲁棒性的区别。

## 代码环境
* ./dependence/robustness_environment.yml是此代码路使用的conda环境文件，使用命令conda env create -f {文件地址}导入环境。

## 模型
1. 本代码库支持使用各种图片分类模型，详见：models/load_model.py;
2. 实际实验仅使用vit_b_16\resnet50\vgg16，其中vit_b_16是使用的代码models/vit_model.py（主要是因为gadcam可视化算法对vit模型需要单独处理，写代码时是基于此模型），其他是调用pytorch自带模型.

## 数据来源
1. 数据集imagenet/val，共1000个类别，每个类别50张图片，共50000张图片；
    * 下载地址：https://pan.quark.cn/s/6e7d21fc7aab（夸克网盘）
    * 下载后解压文件，其中ILSVRC2012_devkit_t12.tar.gz为标签数据，ILSVRC2012_img_val.tar为图片数据，imagenet_class_index.json为类别和索引对应的字典，可以通过json.load(文件路径)查看其内容，data/chinese_class_index.json是类别索引对应的中文名称，源自网络，不一定完全翻译正确，注意甄别；
    * 使用Python脚本imagenet.py将数据转为Pytorch要求的数据格式，注意修改文件路径为自己想要保存的路径。
2. 数据筛选：见data_preprocessor/select_images.py
    * 数据1：筛选所有模型预测正确的指定张图片（实验时两个数据集一个是1000张，一个是100张，分别命名为：ADV1000,ADV100)；
    * 数据2：筛选vit整体分类正确率排名第1,100,200，…,900, 1000的类别，并选出所有预测正确的图片；
    * 所有筛选出来的图片文件保存在data文件夹下（可以自己指定）；
  
## 工具函数
./tools文件夹下是一些常用的工具函数，包括：
1. show_images.py: 画图常用函数；
2. create_video.py: 将图片做成视频；
3. get_classes.py: 根据json文件获取imagenet每个类别的索引对应类别的英文/中文名；
4. compute_topk.py: 计算输入数组的前n大的值和位置；
5. merge_images.py: 将多张图片合并为一整张大图；
6. show_result_one_step.py：对于单步法的结果进行分析并画图；
7. show_result_multi_step.py：对于多步法的结果进行分析并画图；

## 可视化
grad-cam:见./visualization/grad_cam.py
1. 可以选择在类class GradCAM的__call__方法中调用self.get_grad_of_loss()同时输出梯度；
2. 注意在对多个函数进行反向传播时，一定要每次清空梯度（否则会累加），并且第一次反向传播时设置retain_graph=True；
3. 为了使得每个版块独立，不在这里计算梯度，而是在进行对抗攻击时计算梯度。

## 对抗攻击方法
* ./algorithms文件夹实现了不同的基于梯度的对抗攻击方法，实际使用时只会调用封装好的one_step_attacker.py\single_step_attack.py\multi_step_attack.py；
* 关于这些算法的原理与区别见相关参考文献。

### 攻击pixel选择
1. 'all': 所有像素点；
2. 'positive': 梯度为正的像素点；
3. 'negative': 梯度为负的像素点；
4. 'topk':梯度绝对值前topk大的像素点；
5. 'topr':梯度绝对值前topr比例的像素点；
5. 'randomk':随机选择randomk个像素点；
6. 'randomr':随机选择randomr比例的像素；
7. 'channel_randomk':随机选择randomk个像素，但是3个通道最多只能攻击一个通道；
8. 'channel_randomr':随机选择比例为randomr个像素，但是3个通道最多只能攻击一个通道；
9. 'cam_topk':cam绝对值前topk大的像素点；
10. 'cam_topr':cam绝对值前topr（比例）大的像素点。

### 单步法：
主函数：./main/main_single_step_attack.py, 调用algorithms/single_step_attack.py
#### 攻击算法选择
1. fgsm
2. gaussian noise
3. 测试过程中用到的其他算法：fgm、gaussian_noise_sign等，实际实验时不使用

### 多步法
主函数：./main/main_multi_step_attack.py, 调用algorithms/multi_step_attack.py
#### 攻击算法选择
1. i_fgsm
2. 测试过程中用到的其他算法：gd等，实际实验时不使用

### 对指定类别进行实验
主函数：./main/main_class_attack.py, 包括单步法和多步法

## 结果保存
main文件夹下的是代码入口，里面可以看到保存的指标，单步法每次会保存一个excel文件，多步法会对每个模型保存一个excel文件，这是因为多步法耗时较长，避免中断丢失所有数据。设置参数show = True的话会保存中间步骤的一些可视化图片，例如对抗样本、扰动、GradCAM热力图等，但是画图比较慢，大批量试验时不建议设置这个参数。

## 结果分析
主目录是三个文件data_analysis_one_step.ipynb、data_analysis_multi_step.ipynb、data_analysis_classified.ipynb分别是对不同的实验结果进行分析，由于内存限制，数据集较大时分多个batch进行实验，在结果分析中会对每个batch的数据进行合并、基础数据预处理、画图
