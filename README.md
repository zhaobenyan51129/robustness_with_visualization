# robustness_with_visualization
本代码库为利用grad-cam可视化办法进行对抗鲁棒性攻击

## 模型
1. 本代码库支持使用各种图片分类模型，详见：models/load_model.py;
2. 实际实验仅使用vit_b_16\resnet50\vgg16.

## 数据来源
1. 数据集imagenet/val，共1000个类别，每个类别50张图片，共50000张图片；
    * 下载地址：https://pan.quark.cn/s/6e7d21fc7aab（夸克网盘）
    * 下载后解压文件，其中ILSVRC2012_devkit_t12.tar.gz为标签数据，ILSVRC2012_img_val.tar为图片数据，imagenet_class_index.json为类别和索引对应的字典，可以通过json.load(文件路径)查看其内容；
    * 使用Python脚本imagenet.py将数据转为Pytorch要求的数据格式，注意修改文件路径为自己想要保存的路径。
2. 数据筛选：见data_preprocessor/select_images.py
    * 数据1：筛选所有模型预测正确的指定张图片；
    * 数据2：筛选vit整体分类正确率排名第1,101,201，…,901的类别，并选出所有预测正确的图片；
    * 所有筛选出来的图片文件保存在data文件夹下；
  
## 工具函数
./tools文件夹下是一些常用的工具函数，包括：
1. show_images.py:画图常用函数；
2. create_video.py:将图片做成视频；
3. get_classes.py:根据json文件获取imagenet每个类别的索引对应类别的中文名；
4. compute_topk.py:计算输入数组的前n大的值和位置

## 可视化
grad-cam:见./visualization/grad_cam.py
1. 可以选择在类class GradCAM的__call__方法中调用self.get_grad_of_loss()同时输出梯度；
2. 注意在对多个函数进行反向传播时，一定要每次清空梯度（否则会累加），并且第一次反向传播时设置retain_graph=True；
3. 为了使得每个版块独立，不在这里计算梯度，而是在进行对抗攻击时计算梯度。

## 对抗攻击方法
* ./algorithms文件夹实现了不同的基于梯度的对抗攻击方法，实际使用时只用封装好的one_step_attacker.py和multi_step_attacker.py；
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
./algorithms/one_step_attacker.py,封装了不同的单步法
#### 攻击算法选择
1. fgsm
2. fgm
3. gauss noise

### 多步法
* ./algorithms/multi_step_attacker.py,封装了不同的多步法
