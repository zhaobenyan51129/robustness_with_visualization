# robustness_with_visualization
本代码库为利用grad-cam可视化办法进行对抗鲁棒性攻击

## 模型
1. 本代码库支持使用各种图片分类模型，详见：models/load_model.py;
2. 实际实验仅使用vit_b_16.

## 数据来源
1. 数据集imagenet/val，共1000个类别，每个类别50张图片，共50000张图片；
2. 数据筛选：见data_preprocessor/select_images.py
    * 筛选所有模型预测正确的100张图片；
    * 筛选vit整体分类正确率排名第1,101,201，…,901的类别，并选出所有预测正确的图片；
    * 所有筛选出来的图片文件保存在data文件夹下；
## 可视化
1. grad-cam:见./visualization/grad_cam.py
