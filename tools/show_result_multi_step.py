import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import plotly.express as px

font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  # 显示中文字体

if not os.path.exists(font_path):
    print(f"字体文件 {font_path} 未找到。请确保字体文件存在于工作目录中。")
else:
    # 创建FontProperties对象
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

## 1.先只看结果（只关注step0和最后一步攻击的结果）
### 1.1. 画出攻击成功率和损失的柱状图,只考虑Mask Mode('all','negative','positive')
def plot_success_rate_vs_mask_mode(data, **kwargs):
    ''' 绘制攻击成功率与 Mask Mode('all','negative','positive') 的关系
    Args:
        data: DataFrame, 包含攻击成功率和 Mask Mode 的数据
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    
    df_filtered = data[(data['end'] == 1) & (data['mask_mode'].isin(['all','negative','positive']))].copy()
    df_filtered['success_rate'] = df_filtered['success_rate'].round(3) * 100
    # df_filtered['loss'] = - df_filtered['attack_loss']
    metrics_titles = {
    'success_rate': '攻击成功率 (%)',
    'step': '攻击步数',
    # 'loss': '最终损失',
    # 'l1_norm': '梯度的一范数 (L1 Norm)',
    # 'l2_norm': '梯度的二范数 (L2 Norm)'
}
    palette = sns.color_palette("pastel", n_colors=df_filtered['model'].nunique())

    plt.figure(figsize=(20, 5))

    for i, (metric, title) in enumerate(metrics_titles.items(), 1):
        plt.subplot(1, 2, i)
        
        ax = sns.barplot(
            data=df_filtered,
            x='mask_mode',
            y=metric,
            hue='model',
            palette=palette
        )
        
        # 在柱形图上显示值
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  # 检查柱子的高度
                ax.annotate(
                    f'{height:.2f}',  # 保留两位小数
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', 
                    va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points'
                )
        
        plt.title(f'{title} 与 Mask Mode 的关系', fontsize=14, fontproperties=font_prop)
        plt.ylabel(title, fontsize=12, fontproperties=font_prop)
        plt.xticks()
        if i != 3:
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        else:
            ax.get_legend().remove()
        
    plt.tight_layout()
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show()

## 1.2 画成功率、损失、L1范数、L2范数、步数随parameter的变化的折线图
def plot_success_rate_vs_r(data, var, label_list, **kwargs):
    '''
    绘制不同模型下，各个标签（label）与参数（parameter）的关系图。
    每个子图表示一个模型和一个标签的关系，图例表示不同的 mask_mode。

    Args:
        data (pd.DataFrame): 数据集
        var: x轴参数
        label_list (list): 需要绘制的标签列表，例如 ['success_rate', 'l1_norm', 'l2_norm', 'loss', 'step']
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    
    df_filtered = data[(data['end'] == 1) & (data['mask_mode'].isin(['topr', 'channel_topr', 'cam_topr','seed_randomr', 'seed_randomr_lowr', 'cam_lowr', 'channel_lowr', 'lowr']))].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    
    # 获取唯一的模型和mask_mode
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    mask_mode_list = kwargs.get('mask_mode_list', df_filtered['mask_mode'].unique())
    
    ncols = kwargs.get('ncols', len(label_list))
    nrows = kwargs.get('nrows', len(model_list))

    sns.set(style="whitegrid", rc={"grid.linestyle": "--", "grid.color": "0.8"})

    palette = sns.color_palette("tab10", n_colors=len(mask_mode_list))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), squeeze=False)
    axes = axes.flatten()

    for i, model in enumerate(model_list):
        df_model = df_filtered[df_filtered['model'] == model]
        for j, label in enumerate(label_list):
            ax = axes[i * len(label_list) + j]
            
            # 筛选需要绘制的数据
            df_plot = df_model[[var, 'mask_mode', label]].dropna()
            sns.lineplot(
                data=df_plot, 
                x=var, 
                y=label, 
                hue='mask_mode', 
                hue_order=mask_mode_list,
                ax=ax, 
                # marker='o', 
                palette=palette
            )
            
            # 设置子图标题和轴标签
            ax.set_title(f"Model: {model} | {label}", fontsize=12)
            ax.set_xlabel(f'{var}', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis='x')
            
            # 移除子图中的图例
            ax.get_legend().remove()
            
    # 隐藏多余的子图
    total_subplots = nrows * ncols
    used_subplots = len(model_list) * len(label_list)
    for idx in range(used_subplots, total_subplots):
        axes[idx].axis('off')
    
    # 调整子图布局
    plt.tight_layout()
    
    # 添加图例到整个图像的最下面
    if_legend = kwargs.get('if_legend', True)
    if if_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title='Mask Mode', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(mask_mode_list), fontsize=10)
    
    # 增加底部边距以避免图例挡住子图的坐标轴下方的文字
    plt.subplots_adjust(bottom=0.2)
    
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

### 3. loss与pred_loss的散点图
def plot_loss_vs_pred_loss(data, x='attack_loss', y='pred_loss', **kwargs):
    '''
    绘制不同模型attack_loss和pred_loss的散点图
    Args:
        data (pd.DataFrame): 数据集
        eta (float): eta值
        algo (str): 攻击算法
        x: x轴参数
        y: y轴参数
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)

    df_filtered = data.copy()
    
    
    # 获取唯一的模型
    model_list = kwargs.get('model_list', ['vit_b_16', 'resnet50', 'vgg16'])
    mask_mode_list = kwargs.get('mask_mode_list', df_filtered['mask_mode'].unique())
    
    df_filtered = df_filtered[df_filtered['mask_mode'].isin(mask_mode_list)]
    df_filtered = df_filtered.reset_index(drop=True)

    ncols = kwargs.get('ncols', len(model_list))
    nrows = kwargs.get('nrows', 1)

    sns.set(style="whitegrid", rc={"grid.linestyle": "--", "grid.color": "0.8"})
    palette = sns.color_palette("tab10", n_colors=len(mask_mode_list))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), squeeze=False)
    axes = axes.flatten()

    for i, model in enumerate(model_list):
        ax = axes[i]
        df_model = df_filtered[df_filtered['model'] == model]
        sns.scatterplot(
            x=x, y=y, data=df_model, alpha=0.5, hue='mask_mode', 
            hue_order=mask_mode_list, palette=palette, ax=ax
        )
        # 设置子图标题和轴标签
        ax.set_title(f'Model: {model}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.get_legend().remove()
    
    # 调整子图布局
    plt.tight_layout()

    # 添加图例到整个图像的最下面
    if_legend = kwargs.get('if_legend', True)
    if if_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title='Mask Mode', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(mask_mode_list), fontsize=11)

    # 增加底部边距以避免图例挡住子图的坐标轴下方的文字
    plt.subplots_adjust(bottom=0.25)

    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_success_rate_vs_step(data, parameter, var, label_list, **kwargs):
    '''
    绘制不同模型下，各个标签（label）与参数（parameter）的关系图。
    每个子图表示一个模型和一个标签的关系，图例表示不同的 mask_mode。

    Args:
        data (pd.DataFrame): 数据集
        parameter (str): 攻击比例
        var: x轴参数, 例如 'step'
        label_list (list): 需要绘制的标签列表，例如 ['success_rate', 'l1_norm', 'l2_norm', 'loss', 'step']
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    model_list = kwargs.get('model_list', ['vit_b_16', 'resnet50', 'vgg16'])

    mask_mode_list = kwargs.get('mask_mode_list', data['mask_mode'].unique())
    
    
    df_filtered = data[(data['mask_mode'].isin(mask_mode_list)) & (data['parameter'] == parameter)]
    df_filtered = df_filtered.reset_index(drop=True)
    
    ncols = kwargs.get('ncols', len(label_list))
    nrows = kwargs.get('nrows', len(model_list))

    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(mask_mode_list))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), squeeze=False)
    axes = axes.flatten()

    for i, model in enumerate(model_list):
        df_model = df_filtered[df_filtered['model'] == model]
        for j, label in enumerate(label_list):
            ax = axes[i * len(label_list) + j]
            
            # 筛选需要绘制的数据
            df_plot = df_model[[var, 'mask_mode', label]].dropna()
            sns.lineplot(
                data=df_plot, 
                x=var, 
                y=label, 
                hue='mask_mode', 
                hue_order=mask_mode_list, 
                ax=ax, 
                # marker='o', 
                palette=palette
            )
            
            # 设置子图标题和轴标签
            ax.set_title(f"Model: {model} | {label}", fontsize=12)
            ax.set_xlabel(f'{var}', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis='x')
            ax.get_legend().remove()
            
    # 隐藏多余的子图
    total_subplots = nrows * ncols
    used_subplots = len(model_list) * len(label_list)
    for idx in range(used_subplots, total_subplots):
        axes[idx].axis('off')
    
    # 调整子图布局
    plt.tight_layout()

    if_legend = kwargs.get('if_legend', True)
    if if_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title='Mask Mode', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(mask_mode_list), fontsize=11)

    plt.subplots_adjust(bottom=0.25)
    
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show()
