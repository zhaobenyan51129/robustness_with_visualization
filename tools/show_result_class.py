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

# 单步法
def plot_success_rate_vs_r_single(data, eta, algo, var, label_list, **kwargs):
    '''
    绘制不同类别的样本，各个标签（label）与参数（parameter）的关系图。
    每个子图表示一个模型和一个标签的关系，图例表示不同的 mask_mode。

    Args:
        data (pd.DataFrame): 数据集
        eta (float): eta值
        algo (str): 攻击算法
        var: x轴参数
        label_list (list): 需要绘制的标签列表，例如 ['success_rate', 'l1_norm', 'l2_norm', 'attack_loss', 'pixel_attacked']
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    locate = kwargs.get('locate', 'buttle')  # 默认图例位置为底部
    
    df_filtered = data[(round(data['eta'],2) == eta) & (data['algo'] == algo)].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    
    mask_mode_list = kwargs.get('mask_mode_list', df_filtered['mask_mode'].unique())
    id_list = kwargs.get('id_list', [254, 723, 948, 174, 110, 741, 492, 552, 423, 230, 751, 369, 249, 408, 534, 241, 733, 460, 848, 725])
    
    picture_num = len(id_list) * len(label_list)
    
    ncols = kwargs.get('ncols', int(np.ceil(np.sqrt(picture_num))))
    nrows = kwargs.get('nrows', int(np.ceil(picture_num / ncols)))

    palette = sns.color_palette("tab10", n_colors=len(mask_mode_list))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), squeeze=False)
    axes = axes.flatten()

    for i, id in enumerate(id_list):
        df_id = df_filtered[df_filtered['index'] == id]
        for j, label in enumerate(label_list):
            ax = axes[i * len(label_list) + j]
            
            # 筛选需要绘制的数据
            df_plot = df_id[[var, 'mask_mode', label]].dropna()
            sns.lineplot(
                data=df_plot, 
                x=var, 
                y=label, 
                hue='mask_mode', 
                hue_order=mask_mode_list,
                ax=ax, 
                palette=palette
            )
            
            Rank = df_id['Rank'].values[0]
            # 设置子图标题和轴标签
            ax.set_title(f"Index: {id}, Rank: {Rank} | {label}", fontsize=12)
            ax.set_xlabel(f'{var}', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis='x')

            # 移除子图中的图例
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            
            # 设置网格线样式
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='0.8')
            
    # 隐藏多余的子图
    total_subplots = nrows * ncols
    used_subplots = len(id_list) * len(label_list)
    for idx in range(used_subplots, total_subplots):
        axes[idx].axis('off')
    
    # 调整子图布局
    plt.tight_layout()

    # 添加图例
    if_legend = kwargs.get('if_legend', True)
    if if_legend:
        handles, labels = ax.get_legend_handles_labels()
        if locate == 'buttle':
            fig.legend(handles, labels, title='Mask Mode', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(mask_mode_list), fontsize=10)
            # 增加底部边距以避免图例挡住子图的坐标轴下方的文字
            plt.subplots_adjust(bottom=0.02)
        elif locate == 'right':
            fig.legend(handles, labels, title='Mask Mode', loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=10)

    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
# 多步法
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
    id_list = kwargs.get('id_list', [254, 723, 948, 174, 110, 741, 492, 552, 423, 230, 751, 369, 249, 408, 534, 241, 733, 460, 848, 725])

    mask_mode_list = kwargs.get('mask_mode_list', data['mask_mode'].unique())
    
    df_filtered = data[(data['mask_mode'].isin(mask_mode_list)) & (data['parameter'] == parameter)]
    df_filtered = df_filtered.reset_index(drop=True)
    
    picture_num = len(id_list)
        
    ncols = kwargs.get('ncols', int(np.ceil(np.sqrt(picture_num))))
    nrows = kwargs.get('nrows', int(np.ceil(picture_num / ncols)))
    
    palette = sns.color_palette("tab10", n_colors=len(mask_mode_list))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3.5), squeeze=False)
    axes = axes.flatten()

    for i, id in enumerate(id_list):
        df_id = df_filtered[df_filtered['index'] == id]
        for j, label in enumerate(label_list):
            ax = axes[i * len(label_list) + j]
            
            # 筛选需要绘制的数据
            df_plot = df_id[[var, 'mask_mode', label]].dropna()
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
            Rank = df_id['Rank'].values[0]
            ax.set_title(f"Index: {id}, Rank: {Rank} | {label}", fontsize=12)
            ax.set_xlabel(f'{var}', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis='x')
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            # 设置网格线样式
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='0.8')
            
    # 隐藏多余的子图
    total_subplots = nrows * ncols
    used_subplots = len(id_list) * len(label_list)
    for idx in range(used_subplots, total_subplots):
        axes[idx].axis('off')
    
    # 调整子图布局
    plt.tight_layout()

    if_legend = kwargs.get('if_legend', True)
    if if_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title='Mask Mode', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(mask_mode_list), fontsize=20)

    plt.subplots_adjust(bottom=0.05)
    
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show()

def plot_accuracy_success_rate_and_loss(data, **kwargs):
    '''
    绘制Rank与Accuracy, Success Rate和Loss的关系图。

    Args:
        data (pd.DataFrame): 数据集，包含'Rank', 'Accuracy', 'success_rate', 'Loss'列
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)

    # 创建调色板
    palette = sns.color_palette("tab10")

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制Accuracy的散点图和预测曲线
    sns.scatterplot(data=data, x='Rank', y='Accuracy', color=palette[0], ax=ax1, label='Accuracy')
    sns.lineplot(data=data, x='Rank', y='Accuracy', color=palette[0], ax=ax1)

    # 绘制Success Rate的预测曲线
    sns.lineplot(data=data, x='Rank', y='success_rate', color=palette[1], ax=ax1, label='Success Rate')

    # 设置左侧纵坐标标签和范围
    ax1.set_ylabel('Accuracy / Success Rate')
    ax1.set_ylim(0, 1.1)  # 设置纵坐标范围
    ax1.tick_params(axis='y')

    # 创建右侧纵坐标轴
    ax2 = ax1.twinx()
    sns.lineplot(data=data, x='Rank', y='Loss', color=palette[2], ax=ax2, label='ori_Loss')
    sns.lineplot(data=data, x='Rank', y='attack_loss', color=palette[3], ax=ax2, label='attack_loss')
    ax2.set_ylabel('Loss', color=palette[2])
    ax2.tick_params(axis='y', labelcolor=palette[2])

    # 设置图形标题和横坐标标签
    plt.title('Accuracy, Success Rate, and Loss by Rank')
    ax1.set_xlabel('Rank')

    # 确保横坐标是整数并显示所有Rank值
    ax1.set_xticks(data['Rank'].unique())

    # 显示图例
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show()
        
def plot_accuracy_success_rate(data, **kwargs):
    '''
    绘制Rank与Accuracy和Success Rate的关系图。

    Args:
        data (pd.DataFrame): 数据集，包含'Rank', 'Accuracy', 'success_rate'列
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)

    # 创建调色板
    palette = sns.color_palette("tab10")

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制Accuracy的散点图和预测曲线
    sns.scatterplot(data=data, x='Rank', y='Accuracy', color=palette[0], ax=ax1, label='Accuracy')
    sns.lineplot(data=data, x='Rank', y='Accuracy', color=palette[0], ax=ax1)

    # 设置左侧纵坐标标签和范围
    ax1.set_ylabel('Accuracy', color=palette[0])
    ax1.set_ylim(0, 1.1)  # 设置纵坐标范围
    ax1.tick_params(axis='y', labelcolor=palette[0])

    # 创建右侧纵坐标轴
    ax2 = ax1.twinx()
    sns.lineplot(data=data, x='Rank', y='success_rate', color=palette[1], ax=ax2, label='Success Rate')
    ax2.set_ylabel('Success Rate', color=palette[1])
    ax2.set_ylim(0, 1.1)  # 设置纵坐标范围
    ax2.tick_params(axis='y', labelcolor=palette[1])

    # 设置图形标题和横坐标标签
    plt.title('Accuracy and Success Rate by Rank')
    ax1.set_xlabel('Rank')

    # 确保横坐标是整数并显示所有Rank值
    ax1.set_xticks(data['Rank'].unique())

    # 显示图例
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show()