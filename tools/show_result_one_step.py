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

### 1.1 成功率 and  loss vs eta
def plot_success_rate_vs_eta(data, algo_list, var_list, **kwargs):
    '''
    对每一组 algo 和 var 画一个子图。

    Args:
        data (pd.DataFrame): 数据集
        algo_list (list): 算法列表
        var_list (list): 变量列表
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)

    num_plots = len(algo_list) * len(var_list)

    nrows = kwargs.get('nrows', int(np.ceil(np.sqrt(num_plots))))
    ncols = kwargs.get('ncols', int(np.ceil(num_plots / nrows)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 4), squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for algo in algo_list:
        for var in var_list:
            df_filtered = data[data['algo'] == algo].copy()
            df_filtered['success_rate'] = df_filtered['success_rate'] * 100

            ax = axes[plot_idx]

            # 获取唯一的模型和掩码模式列表
            models = df_filtered['model'].unique()
            mask_modes = df_filtered['mask_mode'].unique()

            # 为不同的 mask_mode 指定不同的颜色
            palette = sns.color_palette("tab10", n_colors=len(mask_modes))
            mask_mode_colors = dict(zip(mask_modes, palette))

            # 为不同的模型指定不同的线型和标记
            line_styles = ['-', '--', '-.', ':']
            markers = ['o', '^', 'X', 'h', '*', 'v', 's', 'p', 'D',  'd']

            # 创建模型到线型和标记的映射
            model_line_styles = {model: line_styles[i % len(line_styles)] for i, model in enumerate(models)}
            model_markers = {model: markers[i % len(markers)] for i, model in enumerate(models)}

            # 准备 dashes 映射，Seaborn 需要 dash 样式为点划线序列
            dash_styles = {
                '-': '',             # 实线
                '--': (5, 5),        # 虚线
                ':': (1, 3),          # 点线
                '-.': (3, 5, 1, 5),  # 点划线
                
            }
            model_dashes = {model: dash_styles[model_line_styles[model]] for model in models}

            # 准备 markers 映射
            model_markers_mapping = {model: model_markers[model] for model in models}

            # 绘制图形
            sns.lineplot(
                data=df_filtered,
                x='eta',
                y=var,
                hue='mask_mode',      # 使用不同的颜色
                style='model',        # 使用不同的线型
                markers=model_markers_mapping,
                dashes=model_dashes,
                hue_order=mask_modes,
                style_order=models,
                palette=mask_mode_colors,
                ax=ax
            )

            ax.set_title(f'{var.title()} vs Eta for {algo}', fontsize=12)
            ax.set_xlabel('eta', fontsize=10)
            ax.set_ylabel(f'{var.title()}', fontsize=10)

            # 只在第一个子图中显示图例
            if plot_idx == 0:
                ax.legend(title='Model - Mask Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend().set_visible(False)

            plot_idx += 1

    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show() 

## 1.2 norm vs 成功率
def plot_success_rate_vs_norm(data, algo, eta, **kwargs):
    ''' 绘制攻击成功率与梯度范数的关系
    
    Args:
    - data: DataFrame, 数据集
    - algo: str, 攻击算法
    - eta: float, 攻击强度
    - kwargs: dict, 其他参数
    
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    df_filtered = data[(data['eta'] == eta) & (data['algo'] == algo)].copy()
    df_filtered['success_rate'] = df_filtered['success_rate'].round(3) * 100
    metrics_titles = {
        'success_rate': '攻击成功率 (%)',
        'pixel_attacked': '被攻击的像素数量',
        'l1_norm': '梯度的一范数 (L1 Norm)',
        'l2_norm': '梯度的二范数 (L2 Norm)'
    }
    palette = sns.color_palette("pastel", n_colors=df_filtered['model'].nunique())

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics_titles.items()):
        ax = axes[i]
        
        sns.barplot(
            data=df_filtered,
            x='mask_mode',
            y=metric,
            hue='model',
            palette=palette,
            ax=ax
        )
        
        # 在柱形图上显示值
        for p in ax.patches:
            height = p.get_height()
            if height > 0 and i != 1:  # 检查柱子的高度
                ax.annotate(
                    f'{height:.2f}',  # 保留两位小数
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', 
                    va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points'
                )
        
        ax.set_title(f'{title} 与 Mask Mode 的关系', fontsize=14)
        ax.set_ylabel(title, fontsize=12)
        ax.tick_params(axis='x')
        
        # 移除子图中的图例
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    # 添加图例到整个图像的最下面
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title='Model', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(df_filtered['model'].unique()), fontsize=14, title_fontsize=16)
    
    # 增加底部边距以避免图例挡住子图的坐标轴下方的文字
    plt.subplots_adjust(bottom=0.05)
    
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

## 暂时没用      
def plot_success_rate_vs_algo(data, model_list, var_list, **kwargs):
    '''
    对每一组 model 和 var 画一个子图。
    横坐标是 eta，纵坐标是 success_rate，标签是 algo&mask_mode。

    Args:
        data (pd.DataFrame): 数据集
        model_list (list): 模型列表
        var_list (list): 变量列表
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    
    num_plots = len(model_list) * len(var_list)
    ncols = int(np.ceil(np.sqrt(num_plots)))
    nrows = int(np.ceil(num_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), squeeze=False)
    axes = axes.flatten()
    
    plot_idx = 0
    
    for var in var_list:
        for model in model_list:
            df_filtered = data.copy()
            df_filtered = df_filtered.reset_index(drop=True)
            df_filtered['success_rate'] = df_filtered['success_rate'].round(3) * 100
            
            # 筛选固定模型的数据
            df_model = df_filtered[df_filtered['model'] == model]
            
            # 创建新的标签列 'algo&mask_mode'
            df_model['algo&mask_mode'] = df_model['algo'] + '&' + df_model['mask_mode']
            
            # 获取唯一的算法和mask_mode组合
            algo_mask_mode_list = df_model['algo&mask_mode'].unique()
            
            sns.set(style="whitegrid", rc={"grid.linestyle": "--", "grid.color": "0.8"})
            palette = sns.color_palette("tab10", n_colors=len(algo_mask_mode_list))
            
            ax = axes[plot_idx]
            
            # 筛选需要绘制的数据
            df_plot = df_model[['eta', 'algo&mask_mode', var]].dropna()
            sns.lineplot(
                data=df_plot, 
                x='eta', 
                y=var, 
                hue='algo&mask_mode', 
                ax=ax, 
                palette=palette
            )
            
            # 设置图标题和轴标签
            ax.set_title(f"Model: {model} | {var.title()} vs Eta", fontsize=12)
            ax.set_xlabel('eta', fontsize=10)
            ax.set_ylabel(f'{var.title()}', fontsize=10)
            ax.tick_params(axis='x')
            
            # 只在第一个子图中显示图例
            if plot_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles, labels=labels, title='Algo & Mask Mode', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                ax.legend().set_visible(False)
            
            plot_idx += 1
    
    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    # 调整子图布局
    plt.tight_layout()
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show()

# 2.1 success_rate vs r
def plot_success_rate_vs_r(data, eta, algo, var, label_list, **kwargs):
    '''
    绘制不同模型下，各个标签（label）与参数（parameter）的关系图。
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
    
    # 获取唯一的模型和mask_mode
    model_list = kwargs.get('model_list', ['vit_b_16', 'resnet50', 'vgg16'])
    mask_mode_list = kwargs.get('mask_mode_list', df_filtered['mask_mode'].unique())
    
    picture_num = len(model_list) * len(label_list)
    
    ncols = kwargs.get('ncols', int(np.ceil(np.sqrt(picture_num))))
    nrows = kwargs.get('nrows', int(np.ceil(picture_num / ncols)))

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
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            
    # 隐藏多余的子图
    total_subplots = nrows * ncols
    used_subplots = len(model_list) * len(label_list)
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
            plt.subplots_adjust(bottom=0.2)
        elif locate == 'right':
            fig.legend(handles, labels, title='Mask Mode', loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=10)

    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
### 2.2. l1_norm和l2_norm vs success_rate
def plot_metrics_vs_success_rate_lines(data, eta, algo, label_list=['l1_norm', 'l2_norm'], **kwargs):
    '''
    绘制不同模型下，各个指标（如l1_norm和l2_norm）相对于success_rate的折线图。
    每个模型占据一行，每个指标占据一列，颜色表示不同的mask_mode。
    并为每个mask_mode绘制线性拟合的回归线，以虚线形式显示。

    Args:
        data (pd.DataFrame): 数据集
        eta (float): eta值
        algo (str): 攻击算法
        label_list (list): 需要绘制的指标列表，默认 ['l1_norm', 'l2_norm']
    '''
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    df_filtered = data[(data['eta'] == eta) & (data['algo'] == algo)].copy()
    df_filtered = df_filtered.reset_index(drop=True)

    if 'success_rate' in df_filtered.columns:
        df_filtered['success_rate'] = df_filtered['success_rate'].round(3) * 100

    model_list = df_filtered['model'].unique()
    mask_mode_list = df_filtered['mask_mode'].unique()
    
    picture_num = len(model_list) * len(label_list)
    nrows = int(np.ceil(np.sqrt(picture_num)))
    ncols = int(np.ceil(picture_num / nrows))
    
    sns.set(style="whitegrid", rc={"grid.linestyle": "--", "grid.color": "0.8"})
    palette = sns.color_palette("tab10", n_colors=len(mask_mode_list))
    mask_mode_colors = dict(zip(mask_mode_list, palette))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6.5, nrows*4), squeeze=False)
    axes = axes.flatten()

    for i, model in enumerate(model_list):
        df_model = df_filtered[df_filtered['model'] == model]
        for j, label in enumerate(label_list):
            ax = axes[i * len(label_list) + j]
            for mask_mode in mask_mode_list:
                df_mask = df_model[df_model['mask_mode'] == mask_mode]
                if df_mask.empty:
                    continue
                df_mask_sorted = df_mask.sort_values(by=label)
                
                sns.lineplot(
                    data=df_mask_sorted,
                    x=label,
                    y='success_rate',
                    label=mask_mode,
                    color=mask_mode_colors[mask_mode],
                    # marker='o',
                    ax=ax
                )
            
            ax.set_title(f"Model: {model} | {label} vs Success Rate", fontsize=14)
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Success Rate (%)', fontsize=12)
            ax.tick_params(axis='x')
      
            if i % 2 == 0 and j % 2 == 0:
                ax.legend(title='Mask Mode', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            else:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
    plt.tight_layout()
    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300)
    else:
        plt.show()

### 2.3. 3d图：l1_norm和l2_norm vs success_rate vs parameter
def plot_3d_scatter_plotly(data, eta, algo, metric='l1_norm'):
    '''
    使用Plotly绘制不同模型下，success_rate与parameter和指定metric（l1_norm或l2_norm）的交互式三维散点图。
    每个模型占据一张图，颜色表示不同的mask_mode。
    
    Args:
        data (pd.DataFrame): 数据集
        eta (float): eta值
        algo (str): 攻击算法
        metric (str): 指标名称，'l1_norm'或'l2_norm'
    '''
    df_filtered = data[(data['eta'] == eta) & (data['algo'] == algo)].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    if 'success_rate' in df_filtered.columns:
        df_filtered['success_rate'] = df_filtered['success_rate'].round(3) * 100
    
    model_list = df_filtered['model'].unique()
    mask_mode_list = df_filtered['mask_mode'].unique()
    
    palette = sns.color_palette("tab10", n_colors=len(mask_mode_list))
    mask_mode_colors = dict(zip(mask_mode_list, palette))
    
    for model in model_list:
        df_model = df_filtered[df_filtered['model'] == model]
        
        if metric not in df_model.columns:
            print(f"Warning: Metric '{metric}' not found in data for model '{model}'. Skipping.")
            continue
    
        fig = px.scatter_3d(
            df_model,
            x='parameter',
            y='success_rate',
            z=metric,
            color='mask_mode',
            title=f"Model: {model} | Success Rate vs Parameter and {metric}",
            labels={
                'parameter': 'Parameter',
                'success_rate': 'Success Rate (%)',
                metric: metric.replace('_', ' ').title()
            },
            opacity=0.7
        )
        fig.update_layout(legend_title_text='Mask Mode')
        fig.show()
        # fig.write_image(f'model_{model}_3D_success_rate_vs_parameter_{metric}.png')

### 3.1. loss vs pred_loss散点图
def plot_loss_vs_pred_loss(data, eta, algo, x='attack_loss', y='pred_loss', **kwargs):
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
    
    df_filtered = data[(data['eta'] == eta) & (data['algo'] == algo)].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    
    # 获取唯一的模型
    model_list = kwargs.get('model_list', ['vit_b_16', 'resnet50', 'vgg16'])
    mask_mode_list = kwargs.get('mask_mode_list', df_filtered['mask_mode'].unique())
    
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
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    # 调整子图布局
    plt.tight_layout()

    # 添加图例到整个图像的最下面
    if_legend = kwargs.get('if_legend', True)
    if if_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title='Mask Mode', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(mask_mode_list), fontsize=10)

    # 增加底部边距以避免图例挡住子图的坐标轴下方的文字
    plt.subplots_adjust(bottom=0.25)

    if output_path and save_name:
        plt.savefig(f'{output_path}/{save_name}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()



        
