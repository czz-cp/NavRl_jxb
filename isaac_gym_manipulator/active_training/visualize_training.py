"""
训练可视化脚本
从 debug_output 中的统计数据生成图表，包括：
- 成功率 vs 迭代次数
- 平均奖励 vs 迭代次数
- Episode 成功率散点图
- 平均episode长度 vs 迭代次数
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import sys
import glob
from pathlib import Path

# ------------------------------------------绘图字体初始化---------------------------------------------
# 字体初始化
# 尝试使用 Times New Roman，如果不可用则使用默认字体
font_family = 'Times New Roman'
try:
    plt.rc('font', family=font_family, size=12)  # 全局设置Times New Roman字体
    # 将公式的字体全部设置为常规字体，对本例而言就是Times New Roman字体
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = font_family
    matplotlib.rcParams['mathtext.it'] = f'{font_family}:italic'
    matplotlib.rcParams['mathtext.bf'] = f'{font_family}:bold'
except:
    # 如果 Times New Roman 不可用，使用默认字体
    font_family = 'DejaVu Sans'
    plt.rc('font', family=font_family, size=12)
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

# 字体设置
font1 = {'family': font_family,
         'weight': 'normal',
         'size': 14,
         }

font_title = {'family': font_family,
              'weight': 'normal',
              'size': 18,
              }
# ------------------------------------------绘图字体初始化---------------------------------------------

# ------------------------------------------滑动平均函数---------------------------------------------
def moving_average(a, window_size=50):
    """滑动平均函数，对有噪声的曲线进行降噪处理"""
    if len(a) < window_size:
        return a
    # 简化的滑动平均实现
    result = np.zeros(len(a))
    for i in range(len(a)):
        start = max(0, i - window_size // 2)
        end = min(len(a), i + window_size // 2 + 1)
        result[i] = np.mean(a[start:end])
    return result


# ------------------------------------------主函数---------------------------------------------
def visualize_training(debug_output_path):
    """
    可视化训练数据
    
    Args:
        debug_output_path: debug_output 文件夹路径，例如 'debug_output/20251101_220519'
    """
    print(f"📊 开始可视化训练数据: {debug_output_path}")
    
    # 读取数据
    stats_dir = os.path.join(debug_output_path, 'stats')
    
    # 读取训练统计数据
    train_stats_files = glob.glob(os.path.join(stats_dir, 'training_stats_*.csv'))
    if not train_stats_files:
        print(f"❌ 未找到训练统计数据文件")
        return
    
    train_stats_file = train_stats_files[0]
    print(f"📁 读取训练统计数据: {train_stats_file}")
    train_df = pd.read_csv(train_stats_file)
    
    # 读取episode统计数据
    episode_stats_files = glob.glob(os.path.join(stats_dir, 'episode_stats_*.csv'))
    if not episode_stats_files:
        print(f"❌ 未找到episode统计数据文件")
        return
    
    episode_stats_file = episode_stats_files[0]
    print(f"📁 读取episode统计数据: {episode_stats_file}")
    episode_df = pd.read_csv(episode_stats_file)
    
    # 创建输出目录
    output_dir = os.path.join(debug_output_path, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1️⃣ 成功率 vs 迭代次数
    print("📈 绘制成功率曲线...")
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = train_df['iteration'].values
    success_rates = train_df['success_rate'].values * 100  # 转换为百分比
    
    # 绘制原始数据
    ax.plot(iterations, success_rates, linewidth=1, alpha=0.5, color='lightblue', label='Raw')
    # 绘制滑动平均
    window = min(50, len(success_rates) // 10) if len(success_rates) > 10 else 10
    smoothed_rates = moving_average(success_rates, window_size=window)
    ax.plot(iterations, smoothed_rates, linewidth=2, color='C0', label='Smoothed')
    
    ax.set_title('Success Rate vs Iterations', fontdict=font_title)
    ax.set_xlabel('Iterations', fontdict=font1)
    ax.set_ylabel('Success Rate (%)', fontdict=font1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'success_rate_iterations.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {filename}")
    
    # 2️⃣ 平均奖励 vs 迭代次数
    print("📈 绘制平均奖励曲线...")
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_rewards = train_df['mean_reward'].values
    
    # 绘制原始数据
    ax.plot(iterations, mean_rewards, linewidth=1, alpha=0.5, color='lightgreen', label='Raw')
    # 绘制滑动平均
    smoothed_rewards = moving_average(mean_rewards, window_size=window)
    ax.plot(iterations, smoothed_rewards, linewidth=2, color='C1', label='Smoothed')
    
    ax.set_title('Mean Reward vs Iterations', fontdict=font_title)
    ax.set_xlabel('Iterations', fontdict=font1)
    ax.set_ylabel('Mean Reward', fontdict=font1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'mean_reward_iterations.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {filename}")
    
    # 3️⃣ Episode 成功率散点图
    print("📈 绘制episode成功率散点图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按迭代分组计算成功率
    episode_df['success_int'] = episode_df['success'].astype(int)
    success_by_iter = episode_df.groupby('iteration')['success_int'].agg(['sum', 'count'])
    success_by_iter['success_rate'] = success_by_iter['sum'] / success_by_iter['count'] * 100
    
    iterations_ep = success_by_iter.index.values
    success_rates_ep = success_by_iter['success_rate'].values
    
    # 找到最大最小值
    max_idx = np.argmax(success_rates_ep)
    min_idx = np.argmin(success_rates_ep)
    
    ax.scatter(iterations_ep, success_rates_ep, s=10, alpha=0.6, color='C2')
    # 标记最大最小值
    ax.plot(iterations_ep[max_idx], success_rates_ep[max_idx], 'o', color='r', markersize=10)
    ax.plot(iterations_ep[min_idx], success_rates_ep[min_idx], 'o', color='b', markersize=10)
    ax.text(iterations_ep[max_idx] - 50, success_rates_ep[max_idx] + 5,
            f'max({iterations_ep[max_idx]},{success_rates_ep[max_idx]:.1f}%)',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(iterations_ep[min_idx] - 50, success_rates_ep[min_idx] - 5,
            f'min({iterations_ep[min_idx]},{success_rates_ep[min_idx]:.1f}%)',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Episode Success Rate Distribution', fontdict=font_title)
    ax.set_xlabel('Iterations', fontdict=font1)
    ax.set_ylabel('Success Rate (%)', fontdict=font1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'episode_success_rate_distribution.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {filename}")
    
    # 4️⃣ 平均episode长度 vs 迭代次数
    print("📈 绘制平均episode长度曲线...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 从训练统计数据中获取
    mean_lengths = train_df['mean_episode_length'].values
    
    # 绘制原始数据
    ax.plot(iterations, mean_lengths, linewidth=1, alpha=0.5, color='lightcoral', label='Raw')
    # 绘制滑动平均
    smoothed_lengths = moving_average(mean_lengths, window_size=window)
    ax.plot(iterations, smoothed_lengths, linewidth=2, color='C3', label='Smoothed')
    
    ax.set_title('Mean Episode Length vs Iterations', fontdict=font_title)
    ax.set_xlabel('Iterations', fontdict=font1)
    ax.set_ylabel('Mean Episode Length (Steps)', fontdict=font1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'mean_episode_length_iterations.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {filename}")
    
    # 5️⃣ Actor 和 Critic Loss
    print("📈 绘制Loss曲线...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Actor Loss
    ax1.plot(iterations, train_df['actor_loss'].values, linewidth=1, alpha=0.5, color='lightblue')
    smoothed_actor = moving_average(train_df['actor_loss'].values, window_size=window)
    ax1.plot(iterations, smoothed_actor, linewidth=2, color='C0')
    ax1.set_title('Actor Loss', fontdict=font_title)
    ax1.set_ylabel('Loss', fontdict=font1)
    ax1.grid(True, alpha=0.3)
    
    # Critic Loss
    ax2.plot(iterations, train_df['critic_loss'].values, linewidth=1, alpha=0.5, color='lightcoral')
    smoothed_critic = moving_average(train_df['critic_loss'].values, window_size=window)
    ax2.plot(iterations, smoothed_critic, linewidth=2, color='C1')
    ax2.set_title('Critic Loss', fontdict=font_title)
    ax2.set_xlabel('Iterations', fontdict=font1)
    ax2.set_ylabel('Loss', fontdict=font1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {filename}")
    
    # 6️⃣ 综合统计图
    print("📈 绘制综合统计图...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 成功率
    ax1.plot(iterations, success_rates, linewidth=1, alpha=0.5, color='lightblue')
    ax1.plot(iterations, smoothed_rates, linewidth=2, color='C0')
    ax1.set_title('Success Rate', fontdict=font_title)
    ax1.set_xlabel('Iterations', fontdict=font1)
    ax1.set_ylabel('Success Rate (%)', fontdict=font1)
    ax1.grid(True, alpha=0.3)
    
    # 平均奖励
    ax2.plot(iterations, mean_rewards, linewidth=1, alpha=0.5, color='lightgreen')
    ax2.plot(iterations, smoothed_rewards, linewidth=2, color='C1')
    ax2.set_title('Mean Reward', fontdict=font_title)
    ax2.set_xlabel('Iterations', fontdict=font1)
    ax2.set_ylabel('Mean Reward', fontdict=font1)
    ax2.grid(True, alpha=0.3)
    
    # Episode长度
    ax3.plot(iterations, mean_lengths, linewidth=1, alpha=0.5, color='lightcoral')
    ax3.plot(iterations, smoothed_lengths, linewidth=2, color='C3')
    ax3.set_title('Mean Episode Length', fontdict=font_title)
    ax3.set_xlabel('Iterations', fontdict=font1)
    ax3.set_ylabel('Length (Steps)', fontdict=font1)
    ax3.grid(True, alpha=0.3)
    
    # Entropy
    entropy_values = train_df['entropy'].values
    ax4.plot(iterations, entropy_values, linewidth=1, alpha=0.5, color='lightyellow')
    smoothed_entropy = moving_average(entropy_values, window_size=window)
    ax4.plot(iterations, smoothed_entropy, linewidth=2, color='C4')
    ax4.set_title('Entropy', fontdict=font_title)
    ax4.set_xlabel('Iterations', fontdict=font1)
    ax4.set_ylabel('Entropy', fontdict=font1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'combined_statistics.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {filename}")
    
    print("\n" + "="*80)
    print(f"✅ 所有可视化图表已保存到: {output_dir}")
    print("="*80)


# ------------------------------------------主程序入口---------------------------------------------
if __name__ == "__main__":
    # 获取命令行参数或使用默认路径
    if len(sys.argv) > 1:
        debug_output_path = sys.argv[1]
    else:
        # 默认使用最新的 debug_output 文件夹
        debug_output_base = os.path.join(os.path.dirname(__file__), 'debug_output')
        debug_output_dirs = sorted(glob.glob(os.path.join(debug_output_base, '*')))
        if debug_output_dirs:
            debug_output_path = debug_output_dirs[-1]  # 使用最新的
        else:
            print("❌ 未找到 debug_output 文件夹")
            sys.exit(1)
    
    # 可视化
    visualize_training(debug_output_path)

