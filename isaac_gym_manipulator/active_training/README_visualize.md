# 训练数据可视化

## 快速开始

训练完成后，使用以下命令生成可视化图表：

```bash
python3 visualize_training.py [debug_output路径]
```

如果不指定路径，脚本会自动使用最新的 debug_output 文件夹。

**示例：**
```bash
# 自动使用最新数据
python3 visualize_training.py

# 指定特定数据
python3 visualize_training.py debug_output/20251101_123456
```

## 生成的文件

所有图表保存在 `{debug_output}/visualizations/` 目录下：

1. **success_rate_iterations.png** - 成功率 vs 迭代次数
   - 显示模型学习进展
   - 原始数据 + 滑动平均

2. **mean_reward_iterations.png** - 平均奖励 vs 迭代次数
   - 奖励趋势

3. **episode_success_rate_distribution.png** - Episode 成功率散点图
   - 每次迭代成功率分布
   - 标记最大值和最小值

4. **mean_episode_length_iterations.png** - 平均episode长度 vs 迭代次数
   - 任务完成效率

5. **loss_curves.png** - Actor 和 Critic Loss
   - 学习稳定性指标

6. **combined_statistics.png** - 综合统计图
   - 成功率、奖励、长度、熵
   - 2x2 布局

## 特性

- ✅ Times New Roman 字体（如果可用）
- ✅ 滑动平均平滑曲线
- ✅ 高分辨率输出（300 DPI）
- ✅ 自动图例和网格
- ✅ 散点图标记最值

## 依赖

- pandas
- numpy
- matplotlib

## 参考

基于 `ur5e_DDPG_trajectory_planning_template` 的可视化风格实现。


