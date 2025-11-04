"""
简单的DDPG模型使用演示
"""
import numpy as np
import torch
from ur5e_env import envCube
from DDPG import DDPG
import os

def load_and_test_model(model_path='./expur5e', num_episodes=3):
    """
    加载训练好的模型并进行测试
    
    Args:
        model_path: 模型文件路径
        num_episodes: 测试回合数
    """
    print("=== DDPG模型测试演示 ===")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 初始化环境
    env = envCube()
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = [float(env.action_bound0[1]), float(env.action_bound1[1]), 
                  float(env.action_bound2[1]), float(env.action_bound3[1]),
                  float(env.action_bound4[1]), float(env.action_bound5[1])]
    max_action = torch.FloatTensor(max_action).to(device)
    
    # 创建DDPG智能体
    agent = DDPG(state_dim, action_dim, max_action)
    
    # 加载模型
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"模型路径不存在: {model_path}")
        print("使用随机初始化的模型进行演示")
    
    # 运行测试
    print(f"\n开始测试，共 {num_episodes} 个回合")
    print("-" * 50)
    
    for episode in range(num_episodes):
        print(f"\n第 {episode + 1} 回合:")
        
        # 重置环境
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        # 获取初始状态
        end_point = {'x': 0.5, 'y': 0.5, 'z': 0.5}  # 初始末端点位置
        goal = {'x': 0.5, 'y': 0.5, 'z': 0.5}       # 目标点位置
        dist4 = np.array([goal['x'] - end_point['x'], goal['y'] - end_point['y'], goal['z'] - end_point['z']])
        state = np.concatenate((state, dist4, [0.]), axis=0)
        
        target_init = [state[6], state[7], state[8]]
        
        while not done and step < 100:  # 限制最大步数
            # 使用模型选择动作（不添加噪声）
            action = agent.select_action(state)
            action = action.clip(env.action_bound0[0], env.action_bound0[1])
            
            # 执行动作
            next_state, reward, done = env.step(action, target_init, goal, end_point)
            
            total_reward += reward
            state = next_state
            step += 1
            
            # 更新末端点位置（简化版本）
            end_point['x'] += action[0] * 0.1  # 简化的位置更新
            end_point['y'] += action[1] * 0.1
            end_point['z'] += action[2] * 0.1
        
        # 计算最终距离
        final_distance = np.sqrt((goal['x'] - end_point['x'])**2 + 
                                (goal['y'] - end_point['y'])**2 + 
                                (goal['z'] - end_point['z'])**2)
        
        print(f"  步数: {step}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  最终距离目标: {final_distance:.3f}")
        print(f"  是否到达目标: {'是' if final_distance < 0.02 else '否'}")

def show_model_info(model_path='./expur5e'):
    """显示模型信息"""
    print("=== 模型信息 ===")
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return
    
    # 列出所有模型文件
    files = os.listdir(model_path)
    actor_files = [f for f in files if f.startswith('actor') and f.endswith('.pth')]
    critic_files = [f for f in files if f.startswith('critic') and f.endswith('.pth')]
    
    print(f"找到 {len(actor_files)} 个Actor模型文件:")
    for f in sorted(actor_files):
        print(f"  - {f}")
    
    print(f"找到 {len(critic_files)} 个Critic模型文件:")
    for f in sorted(critic_files):
        print(f"  - {f}")
    
    if actor_files:
        # 显示最佳模型
        actor_files.sort(key=lambda x: float(x.split('_')[0][5:]), reverse=True)
        best_model = actor_files[0]
        reward = best_model.split('_')[0][5:]
        episode = best_model.split('_')[1].split('.')[0]
        print(f"\n最佳模型: {best_model}")
        print(f"奖励值: {reward}")
        print(f"训练回合: {episode}")

if __name__ == "__main__":
    # 显示模型信息
    show_model_info()
    
    print("\n" + "="*60)
    
    # 运行测试
    load_and_test_model(num_episodes=3)
    
    print("\n=== 演示完成 ===")
    print("提示：")
    print("1. 如果模型路径不存在，程序会使用随机初始化的模型")
    print("2. 要使用训练好的模型，请确保模型文件在 './expur5e' 目录下")
    print("3. 可以使用 ur5e_inference.py 进行完整的可视化推理")


