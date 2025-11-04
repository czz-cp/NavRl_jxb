"""
使用训练好的DDPG模型进行推理的脚本
"""
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ur5e_env import envCube
import torch
from DDPG import DDPG
import os

# 基本设置
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['axes.unicode_minus']=False

# MuJoCo设置
xml_path = './universal_robots_ur5e/scene.xml'
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo数据结构
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 可视化设置
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False

# 全局变量
state = np.zeros(6)
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model, data):
    """初始化控制器"""
    set_torque_servo(0, 1)
    set_torque_servo(1, 1)
    set_torque_servo(2, 1)
    set_torque_servo(3, 1)
    set_torque_servo(4, 1)
    set_torque_servo(5, 1)

def controller(model, data):
    """控制器函数"""
    data.ctrl[0] = -3500 * (data.qpos[0] - state[0]) - 100 * (data.qvel[0] - 0)
    data.ctrl[1] = -3500 * (data.qpos[1] - state[1]) - 100 * (data.qvel[1] - 0)
    data.ctrl[2] = -3500 * (data.qpos[2] - state[2]) - 100 * (data.qvel[2] - 0)
    data.ctrl[3] = -3000 * (data.qpos[3] - state[3]) - 100 * (data.qvel[3] - 0)
    data.ctrl[4] = -3000 * (data.qpos[4] - state[4]) - 100 * (data.qvel[4] - 0)
    data.ctrl[5] = -3000 * (data.qpos[5] - state[5]) - 100 * (data.qvel[5] - 0)

def set_torque_servo(actuator_no, flag):
    """设置力矩伺服器"""
    if flag == 0:
        model.actuator_gainprm[actuator_no, 0] = 0
    else:
        model.actuator_gainprm[actuator_no, 0] = 1

def keyboard(window, key, scancode, act, mods):
    """键盘回调函数"""
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    """鼠标按钮回调函数"""
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    """鼠标移动回调函数"""
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if (not button_left) and (not button_middle) and (not button_right):
        return

    width, height = glfw.get_window_size(window)
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    """滚轮回调函数"""
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

class DDPGInference:
    """DDPG推理类"""
    
    def __init__(self, model_path=None):
        """初始化推理环境"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.env = envCube()
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.max_action = [float(self.env.action_bound0[1]), float(self.env.action_bound1[1]), 
                          float(self.env.action_bound2[1]), float(self.env.action_bound3[1]),
                          float(self.env.action_bound4[1]), float(self.env.action_bound5[1])]
        self.max_action = torch.FloatTensor(self.max_action).to(self.device)
        
        # 创建DDPG智能体
        self.agent = DDPG(self.state_dim, self.action_dim, self.max_action)
        
        # 加载训练好的模型
        if model_path:
            self.load_model(model_path)
        else:
            print("警告：未指定模型路径，使用随机初始化的模型")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            # 查找最新的模型文件
            if os.path.isdir(model_path):
                actor_files = [f for f in os.listdir(model_path) if f.startswith('actor') and f.endswith('.pth')]
                if actor_files:
                    # 按奖励值排序，选择最好的模型
                    actor_files.sort(key=lambda x: float(x.split('_')[0][5:]), reverse=True)
                    best_actor = actor_files[0]
                    best_critic = best_actor.replace('actor', 'critic')
                    
                    actor_path = os.path.join(model_path, best_actor)
                    critic_path = os.path.join(model_path, best_critic)
                    
                    print(f"加载模型: {best_actor}")
                    self.agent.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                    self.agent.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                    print("模型加载成功！")
                else:
                    print("未找到模型文件")
            else:
                print(f"模型路径不存在: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
    
    def run_inference(self, num_episodes=5, visualize=True):
        """运行推理"""
        print(f"开始推理，共运行 {num_episodes} 个回合")
        
        for episode in range(num_episodes):
            print(f"\n=== 第 {episode + 1} 回合 ===")
            
            # 重置环境
            state = self.env.reset()
            for j in range(6):
                data.qpos[j] = state[j]
            mj.mj_forward(model, data)
            mj.mj_step(model, data)
            
            total_reward = 0
            step = 0
            done = False
            
            # 获取初始状态
            end_point = {'x': data.site_xpos[0][0], 'y': data.site_xpos[0][1], 'z': data.site_xpos[0][2]}
            goal = {'x': data.site_xpos[1][0], 'y': data.site_xpos[1][1], 'z': data.site_xpos[1][2]}
            dist4 = np.array([goal['x'] - end_point['x'], goal['y'] - end_point['y'], goal['z'] - end_point['z']])
            state = np.concatenate((state, dist4, [0.]), axis=0)
            
            target_init = [state[6], state[7], state[8]]
            
            while not done:
                # 获取当前状态
                end_point = {'x': data.site_xpos[0][0], 'y': data.site_xpos[0][1], 'z': data.site_xpos[0][2]}
                goal = {'x': data.site_xpos[1][0], 'y': data.site_xpos[1][1], 'z': data.site_xpos[1][2]}
                
                # 使用训练好的模型选择动作（不添加噪声）
                action = self.agent.select_action(state)
                action = action.clip(self.env.action_bound0[0], self.env.action_bound0[1])
                
                # 执行动作
                next_state, reward, done = self.env.step(action, target_init, goal, end_point)
                
                if step <= 998:
                    total_reward += reward
                
                # 更新状态
                state = next_state
                
                # 更新仿真
                time_prev = 0
                time = 0
                dt = 0.001
                while (time - time_prev < 1.0 / 60.0):
                    mj.mj_forward(model, data)
                    time += dt
                    mj.mj_step(model, data)
                
                if done:
                    break
                
                step += 1
                
                # 可视化
                if visualize:
                    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
                    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
                    
                    mj.mjv_updateScene(model, data, opt, None, cam,
                                       mj.mjtCatBit.mjCAT_ALL.value, scene)
                    mj.mjr_render(viewport, scene, context)
                    
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                    
                    if glfw.window_should_close(window):
                        return
            
            print(f"回合 {episode + 1} 完成，总奖励: {total_reward:.2f}，步数: {step}")
            print(f"末端点最终位置: ({end_point['x']:.3f}, {end_point['y']:.3f}, {end_point['z']:.3f})")
            print(f"目标点位置: ({goal['x']:.3f}, {goal['y']:.3f}, {goal['z']:.3f})")
            print(f"距离目标点: {np.sqrt((goal['x'] - end_point['x'])**2 + (goal['y'] - end_point['y'])**2 + (goal['z'] - end_point['z'])**2):.3f}")

def main():
    """主函数"""
    # 初始化GLFW
    glfw.init()
    window = glfw.create_window(1200, 900, "UR5e DDPG Inference", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # 设置回调函数
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)
    
    # 设置相机
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    cam.azimuth = 118.00000000000006
    cam.elevation = -52.800000000000004
    cam.distance = 2.840250715599666
    cam.lookat = np.array([-0.02409581650619193, 0.010921802046488856, 0.24125779903848252])
    
    # 初始化控制器
    init_controller(model, data)
    mj.set_mjcb_control(controller)
    
    # 创建推理对象
    model_path = './expur5e'  # 模型保存路径
    inference = DDPGInference(model_path)
    
    # 运行推理
    try:
        inference.run_inference(num_episodes=3, visualize=True)
    except KeyboardInterrupt:
        print("\n推理被用户中断")
    finally:
        glfw.terminate()

if __name__ == "__main__":
    main()
