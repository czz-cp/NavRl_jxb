# 将项目推送到GitHub指南

## 前置准备

### 1. 配置Git用户信息（首次使用需要）

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**注意：** 如果只想为这个项目设置，去掉 `--global` 参数。

### 2. 确保已安装Git并配置好GitHub访问

- 如果使用HTTPS，需要GitHub Personal Access Token
- 如果使用SSH，需要配置SSH密钥

## 推送步骤

### 方法1：在GitHub网页上创建仓库（推荐）

1. **登录GitHub**，访问 https://github.com/new

2. **创建新仓库**：
   - Repository name: `NavRL-main` (或你喜欢的名字)
   - Description: 可选描述
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为本地已有代码）
   - 点击 "Create repository"

3. **推送代码**（在项目目录执行）：

```bash
cd /home/zar/Downloads/NavRL-main

# 如果还没有初始化git（已完成可跳过）
# git init
# git branch -m main

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: NavRL manipulator training with GPU support"

# 添加GitHub远程仓库（将 YOUR_USERNAME 和 REPO_NAME 替换为实际值）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 或者使用SSH（如果已配置SSH密钥）
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 推送到GitHub
git push -u origin main
```

### 方法2：使用GitHub CLI（如果已安装）

```bash
# 安装GitHub CLI后
gh repo create NavRL-main --public --source=. --remote=origin --push
```

## 处理嵌入式Git仓库警告

如果看到 "embedded git repository" 警告，有两种处理方式：

### 方式1：作为子模块添加（推荐，如果这些是外部依赖）

```bash
# 移除并重新添加为子模块（需要知道原始仓库URL）
git rm --cached isaac_gym_manipulator/ros_sources/Universal_Robots_ROS_Driver
git rm --cached isaac_gym_manipulator/ros_sources/rosdistro
git rm --cached isaac_gym_manipulator/ros_sources/universal_robot
git rm --cached visual_servo_py/visual_servo

# 然后作为子模块添加（需要知道URL）
git submodule add <URL> isaac_gym_manipulator/ros_sources/Universal_Robots_ROS_Driver
# ... 其他子模块
```

### 方式2：移除.git目录（如果这些只是本地代码）

```bash
# 移除这些目录的.git文件夹
rm -rf isaac_gym_manipulator/ros_sources/Universal_Robots_ROS_Driver/.git
rm -rf isaac_gym_manipulator/ros_sources/rosdistro/.git
rm -rf isaac_gym_manipulator/ros_sources/universal_robot/.git
rm -rf visual_servo_py/visual_servo/.git

# 然后重新添加
git add .
git commit -m "Remove embedded git repositories"
```

## 后续更新

推送代码后，后续更新代码到GitHub：

```bash
# 查看更改状态
git status

# 添加更改的文件
git add .

# 提交更改
git commit -m "描述你的更改"

# 推送到GitHub
git push
```

## 常见问题

### Q: 推送时提示需要认证？

**A:** 使用HTTPS需要Personal Access Token：
1. GitHub Settings → Developer settings → Personal access tokens → Generate new token
2. 选择权限：`repo`
3. 复制token，推送时在密码处输入token

或者配置SSH密钥（更安全）。

### Q: 如何推送大文件？

**A:** 如果文件超过100MB，考虑：
- 使用Git LFS: `git lfs install && git lfs track "*.pt" "*.pth"`
- 或者将大文件添加到 `.gitignore`

### Q: 如何忽略某些文件？

**A:** 已更新 `.gitignore`，包含：
- 训练输出 (`debug_output/`)
- 模型检查点 (`*.pt`, `*.pth`)
- Python缓存 (`__pycache__/`)
- 等等

如需添加更多，编辑 `.gitignore` 文件。

## 快速命令总结

```bash
# 1. 配置Git（首次）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 2. 初始化并提交（已完成）
git init
git branch -m main
git add .
git commit -m "Initial commit"

# 3. 添加远程仓库并推送
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

## 验证

推送成功后，访问你的GitHub仓库页面，应该能看到所有文件。

