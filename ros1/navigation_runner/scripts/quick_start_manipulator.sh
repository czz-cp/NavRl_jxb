#!/bin/bash
# Quick Start Script for Manipulator Navigation
# 机械臂导航快速启动脚本

echo "========================================="
echo "UR10e Manipulator Navigation Quick Start"
echo "========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查ROS环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}Error: ROS not sourced!${NC}"
    echo "Please run: source /opt/ros/noetic/setup.bash"
    exit 1
fi

echo -e "${GREEN}✓ ROS environment detected: $ROS_DISTRO${NC}"

# 检查工作空间
if [ ! -f "devel/setup.bash" ]; then
    echo -e "${RED}Error: Not in catkin workspace!${NC}"
    echo "Please cd to your catkin_ws"
    exit 1
fi

echo -e "${GREEN}✓ Catkin workspace found${NC}"

# Source工作空间
source devel/setup.bash

# 检查必要的包
REQUIRED_PKGS=("map_manager" "onboard_detector" "navigation_runner" "visual_servo")
for pkg in "${REQUIRED_PKGS[@]}"; do
    if rospack find $pkg > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Package $pkg found${NC}"
    else
        echo -e "${RED}✗ Package $pkg not found${NC}"
        exit 1
    fi
done

# 检查检查点文件
CHECKPOINT="src/navigation_runner/scripts/ckpts/manipulator_checkpoint.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${YELLOW}⚠ Warning: Checkpoint not found at $CHECKPOINT${NC}"
    echo "Please train the model first or provide a checkpoint"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 启动选项
echo ""
echo "Select launch mode:"
echo "1) Full system (UR10e + Camera + Navigation)"
echo "2) Navigation only (UR already running)"
echo "3) Debug mode (with verbose output)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo -e "${GREEN}Starting full system...${NC}"
        roslaunch navigation_runner manipulator_navigation.launch
        ;;
    2)
        echo -e "${GREEN}Starting navigation only...${NC}"
        roslaunch navigation_runner manipulator_navigation.launch \
            skip_ur:=true
        ;;
    3)
        echo -e "${GREEN}Starting debug mode...${NC}"
        roslaunch navigation_runner manipulator_navigation.launch \
            debug:=true
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

