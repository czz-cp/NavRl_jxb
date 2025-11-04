#!/bin/bash

# GitHub推送脚本
# 使用方法: ./push_to_github.sh YOUR_USERNAME REPO_NAME

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  NavRL GitHub推送脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查参数
if [ $# -lt 2 ]; then
    echo -e "${RED}错误: 需要提供GitHub用户名和仓库名${NC}"
    echo "使用方法: $0 YOUR_USERNAME REPO_NAME"
    echo ""
    echo "示例: $0 zar NavRL-main"
    exit 1
fi

GITHUB_USER=$1
REPO_NAME=$2
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo -e "${YELLOW}配置信息:${NC}"
echo "  GitHub用户: $GITHUB_USER"
echo "  仓库名: $REPO_NAME"
echo "  仓库URL: $REPO_URL"
echo ""

# 检查Git配置
echo -e "${YELLOW}检查Git配置...${NC}"
if ! git config user.name > /dev/null 2>&1; then
    echo -e "${RED}错误: 未配置Git用户信息${NC}"
    echo "请先运行:"
    echo "  git config --global user.name \"Your Name\""
    echo "  git config --global user.email \"your.email@example.com\""
    exit 1
fi

echo -e "${GREEN}✓ Git用户: $(git config user.name)${NC}"
echo -e "${GREEN}✓ Git邮箱: $(git config user.email)${NC}"
echo ""

# 检查是否已初始化
if [ ! -d .git ]; then
    echo -e "${YELLOW}初始化Git仓库...${NC}"
    git init
    git branch -m main
fi

# 检查远程仓库
if git remote get-url origin > /dev/null 2>&1; then
    CURRENT_URL=$(git remote get-url origin)
    if [ "$CURRENT_URL" != "$REPO_URL" ]; then
        echo -e "${YELLOW}远程仓库已存在，但URL不同:${NC}"
        echo "  当前: $CURRENT_URL"
        echo "  新URL: $REPO_URL"
        read -p "是否更新远程仓库URL? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote set-url origin "$REPO_URL"
            echo -e "${GREEN}✓ 已更新远程仓库URL${NC}"
        fi
    else
        echo -e "${GREEN}✓ 远程仓库已配置: $REPO_URL${NC}"
    fi
else
    echo -e "${YELLOW}添加远程仓库...${NC}"
    git remote add origin "$REPO_URL"
    echo -e "${GREEN}✓ 已添加远程仓库${NC}"
fi
echo ""

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}发现未提交的更改，正在添加...${NC}"
    git add .
    echo -e "${GREEN}✓ 文件已添加到暂存区${NC}"
    
    # 检查是否有提交
    if ! git rev-parse --verify HEAD > /dev/null 2>&1; then
        COMMIT_MSG="Initial commit: NavRL manipulator training with GPU support"
    else
        COMMIT_MSG="Update: $(date +'%Y-%m-%d %H:%M:%S')"
    fi
    
    echo -e "${YELLOW}创建提交: ${COMMIT_MSG}${NC}"
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ 提交完成${NC}"
else
    echo -e "${GREEN}✓ 没有未提交的更改${NC}"
fi
echo ""

# 推送
echo -e "${YELLOW}推送到GitHub...${NC}"
echo -e "${YELLOW}注意: 如果提示输入密码，请输入GitHub Personal Access Token${NC}"
echo ""

# 检查当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "当前分支: $CURRENT_BRANCH"

# 尝试推送
if git push -u origin "$CURRENT_BRANCH" 2>&1; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ 推送成功！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "仓库地址: $REPO_URL"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}推送失败！${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "可能的原因:"
    echo "1. GitHub仓库不存在，请先在GitHub上创建仓库"
    echo "2. 认证失败，需要配置Personal Access Token或SSH密钥"
    echo "3. 网络问题"
    echo ""
    echo "请查看错误信息并参考 GITHUB_PUSH_GUIDE.md"
    exit 1
fi

