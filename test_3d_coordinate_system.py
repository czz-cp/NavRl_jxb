#!/usr/bin/env python3
"""
测试完整3D方向的坐标系构建
Test Full 3D Coordinate System for Manipulator
"""
import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.append('/home/zar/Downloads/NavRL-main/isaac-training/training/scripts/')
from utils import vec_to_new_frame


def test_coordinate_system():
    """测试不同场景下的坐标系构建"""
    
    print("=" * 70)
    print("测试完整3D坐标系构建 - 机械臂版本")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    # 测试用例
    test_cases = [
        {
            'name': '水平前方',
            'goal_direction': torch.tensor([[1., 0., 0.]], device=device),
        },
        {
            'name': '水平右侧',
            'goal_direction': torch.tensor([[0., 1., 0.]], device=device),
        },
        {
            'name': '45度斜上方（重要！）',
            'goal_direction': torch.tensor([[0.707, 0., 0.707]], device=device),
        },
        {
            'name': '30度斜上方',
            'goal_direction': torch.tensor([[0.866, 0., 0.5]], device=device),
        },
        {
            'name': '正上方（奇异点测试）',
            'goal_direction': torch.tensor([[0., 0., 1.]], device=device),
        },
        {
            'name': '正下方（奇异点测试）',
            'goal_direction': torch.tensor([[0., 0., -1.]], device=device),
        },
        {
            'name': '接近垂直（z=0.95）',
            'goal_direction': torch.tensor([[0.1, 0., 0.95]], device=device),
        },
        {
            'name': '复杂方向',
            'goal_direction': torch.tensor([[0.5, 0.3, 0.8]], device=device),
        },
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"测试 {i+1}: {case['name']}")
        print(f"{'='*70}")
        
        goal_dir = case['goal_direction']
        goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True)
        
        # 计算坐标轴（与utils.py相同的逻辑）
        goal_dir_x = goal_dir
        
        z_component = goal_dir_x[..., 2:3]
        is_near_vertical = (torch.abs(z_component) > 0.9)
        
        reference = torch.where(
            is_near_vertical.expand_as(goal_dir_x),
            torch.tensor([1., 0., 0.], device=device).expand_as(goal_dir_x),
            torch.tensor([0., 0., 1.], device=device).expand_as(goal_dir_x)
        )
        
        goal_dir_y = torch.cross(reference, goal_dir_x, dim=-1)
        goal_dir_y = goal_dir_y / goal_dir_y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        
        goal_dir_z = torch.cross(goal_dir_x, goal_dir_y, dim=-1)
        goal_dir_z = goal_dir_z / goal_dir_z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        
        print(f"目标方向: [{goal_dir[0,0]:.4f}, {goal_dir[0,1]:.4f}, {goal_dir[0,2]:.4f}]")
        print(f"使用参考向量: {'[1,0,0] (垂直)' if is_near_vertical.item() else '[0,0,1] (水平)'}")
        print(f"\n新坐标系:")
        print(f"  X新: [{goal_dir_x[0,0]:.4f}, {goal_dir_x[0,1]:.4f}, {goal_dir_x[0,2]:.4f}]")
        print(f"  Y新: [{goal_dir_y[0,0]:.4f}, {goal_dir_y[0,1]:.4f}, {goal_dir_y[0,2]:.4f}]")
        print(f"  Z新: [{goal_dir_z[0,0]:.4f}, {goal_dir_z[0,1]:.4f}, {goal_dir_z[0,2]:.4f}]")
        
        # 检查正交性
        dot_xy = torch.sum(goal_dir_x * goal_dir_y, dim=-1).item()
        dot_xz = torch.sum(goal_dir_x * goal_dir_z, dim=-1).item()
        dot_yz = torch.sum(goal_dir_y * goal_dir_z, dim=-1).item()
        
        print(f"\n正交性检查:")
        print(f"  X·Y = {dot_xy:.8f} {'✅' if abs(dot_xy) < 1e-5 else '❌'}")
        print(f"  X·Z = {dot_xz:.8f} {'✅' if abs(dot_xz) < 1e-5 else '❌'}")
        print(f"  Y·Z = {dot_yz:.8f} {'✅' if abs(dot_yz) < 1e-5 else '❌'}")
        
        # 检查单位向量
        norm_x = goal_dir_x.norm(dim=-1).item()
        norm_y = goal_dir_y.norm(dim=-1).item()
        norm_z = goal_dir_z.norm(dim=-1).item()
        
        print(f"\n单位向量检查:")
        print(f"  |X| = {norm_x:.8f} {'✅' if abs(norm_x - 1.0) < 1e-5 else '❌'}")
        print(f"  |Y| = {norm_y:.8f} {'✅' if abs(norm_y - 1.0) < 1e-5 else '❌'}")
        print(f"  |Z| = {norm_z:.8f} {'✅' if abs(norm_z - 1.0) < 1e-5 else '❌'}")
        
        # 检查右手定则
        cross_xy = torch.cross(goal_dir_x, goal_dir_y, dim=-1)
        matches_z = torch.allclose(cross_xy, goal_dir_z, atol=1e-5)
        print(f"\n右手定则: X × Y = Z  {'✅' if matches_z else '❌'}")
        
        # 检查是否有NaN或Inf
        has_nan = (
            torch.isnan(goal_dir_x).any() or 
            torch.isnan(goal_dir_y).any() or 
            torch.isnan(goal_dir_z).any()
        )
        has_inf = (
            torch.isinf(goal_dir_x).any() or 
            torch.isinf(goal_dir_y).any() or 
            torch.isinf(goal_dir_z).any()
        )
        print(f"数值稳定性: NaN={has_nan} Inf={has_inf} {'✅' if not (has_nan or has_inf) else '❌'}")
        
        # 总结
        test_passed = (
            abs(dot_xy) < 1e-5 and
            abs(dot_xz) < 1e-5 and
            abs(dot_yz) < 1e-5 and
            abs(norm_x - 1.0) < 1e-5 and
            abs(norm_y - 1.0) < 1e-5 and
            abs(norm_z - 1.0) < 1e-5 and
            matches_z and
            not has_nan and
            not has_inf
        )
        
        print(f"\n{'✅ 测试通过' if test_passed else '❌ 测试失败'}")
        
        if not test_passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有测试通过！坐标系构建正确。")
    else:
        print("⚠️  部分测试失败，请检查代码。")
    print("=" * 70)
    
    return all_passed


def compare_with_horizontal_projection():
    """对比水平投影和完整3D方向"""
    
    print("\n" + "=" * 70)
    print("对比: 水平投影 vs 完整3D方向")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    # 测试场景：目标在45度斜上方
    ee_pos = torch.tensor([[0., 0., 0.]], device=device)
    target_pos = torch.tensor([[0.5, 0., 0.5]], device=device)
    rpos = target_pos - ee_pos
    
    print(f"\n场景: 末端在原点，目标在45度斜上方")
    print(f"  末端位置: {ee_pos[0].numpy()}")
    print(f"  目标位置: {target_pos[0].numpy()}")
    print(f"  相对位置: {rpos[0].numpy()}")
    
    # 方法1: 水平投影
    print(f"\n{'='*70}")
    print("方法1: 水平投影（无人机风格）")
    print(f"{'='*70}")
    
    target_dir_h = rpos.clone()
    target_dir_h[..., 2] = 0
    target_dir_h = target_dir_h / target_dir_h.norm(dim=-1, keepdim=True).clamp(1e-6)
    print(f"  目标方向: {target_dir_h[0].numpy()}")
    print(f"  → 投影到XY平面，丢失了Z分量")
    
    # 在新坐标系中表示原始相对位置
    rpos_normalized_h = rpos / rpos.norm(dim=-1, keepdim=True)
    rpos_new_h = vec_to_new_frame_manual(rpos_normalized_h, target_dir_h, horizontal=True)
    print(f"  新坐标系中的相对位置: {rpos_new_h[0].numpy()}")
    print(f"  → 目标在\"前方({rpos_new_h[0,0]:.3f}) + 上方({rpos_new_h[0,2]:.3f})\"")
    
    # 方法2: 完整3D
    print(f"\n{'='*70}")
    print("方法2: 完整3D方向（机械臂风格）")
    print(f"{'='*70}")
    
    target_dir_3d = rpos / rpos.norm(dim=-1, keepdim=True).clamp(1e-6)
    print(f"  目标方向: {target_dir_3d[0].numpy()}")
    print(f"  → 保留完整的3D方向信息")
    
    rpos_normalized_3d = rpos / rpos.norm(dim=-1, keepdim=True)
    rpos_new_3d = vec_to_new_frame_manual(rpos_normalized_3d, target_dir_3d, horizontal=False)
    print(f"  新坐标系中的相对位置: {rpos_new_3d[0].numpy()}")
    print(f"  → 目标在\"正前方\"（X新={rpos_new_3d[0,0]:.3f}）")
    
    print(f"\n{'='*70}")
    print("结论:")
    print(f"{'='*70}")
    print("  水平投影: 将3D运动分解为\"水平接近+垂直调整\"")
    print("  完整3D:   将3D运动统一为\"沿目标方向接近\"")
    print("  → 对于机械臂，完整3D更自然！")


def vec_to_new_frame_manual(vec, goal_direction, horizontal=False):
    """手动实现坐标变换（用于对比）"""
    device = vec.device
    
    goal_dir_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    
    if horizontal:
        # 水平投影方式
        reference = torch.tensor([0., 0., 1.], device=device).expand_as(goal_dir_x)
    else:
        # 完整3D方式
        z_component = goal_dir_x[..., 2:3]
        is_near_vertical = (torch.abs(z_component) > 0.9)
        reference = torch.where(
            is_near_vertical.expand_as(goal_dir_x),
            torch.tensor([1., 0., 0.], device=device).expand_as(goal_dir_x),
            torch.tensor([0., 0., 1.], device=device).expand_as(goal_dir_x)
        )
    
    goal_dir_y = torch.cross(reference, goal_dir_x, dim=-1)
    goal_dir_y = goal_dir_y / goal_dir_y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    goal_dir_z = torch.cross(goal_dir_x, goal_dir_y, dim=-1)
    goal_dir_z = goal_dir_z / goal_dir_z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    # 投影
    vec_x = torch.sum(vec * goal_dir_x, dim=-1, keepdim=True)
    vec_y = torch.sum(vec * goal_dir_y, dim=-1, keepdim=True)
    vec_z = torch.sum(vec * goal_dir_z, dim=-1, keepdim=True)
    
    return torch.cat([vec_x, vec_y, vec_z], dim=-1)


if __name__ == '__main__':
    print("=" * 70)
    print("机械臂3D坐标系测试")
    print("=" * 70)
    
    # 主要测试
    success = test_coordinate_system()
    
    # 对比测试
    compare_with_horizontal_projection()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ 所有测试通过！可以开始训练。")
        print("\n下一步:")
        print("  cd isaac-training/training/scripts/")
        print("  python train_manipulator.py")
    else:
        print("❌ 测试失败，请检查代码修改。")
    print("=" * 70)









