#!/usr/bin/env python3
"""
简化验证脚本 - 不依赖PyTorch
Verify 3D Direction without PyTorch
"""
import numpy as np


def cross(a, b):
    """3D叉乘"""
    return np.cross(a, b)


def normalize(v):
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def build_coordinate_system(goal_direction):
    """
    构建完整3D坐标系
    与 utils.py 中的逻辑相同
    """
    # X新 = 目标方向
    x_new = normalize(goal_direction)
    
    # 选择参考向量（避免奇异点）
    z_component = abs(x_new[2])
    is_near_vertical = (z_component > 0.9)
    
    if is_near_vertical:
        reference = np.array([1., 0., 0.])  # 水平
    else:
        reference = np.array([0., 0., 1.])  # 垂直
    
    # Y新 = reference × X新
    y_new = cross(reference, x_new)
    y_new = normalize(y_new)
    
    # Z新 = X新 × Y新
    z_new = cross(x_new, y_new)
    z_new = normalize(z_new)
    
    return x_new, y_new, z_new, reference


def test_case(name, goal_direction):
    """测试一个案例"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    goal_dir = normalize(goal_direction)
    x_new, y_new, z_new, ref = build_coordinate_system(goal_dir)
    
    print(f"目标方向: [{goal_dir[0]:.4f}, {goal_dir[1]:.4f}, {goal_dir[2]:.4f}]")
    print(f"参考向量: {ref}")
    print(f"\n新坐标系:")
    print(f"  X新: [{x_new[0]:.4f}, {x_new[1]:.4f}, {x_new[2]:.4f}]")
    print(f"  Y新: [{y_new[0]:.4f}, {y_new[1]:.4f}, {y_new[2]:.4f}]")
    print(f"  Z新: [{z_new[0]:.4f}, {z_new[1]:.4f}, {z_new[2]:.4f}]")
    
    # 检查正交性
    dot_xy = np.dot(x_new, y_new)
    dot_xz = np.dot(x_new, z_new)
    dot_yz = np.dot(y_new, z_new)
    
    print(f"\n正交性检查:")
    print(f"  X·Y = {dot_xy:.8f} {'✅' if abs(dot_xy) < 1e-5 else '❌'}")
    print(f"  X·Z = {dot_xz:.8f} {'✅' if abs(dot_xz) < 1e-5 else '❌'}")
    print(f"  Y·Z = {dot_yz:.8f} {'✅' if abs(dot_yz) < 1e-5 else '❌'}")
    
    # 检查单位向量
    norm_x = np.linalg.norm(x_new)
    norm_y = np.linalg.norm(y_new)
    norm_z = np.linalg.norm(z_new)
    
    print(f"\n单位向量检查:")
    print(f"  |X| = {norm_x:.8f} {'✅' if abs(norm_x - 1.0) < 1e-5 else '❌'}")
    print(f"  |Y| = {norm_y:.8f} {'✅' if abs(norm_y - 1.0) < 1e-5 else '❌'}")
    print(f"  |Z| = {norm_z:.8f} {'✅' if abs(norm_z - 1.0) < 1e-5 else '❌'}")
    
    # 检查右手定则
    cross_xy = cross(x_new, y_new)
    matches_z = np.allclose(cross_xy, z_new, atol=1e-5)
    print(f"\n右手定则: X × Y = Z  {'✅' if matches_z else '❌'}")
    
    # 检查NaN
    has_nan = np.any(np.isnan(x_new)) or np.any(np.isnan(y_new)) or np.any(np.isnan(z_new))
    print(f"数值稳定性: {'✅' if not has_nan else '❌ 包含NaN'}")
    
    # 总结
    passed = (
        abs(dot_xy) < 1e-5 and
        abs(dot_xz) < 1e-5 and
        abs(dot_yz) < 1e-5 and
        abs(norm_x - 1.0) < 1e-5 and
        abs(norm_y - 1.0) < 1e-5 and
        abs(norm_z - 1.0) < 1e-5 and
        matches_z and
        not has_nan
    )
    
    print(f"\n{'✅ 测试通过' if passed else '❌ 测试失败'}")
    return passed


def main():
    """主测试"""
    print("=" * 70)
    print("机械臂完整3D坐标系验证")
    print("=" * 70)
    
    tests = [
        ("水平前方", np.array([1., 0., 0.])),
        ("水平右侧", np.array([0., 1., 0.])),
        ("45度斜上方 ⭐", np.array([0.707, 0., 0.707])),
        ("30度斜上方", np.array([0.866, 0., 0.5])),
        ("正上方（奇异点）⭐", np.array([0., 0., 1.])),
        ("正下方（奇异点）⭐", np.array([0., 0., -1.])),
        ("接近垂直 (z=0.95)", np.array([0.1, 0., 0.95])),
        ("复杂方向", np.array([0.5, 0.3, 0.8])),
    ]
    
    results = []
    for name, direction in tests:
        passed = test_case(name, direction)
        results.append(passed)
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed_count = sum(results)
    total_count = len(results)
    
    print(f"通过: {passed_count}/{total_count}")
    
    if all(results):
        print("\n🎉 所有测试通过！")
        print("\n✅ 完整3D坐标系构建正确")
        print("✅ 奇异点处理正常")
        print("✅ 数值稳定")
        print("\n可以开始训练了:")
        print("  cd isaac-training/training/scripts/")
        print("  python train_manipulator.py")
    else:
        print("\n⚠️  部分测试失败")
        failed_tests = [tests[i][0] for i, r in enumerate(results) if not r]
        print(f"失败的测试: {', '.join(failed_tests)}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()

