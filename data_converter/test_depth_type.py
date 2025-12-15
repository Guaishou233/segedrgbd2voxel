#!/usr/bin/env python3
"""
测试深度图类型：Z深度 vs 欧几里得深度（射线深度）

欧几里得深度（射线深度）：从相机中心到3D点的直线距离
Z深度（平面深度）：沿相机Z轴的距离（Open3D期望的格式）

如果深度图是欧几里得深度，边缘点的深度值会比实际Z深度更大，
导致反投影时产生弧形边界。
"""

import numpy as np
import json
from pathlib import Path

def analyze_depth_type():
    """分析深度数据类型"""
    
    # 加载相机参数
    camera_info_path = Path("/home/qiansongtang/Documents/program/dataset2/task_2/camera_info_default_session.json")
    with open(camera_info_path) as f:
        camera_info = json.load(f)
    
    cam = camera_info["cameras"][0]
    fx = cam["fx"]
    fy = cam["fy"]
    cx = cam["cx"]
    cy = cam["cy"]
    width, height = cam["resolution"]
    
    print(f"相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"分辨率: {width}x{height}")
    
    # 加载一个深度图 - 注意文件夹名是 "distance_to_camera"，表示欧几里得深度！
    depth_dir = Path("/home/qiansongtang/Documents/program/dataset2/task_2/basic_1765521294145/Replicator/distance_to_camera")
    depth_files = sorted(depth_dir.glob("*.npy"))
    
    if not depth_files:
        print("没有找到深度文件")
        return
    
    depth_path = depth_files[0]
    depth = np.load(depth_path)
    
    print(f"\n深度图: {depth_path.name}")
    print(f"深度范围: {depth.min():.4f} - {depth.max():.4f} 米")
    
    # 检查中心和边缘的深度差异
    center_y, center_x = height // 2, width // 2
    
    # 取中心区域的平均深度
    center_region = depth[center_y-10:center_y+10, center_x-10:center_x+10]
    center_depth = np.mean(center_region[center_region > 0])
    
    # 取四个角落的深度
    margin = 20
    corners = [
        depth[:margin, :margin],  # 左上
        depth[:margin, -margin:],  # 右上
        depth[-margin:, :margin],  # 左下
        depth[-margin:, -margin:]  # 右下
    ]
    corner_depths = [np.mean(c[c > 0]) if np.any(c > 0) else 0 for c in corners]
    
    print(f"\n中心深度: {center_depth:.4f} 米")
    print(f"角落深度: 左上={corner_depths[0]:.4f}, 右上={corner_depths[1]:.4f}, "
          f"左下={corner_depths[2]:.4f}, 右下={corner_depths[3]:.4f}")
    
    # 如果是平面（如墙壁或地板），Z深度应该在整个图像上相同
    # 但如果是欧几里得深度，边缘会更大
    
    # 计算理论上的欧几里得深度与Z深度的比值（在角落）
    corner_u = margin // 2
    corner_v = margin // 2
    
    # 左上角
    x_norm = (corner_u - cx) / fx
    y_norm = (corner_v - cy) / fy
    euclidean_factor = np.sqrt(x_norm**2 + y_norm**2 + 1)
    
    # 右下角
    corner_u = width - margin // 2
    corner_v = height - margin // 2
    x_norm = (corner_u - cx) / fx
    y_norm = (corner_v - cy) / fy
    euclidean_factor_rb = np.sqrt(x_norm**2 + y_norm**2 + 1)
    
    print(f"\n理论欧几里得/Z深度比值:")
    print(f"  左上角: {euclidean_factor:.4f}")
    print(f"  右下角: {euclidean_factor_rb:.4f}")
    
    # 检查实际数据
    if center_depth > 0:
        avg_corner_depth = np.mean([d for d in corner_depths if d > 0])
        actual_ratio = avg_corner_depth / center_depth if center_depth > 0 else 0
        print(f"\n实际角落/中心深度比值: {actual_ratio:.4f}")
        
        expected_ratio = (euclidean_factor + euclidean_factor_rb) / 2
        print(f"预期欧几里得深度比值: {expected_ratio:.4f}")
        
        if abs(actual_ratio - expected_ratio) < 0.1:
            print("\n结论: 深度图很可能是【欧几里得深度】（射线深度）")
            print("需要转换为Z深度以获得正确的点云")
        elif abs(actual_ratio - 1.0) < 0.1:
            print("\n结论: 深度图很可能是【Z深度】（平面深度）")
            print("不需要转换")
        else:
            print("\n结论: 无法确定深度类型，场景可能不是平面")
    
    # 检查是否是平面场景的简单方法：看深度图的变化是否符合欧几里得模式
    # 对于每个像素，计算如果是欧几里得深度，对应的Z深度
    print("\n\n=== 尝试将欧几里得深度转换为Z深度 ===")
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 计算归一化坐标
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    
    # 欧几里得深度到Z深度的转换因子
    euclidean_to_z_factor = 1.0 / np.sqrt(x_norm**2 + y_norm**2 + 1)
    
    # 转换
    z_depth = depth * euclidean_to_z_factor
    
    print(f"原始深度范围（假设欧几里得）: {depth.min():.4f} - {depth.max():.4f}")
    print(f"转换后Z深度范围: {z_depth.min():.4f} - {z_depth.max():.4f}")
    
    # 检查转换后的边缘是否更平
    z_center = z_depth[center_y-10:center_y+10, center_x-10:center_x+10]
    z_center_depth = np.mean(z_center[z_center > 0])
    
    z_corners = [
        z_depth[:margin, :margin],
        z_depth[:margin, -margin:],
        z_depth[-margin:, :margin],
        z_depth[-margin:, -margin:]
    ]
    z_corner_depths = [np.mean(c[c > 0]) if np.any(c > 0) else 0 for c in z_corners]
    
    print(f"\n转换后中心深度: {z_center_depth:.4f} 米")
    print(f"转换后角落深度: 左上={z_corner_depths[0]:.4f}, 右上={z_corner_depths[1]:.4f}, "
          f"左下={z_corner_depths[2]:.4f}, 右下={z_corner_depths[3]:.4f}")

if __name__ == "__main__":
    analyze_depth_type()

