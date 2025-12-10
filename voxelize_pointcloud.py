#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将点云数据体素化处理
构建64x64x64的体素空间，范围为2米x2米x2米
"""

import os
import numpy as np
import struct
import json
from tqdm import tqdm

def load_pointcloud_with_semantics(ply_path):
    """
    加载带语义信息的点云
    
    Returns:
        points: Nx3 点云坐标
        colors: Nx3 RGB颜色
        semantics: Nx1 语义标签
    """
    points = []
    colors = []
    semantics = []
    
    with open(ply_path, 'rb') as f:
        # 读取header
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if b'end_header' in line:
                break
        
        # 解析header获取点数
        num_points = 0
        for line in header_lines:
            if b'element vertex' in line:
                num_points = int(line.split()[-1])
                break
        
        # 读取点云数据
        # 注意：虽然header声明semantic是1字节，但实际文件保存的是3字节
        for _ in tqdm(range(num_points), desc="读取点云数据"):
            # x, y, z (float)
            x, y, z = struct.unpack('<fff', f.read(12))
            # r, g, b (uchar)
            r, g, b = struct.unpack('<BBB', f.read(3))
            # semantic (实际保存为3字节，但header声明为1字节)
            # 使用第一个字节作为语义标签
            semantic_r, semantic_g, semantic_b = struct.unpack('<BBB', f.read(3))
            semantic = semantic_r  # 使用第一个字节作为语义标签
            
            # 过滤无效点（NaN或Inf）
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or 
                    np.isinf(x) or np.isinf(y) or np.isinf(z)):
                points.append([x, y, z])
                colors.append([r, g, b])
                semantics.append(semantic)
    
    points = np.array(points)
    colors = np.array(colors)
    semantics = np.array(semantics)
    
    print(f"有效点数: {len(points)} (过滤了 {num_points - len(points)} 个无效点)")
    
    return points, colors, semantics


def voxelize_pointcloud(points, colors, semantics, 
                       voxel_size=64, physical_size=2.0):
    """
    将点云体素化
    
    Args:
        points: Nx3 点云坐标
        colors: Nx3 RGB颜色
        semantics: Nx1 语义标签
        voxel_size: 体素网格大小 (64x64x64)
        physical_size: 物理空间大小（米），默认2.0米
    
    Returns:
        voxel_grid: 体素网格，形状为 (voxel_size, voxel_size, voxel_size)
        voxel_colors: 体素颜色，形状为 (voxel_size, voxel_size, voxel_size, 3)
        voxel_semantics: 体素语义，形状为 (voxel_size, voxel_size, voxel_size)
        voxel_origin: 体素空间原点的世界坐标 (3,)
        voxel_resolution: 每个体素的物理尺寸（米）
    """
    # 计算点云的边界框
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    
    print(f"\n点云边界框:")
    print(f"  最小: ({min_bound[0]:.4f}, {min_bound[1]:.4f}, {min_bound[2]:.4f})")
    print(f"  最大: ({max_bound[0]:.4f}, {max_bound[1]:.4f}, {max_bound[2]:.4f})")
    print(f"  中心: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    # 计算体素空间的原点（以中心为基准，向各个方向扩展physical_size/2）
    voxel_origin = center - physical_size / 2.0
    
    # 体素分辨率（每个体素的物理尺寸）
    voxel_resolution = physical_size / voxel_size
    
    print(f"\n体素空间配置:")
    print(f"  物理尺寸: {physical_size}m x {physical_size}m x {physical_size}m")
    print(f"  体素网格: {voxel_size}x{voxel_size}x{voxel_size}")
    print(f"  体素分辨率: {voxel_resolution:.6f}m/体素")
    print(f"  体素原点（世界坐标）: ({voxel_origin[0]:.4f}, {voxel_origin[1]:.4f}, {voxel_origin[2]:.4f})")
    
    
    # 初始化体素网格
    voxel_grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=bool)
    voxel_colors = np.zeros((voxel_size, voxel_size, voxel_size, 3), dtype=np.uint8)
    voxel_semantics = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.uint8)
    voxel_counts = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.int32)
    
    # 将点云映射到体素空间
    print(f"\n开始体素化...")
    for i in tqdm(range(len(points)), desc="处理点云"):
        point = points[i]
        color = colors[i]
        semantic = semantics[i]
        
        # 计算体素索引
        voxel_idx = ((point - voxel_origin) / voxel_resolution).astype(np.int32)
        
        # 检查是否在体素空间范围内
        if (voxel_idx >= 0).all() and (voxel_idx < voxel_size).all():
            x_idx, y_idx, z_idx = voxel_idx
            
            # 标记体素为占用
            voxel_grid[x_idx, y_idx, z_idx] = True
            
            # 累积颜色和语义（使用平均值）
            voxel_counts[x_idx, y_idx, z_idx] += 1
            count = voxel_counts[x_idx, y_idx, z_idx]
            
            # 更新颜色（使用累积平均）
            voxel_colors[x_idx, y_idx, z_idx] = (
                (voxel_colors[x_idx, y_idx, z_idx] * (count - 1) + color) / count
            ).astype(np.uint8)
            
            # 更新语义（使用众数或最后一个点的语义）
            # 统计每个体素索引下出现的所有语义标签，使用Counter取众数
            if voxel_counts[x_idx, y_idx, z_idx] == 1:
                # 第一次遇到该体素，建立一个列表用于收集语义标签
                voxel_semantics[x_idx, y_idx, z_idx] = semantic
                if "semantic_lists" not in locals():
                    semantic_lists = {}
                semantic_lists[(x_idx, y_idx, z_idx)] = [semantic]
            else:
                semantic_lists[(x_idx, y_idx, z_idx)].append(semantic)
                from collections import Counter
                counter = Counter(semantic_lists[(x_idx, y_idx, z_idx)])
                mode_sem, _ = counter.most_common(1)[0]
                voxel_semantics[x_idx, y_idx, z_idx] = mode_sem

    # 统计信息
    occupied_voxels = np.sum(voxel_grid)
    print(f"\n体素化完成:")
    print(f"  占用体素数: {occupied_voxels} / {voxel_size**3}")
    print(f"  占用率: {occupied_voxels / (voxel_size**3) * 100:.2f}%")
    
    return voxel_grid, voxel_colors, voxel_semantics, voxel_origin, voxel_resolution


def save_voxel_data(voxel_grid, voxel_colors, voxel_semantics, 
                   voxel_origin, voxel_resolution, output_dir):
    """
    保存体素数据
    
    Args:
        voxel_grid: 体素网格
        voxel_colors: 体素颜色
        voxel_semantics: 体素语义
        voxel_origin: 体素原点世界坐标
        voxel_resolution: 体素分辨率
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存体素网格（占用信息）
    voxel_grid_path = os.path.join(output_dir, "voxel_grid.npy")
    np.save(voxel_grid_path, voxel_grid)
    print(f"\n保存体素网格: {voxel_grid_path}")
    
    # 保存体素颜色
    voxel_colors_path = os.path.join(output_dir, "voxel_colors.npy")
    np.save(voxel_colors_path, voxel_colors)
    print(f"保存体素颜色: {voxel_colors_path}")
    
    # 保存体素语义
    voxel_semantics_path = os.path.join(output_dir, "voxel_semantics.npy")
    np.save(voxel_semantics_path, voxel_semantics)
    print(f"保存体素语义: {voxel_semantics_path}")
    
    # 保存元数据
    metadata = {
        "voxel_size": int(voxel_grid.shape[0]),
        "physical_size": 2.0,  # 米
        "voxel_resolution": float(voxel_resolution),  # 米/体素
        "voxel_origin": {
            "x": float(voxel_origin[0]),
            "y": float(voxel_origin[1]),
            "z": float(voxel_origin[2])
        },
        "voxel_origin_array": voxel_origin.tolist(),
        "occupied_voxels": int(np.sum(voxel_grid)),
        "total_voxels": int(voxel_grid.size),
        "occupied_ratio": float(np.sum(voxel_grid) / voxel_grid.size)
    }
    
    metadata_path = os.path.join(output_dir, "voxel_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"保存元数据: {metadata_path}")
    
    return metadata


def main():
    # 输入文件路径
    ply_file = "/home/qiansongtang/Documents/program/rgb2voxel/data/6cam_dataset/task_0001_user_0016_scene_0001_cfg_0003/POINTCLOUDS_MULTIVIEW/1631270646918.ply"
    
    # 输出目录
    output_dir = "/home/qiansongtang/Documents/program/rgb2voxel/data/6cam_dataset/task_0001_user_0016_scene_0001_cfg_0003/voxelized"
    
    # 体素化参数
    voxel_size = 64
    physical_size = 1.26  # 米
    
    
    print("=" * 60)
    print("点云体素化处理")
    print("=" * 60)
    print(f"输入文件: {ply_file}")
    print(f"输出目录: {output_dir}")
    print(f"体素网格: {voxel_size}x{voxel_size}x{voxel_size}")
    print(f"物理尺寸: {physical_size}m x {physical_size}m x {physical_size}m")
    print("=" * 60)
    
    # 1. 加载点云
    print("\n步骤 1: 加载点云数据...")
    points, colors, semantics = load_pointcloud_with_semantics(ply_file)
    print(f"加载完成: {len(points)} 个点")
    
    # 2. 体素化
    print("\n步骤 2: 体素化处理...")
    voxel_grid, voxel_colors, voxel_semantics, voxel_origin, voxel_resolution = \
        voxelize_pointcloud(points, colors, semantics, voxel_size, physical_size)
    
    # 3. 保存结果
    print("\n步骤 3: 保存体素数据...")
    metadata = save_voxel_data(voxel_grid, voxel_colors, voxel_semantics,
                              voxel_origin, voxel_resolution, output_dir)
    
    print("\n" + "=" * 60)
    print("体素化处理完成！")
    print("=" * 60)
    print(f"\n体素原点（世界坐标）: ({voxel_origin[0]:.4f}, {voxel_origin[1]:.4f}, {voxel_origin[2]:.4f})")
    print(f"体素分辨率: {voxel_resolution:.6f} 米/体素")
    print(f"占用体素数: {metadata['occupied_voxels']} / {metadata['total_voxels']}")
    print(f"占用率: {metadata['occupied_ratio'] * 100:.2f}%")
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

