#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证点云数据并显示统计信息
"""

import os
import numpy as np
import struct
import json

def load_pointcloud_info(ply_path):
    """加载点云基本信息"""
    points = []
    colors = []
    semantics = []
    
    with open(ply_path, 'rb') as f:
        # 读取header
        header_lines = []
        header_bytes = 0
        while True:
            line = f.readline()
            header_lines.append(line)
            header_bytes += len(line)
            if b'end_header' in line:
                break
        
        # 解析header获取点数
        num_points = 0
        for line in header_lines:
            if b'element vertex' in line:
                num_points = int(line.split()[-1])
                break
        
        # 读取所有点云数据
        for _ in range(num_points):
            # x, y, z (float) = 12 bytes
            x, y, z = struct.unpack('<fff', f.read(12))
            # r, g, b (uchar) = 3 bytes
            r, g, b = struct.unpack('<BBB', f.read(3))
            # semantic (uchar) = 1 byte
            semantic = struct.unpack('<B', f.read(1))[0]
            
            points.append([x, y, z])
            colors.append([r, g, b])
            semantics.append(semantic)
    
    return np.array(points), np.array(colors), np.array(semantics), num_points

def verify_pointclouds(dataset_dir, num_samples=10):
    """验证点云数据"""
    pointcloud_dir = os.path.join(dataset_dir, "POINTCLOUDS")
    meta_json_path = os.path.join(dataset_dir, "meta_info.json")
    
    # 读取meta_info
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 获取有点云信息的图像
    images_with_pc = [img for img in coco_data['images'] if 'pointcloud' in img]
    images_with_pc = images_with_pc[:num_samples]
    
    print(f"验证 {len(images_with_pc)} 个点云文件...\n")
    
    for i, img_info in enumerate(images_with_pc):
        pc_filename = img_info.get('pointcloud', '')
        if not pc_filename:
            continue
        
        pc_path = os.path.join(pointcloud_dir, pc_filename)
        
        if not os.path.exists(pc_path):
            print(f"{i+1}. {pc_filename}: 文件不存在")
            continue
        
        try:
            points, colors, semantics, total_points = load_pointcloud_info(pc_path)
            
            # 统计信息
            robot_arm_count = np.sum(semantics == 1)
            others_count = np.sum(semantics == 2)
            
            # 坐标范围
            x_range = [points[:, 0].min(), points[:, 0].max()]
            y_range = [points[:, 1].min(), points[:, 1].max()]
            z_range = [points[:, 2].min(), points[:, 2].max()]
            
            print(f"{i+1}. {pc_filename}")
            print(f"   总点数: {total_points:,}")
            print(f"   Robot arm: {robot_arm_count:,} ({robot_arm_count/len(semantics)*100:.1f}%)")
            print(f"   Others: {others_count:,} ({others_count/len(semantics)*100:.1f}%)")
            print(f"   坐标范围:")
            print(f"     X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
            print(f"     Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
            print(f"     Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
            print(f"   文件大小: {os.path.getsize(pc_path) / 1024 / 1024:.2f} MB")
            print()
        
        except Exception as e:
            print(f"{i+1}. {pc_filename}: 错误 - {e}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证点云数据")
    parser.add_argument("--num_samples", type=int, default=10, 
                       help="验证的样本数量")
    
    args = parser.parse_args()
    
    dataset_dir = "/data/tangqiansong/rgb2voxel/data/dataset"
    verify_pointclouds(dataset_dir, num_samples=args.num_samples)

