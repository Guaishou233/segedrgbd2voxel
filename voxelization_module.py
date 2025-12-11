#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体素化模块
将点云数据体素化，构建带语义的3D体素空间
"""

import os
import json
import struct
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

from color_palette import ColorPalette
from config_manager import ConfigManager


class Voxelizer:
    """体素化处理器"""
    
    def __init__(self, config: ConfigManager, palette: ColorPalette = None):
        """
        初始化体素化器
        
        Args:
            config: 配置管理器
            palette: 调色板
        """
        self.config = config
        self.palette = palette
        
        # 体素化参数
        self.voxel_size = config.get('voxelization.voxel_size', 64)
        self.physical_size = config.get('voxelization.physical_size', 1.26)
        
        # 输出配置
        self.output_folder = config.get('voxelization.output_folder', 'VOXELS')
        self.overwrite = config.get('voxelization.overwrite', False)
        
        # 点云输入文件夹
        self.pointcloud_folder = config.get('pointcloud.output_folder', 'POINTCLOUDS')
    
    def load_pointcloud(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载带语义信息的点云
        
        Args:
            ply_path: PLY文件路径
        
        Returns:
            points: Nx3 点云坐标
            colors: Nx3 RGB颜色
            semantics: Nx3 语义颜色
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
            for _ in range(num_points):
                # x, y, z (float)
                x, y, z = struct.unpack('<fff', f.read(12))
                # r, g, b (uchar)
                r, g, b = struct.unpack('<BBB', f.read(3))
                # semantic_r, semantic_g, semantic_b (uchar)
                semantic_r, semantic_g, semantic_b = struct.unpack('<BBB', f.read(3))
                
                # 过滤无效点
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or
                        np.isinf(x) or np.isinf(y) or np.isinf(z)):
                    points.append([x, y, z])
                    colors.append([r, g, b])
                    semantics.append([semantic_r, semantic_g, semantic_b])
        
        return np.array(points), np.array(colors), np.array(semantics)
    
    def voxelize(self, points: np.ndarray, colors: np.ndarray, 
                 semantics: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        将点云体素化
        
        Args:
            points: Nx3 点云坐标
            colors: Nx3 RGB颜色
            semantics: Nx3 语义颜色
        
        Returns:
            voxel_grid: 体素网格 (voxel_size, voxel_size, voxel_size)
            voxel_colors: 体素颜色 (voxel_size, voxel_size, voxel_size, 3)
            voxel_semantics: 体素语义颜色 (voxel_size, voxel_size, voxel_size, 3)
            voxel_origin: 体素原点世界坐标 (3,)
            voxel_resolution: 体素分辨率
        """
        # 计算点云边界框
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2.0
        
        # 计算体素空间原点
        voxel_origin = center - self.physical_size / 2.0
        
        # 体素分辨率
        voxel_resolution = self.physical_size / self.voxel_size
        
        # 初始化体素网格
        voxel_grid = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=bool)
        voxel_colors = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size, 3), dtype=np.uint8)
        voxel_semantics = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size, 3), dtype=np.uint8)
        voxel_counts = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.int32)
        
        # 用于收集语义标签（取众数）
        semantic_lists = {}
        
        # 将点云映射到体素空间
        for i in range(len(points)):
            point = points[i]
            color = colors[i]
            semantic = semantics[i]
            
            # 计算体素索引
            voxel_idx = ((point - voxel_origin) / voxel_resolution).astype(np.int32)
            
            # 检查是否在体素空间范围内
            if (voxel_idx >= 0).all() and (voxel_idx < self.voxel_size).all():
                x_idx, y_idx, z_idx = voxel_idx
                
                # 标记体素为占用
                voxel_grid[x_idx, y_idx, z_idx] = True
                
                # 累积颜色（使用平均值）
                voxel_counts[x_idx, y_idx, z_idx] += 1
                count = voxel_counts[x_idx, y_idx, z_idx]
                
                voxel_colors[x_idx, y_idx, z_idx] = (
                    (voxel_colors[x_idx, y_idx, z_idx] * (count - 1) + color) / count
                ).astype(np.uint8)
                
                # 收集语义颜色（使用众数）
                key = (x_idx, y_idx, z_idx)
                if key not in semantic_lists:
                    semantic_lists[key] = []
                semantic_lists[key].append(tuple(semantic))
        
        # 计算每个体素的语义众数
        for key, sem_list in semantic_lists.items():
            x_idx, y_idx, z_idx = key
            counter = Counter(sem_list)
            mode_sem, _ = counter.most_common(1)[0]
            voxel_semantics[x_idx, y_idx, z_idx] = mode_sem
        
        return voxel_grid, voxel_colors, voxel_semantics, voxel_origin, voxel_resolution
    
    def save_voxel_data(self, voxel_grid: np.ndarray, voxel_colors: np.ndarray,
                        voxel_semantics: np.ndarray, voxel_origin: np.ndarray,
                        voxel_resolution: float, output_dir: str,
                        timestamp: str) -> Dict:
        """
        保存体素数据
        
        Args:
            voxel_grid: 体素网格
            voxel_colors: 体素颜色
            voxel_semantics: 体素语义
            voxel_origin: 体素原点
            voxel_resolution: 体素分辨率
            output_dir: 输出目录
            timestamp: 时间戳
        
        Returns:
            metadata: 元数据字典
        """
        # 创建以时间戳命名的子目录
        voxel_dir = os.path.join(output_dir, timestamp)
        os.makedirs(voxel_dir, exist_ok=True)
        
        # 保存体素网格
        np.save(os.path.join(voxel_dir, "voxel_grid.npy"), voxel_grid)
        
        # 保存体素颜色
        np.save(os.path.join(voxel_dir, "voxel_colors.npy"), voxel_colors)
        
        # 保存体素语义
        np.save(os.path.join(voxel_dir, "voxel_semantics.npy"), voxel_semantics)
        
        # 保存元数据
        metadata = {
            "timestamp": timestamp,
            "voxel_size": int(voxel_grid.shape[0]),
            "physical_size": float(self.physical_size),
            "voxel_resolution": float(voxel_resolution),
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
        
        with open(os.path.join(voxel_dir, "voxel_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def process_pointcloud(self, ply_path: str, output_dir: str) -> Dict:
        """
        处理单个点云文件
        
        Args:
            ply_path: 点云文件路径
            output_dir: 输出目录
        
        Returns:
            处理结果
        """
        # 获取时间戳
        timestamp = os.path.splitext(os.path.basename(ply_path))[0]
        
        # 检查是否需要跳过
        voxel_dir = os.path.join(output_dir, timestamp)
        if not self.overwrite and os.path.exists(os.path.join(voxel_dir, "voxel_grid.npy")):
            return {'success': True, 'skipped': True, 'timestamp': timestamp}
        
        try:
            # 加载点云
            points, colors, semantics = self.load_pointcloud(ply_path)
            
            if len(points) == 0:
                return {'success': False, 'error': '空点云', 'timestamp': timestamp}
            
            # 体素化
            voxel_grid, voxel_colors, voxel_semantics, voxel_origin, voxel_resolution = \
                self.voxelize(points, colors, semantics)
            
            # 保存
            metadata = self.save_voxel_data(
                voxel_grid, voxel_colors, voxel_semantics,
                voxel_origin, voxel_resolution, output_dir, timestamp
            )
            
            return {
                'success': True,
                'skipped': False,
                'timestamp': timestamp,
                'occupied_voxels': metadata['occupied_voxels'],
                'occupied_ratio': metadata['occupied_ratio']
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': timestamp}
    
    def process_task(self, task_dir: str) -> Dict:
        """
        处理单个任务的体素化
        
        Args:
            task_dir: 任务目录路径
        
        Returns:
            处理结果统计
        """
        task_name = os.path.basename(task_dir)
        print(f"\n处理任务体素化: {task_name}")
        
        # 点云目录
        pointcloud_dir = os.path.join(task_dir, self.pointcloud_folder)
        if not os.path.exists(pointcloud_dir):
            return {'success': False, 'error': f'点云目录不存在: {pointcloud_dir}'}
        
        # 获取所有点云文件
        ply_files = sorted([f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')])
        
        if len(ply_files) == 0:
            return {'success': False, 'error': '无点云文件'}
        
        print(f"  找到 {len(ply_files)} 个点云文件")
        
        # 创建输出目录
        output_dir = os.path.join(task_dir, self.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理统计
        stats = {
            'success': True,
            'total_files': len(ply_files),
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0
        }
        
        # 处理每个点云文件
        print(f"  体素化处理...")
        
        for ply_file in tqdm(ply_files, desc="  体素化"):
            ply_path = os.path.join(pointcloud_dir, ply_file)
            result = self.process_pointcloud(ply_path, output_dir)
            
            if result.get('skipped'):
                stats['skipped_files'] += 1
            elif result.get('success'):
                stats['processed_files'] += 1
            else:
                stats['failed_files'] += 1
        
        return stats


def voxelize_dataset(config: ConfigManager, palette: ColorPalette = None) -> Dict:
    """
    体素化整个数据集
    
    Args:
        config: 配置管理器
        palette: 调色板
    
    Returns:
        处理结果统计
    """
    voxelizer = Voxelizer(config, palette)
    
    # 获取要处理的任务
    tasks = config.tasks
    if not tasks:
        print("没有找到要处理的任务")
        return {'success': False, 'error': '无任务'}
    
    print(f"找到 {len(tasks)} 个任务")
    
    # 处理结果
    results = {
        'success': True,
        'tasks': {}
    }
    
    # 处理每个任务
    for task_name in tasks:
        task_dir = config.get_task_dir(task_name)
        task_stats = voxelizer.process_task(str(task_dir))
        results['tasks'][task_name] = task_stats
    
    return results


if __name__ == "__main__":
    # 测试体素化模块
    from config_manager import load_config
    
    config = load_config()
    results = voxelize_dataset(config)
    
    print("\n体素化完成！")
    for task_name, task_stats in results.get('tasks', {}).items():
        print(f"  任务 {task_name}: {task_stats}")

