#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 segmented_dataset 数据集的属性信息
包括 RGB图、Depth图、语义分割图和相机参数
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_image(image_path, image_type="RGB"):
    """分析图像的基本属性"""
    print(f"\n{'='*60}")
    print(f"分析 {image_type} 图像: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return None
    
    # 读取图像
    img = Image.open(image_path)
    
    # 基本属性
    print(f"文件路径: {image_path}")
    print(f"文件大小: {os.path.getsize(image_path) / 1024:.2f} KB")
    print(f"图像格式: {img.format}")
    print(f"图像模式: {img.mode}")
    print(f"图像尺寸: {img.size} (宽 x 高)")
    
    # 转换为numpy数组进行分析
    img_array = np.array(img)
    print(f"数组形状: {img_array.shape}")
    print(f"数据类型: {img_array.dtype}")
    
    if image_type == "RGB":
        print(f"通道数: {img_array.shape[2] if len(img_array.shape) == 3 else 1}")
        print(f"像素值范围: [{img_array.min()}, {img_array.max()}]")
        print(f"平均像素值: {img_array.mean():.2f}")
        print(f"标准差: {img_array.std():.2f}")
    elif image_type == "DEPTH":
        print(f"深度值范围: [{img_array.min()}, {img_array.max()}]")
        print(f"平均深度值: {img_array.mean():.2f}")
        print(f"标准差: {img_array.std():.2f}")
        # 统计非零像素（有效深度值）
        non_zero = np.count_nonzero(img_array)
        total = img_array.size
        print(f"有效深度像素比例: {non_zero/total*100:.2f}%")
    elif image_type == "SEG":
        unique_values = np.unique(img_array)
        total = img_array.size
        print(f"唯一标签值数量: {len(unique_values)}")
        print(f"标签值范围: [{unique_values.min()}, {unique_values.max()}]")
        print(f"唯一标签值: {unique_values[:20]}..." if len(unique_values) > 20 else f"唯一标签值: {unique_values}")
        # 统计每个标签的像素数量
        if len(unique_values) <= 20:
            for label in unique_values:
                count = np.sum(img_array == label)
                percentage = count / total * 100
                print(f"  标签 {label}: {count} 像素 ({percentage:.2f}%)")
    
    return {
        'path': image_path,
        'size': img.size,
        'mode': img.mode,
        'format': img.format,
        'shape': img_array.shape,
        'dtype': str(img_array.dtype),
        'min': int(img_array.min()),
        'max': int(img_array.max()),
        'mean': float(img_array.mean()),
        'std': float(img_array.std())
    }

def analyze_camera_params(params_path):
    """分析相机参数"""
    print(f"\n{'='*60}")
    print(f"分析相机参数: {os.path.basename(params_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(params_path):
        print(f"文件不存在: {params_path}")
        return None
    
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    print(f"文件路径: {params_path}")
    print(f"文件大小: {os.path.getsize(params_path) / 1024:.2f} KB")
    print(f"\n相机参数内容:")
    print(json.dumps(params, indent=2, ensure_ascii=False))
    
    # 提取关键参数
    if 'camera' in params:
        camera = params['camera']
        print(f"\n相机参数解析:")
        if 'intrinsic' in camera:
            intrinsic = camera['intrinsic']
            print(f"  内参矩阵 (K):")
            print(f"    fx: {intrinsic.get('fx', 'N/A')}")
            print(f"    fy: {intrinsic.get('fy', 'N/A')}")
            print(f"    cx: {intrinsic.get('cx', 'N/A')}")
            print(f"    cy: {intrinsic.get('cy', 'N/A')}")
        if 'extrinsic' in camera:
            extrinsic = camera['extrinsic']
            print(f"  外参矩阵:")
            print(f"    旋转矩阵 R: {extrinsic.get('rotation', 'N/A')}")
            print(f"    平移向量 t: {extrinsic.get('translation', 'N/A')}")
        if 'resolution' in camera:
            print(f"  分辨率: {camera['resolution']}")
    
    return params

def main():
    """主函数"""
    base_dir = "/data/tangqiansong/rgb2voxel/data/segmented_dataset"
    
    # 选择一个示例文件进行分析
    sample_name = "task_0001_user_0016_scene_0001_cfg_0003_036422060909_frame_000000"
    
    rgb_path = os.path.join(base_dir, "RGB", f"{sample_name}.png")
    depth_path = os.path.join(base_dir, "DEPTH", f"{sample_name}.png")
    seg_path = os.path.join(base_dir, "SEG", f"{sample_name}.png")
    params_path = os.path.join(base_dir, "params", f"{sample_name}.json")
    
    print("="*60)
    print("数据集属性分析报告")
    print("="*60)
    
    # 分析RGB图
    rgb_info = analyze_image(rgb_path, "RGB")
    
    # 分析Depth图
    depth_info = analyze_image(depth_path, "DEPTH")
    
    # 分析语义分割图
    seg_info = analyze_image(seg_path, "SEG")
    
    # 分析相机参数
    camera_params = analyze_camera_params(params_path)
    
    # 统计数据集规模
    print(f"\n{'='*60}")
    print("数据集统计信息")
    print(f"{'='*60}")
    
    rgb_dir = os.path.join(base_dir, "RGB")
    depth_dir = os.path.join(base_dir, "DEPTH")
    seg_dir = os.path.join(base_dir, "SEG")
    params_dir = os.path.join(base_dir, "params")
    
    rgb_count = len([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_count = len([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    seg_count = len([f for f in os.listdir(seg_dir) if f.endswith('.png')])
    params_count = len([f for f in os.listdir(params_dir) if f.endswith('.json')])
    
    print(f"RGB图像数量: {rgb_count}")
    print(f"Depth图像数量: {depth_count}")
    print(f"语义分割图像数量: {seg_count}")
    print(f"相机参数文件数量: {params_count}")
    
    # 检查数据一致性
    print(f"\n数据一致性检查:")
    if rgb_count == depth_count == seg_count == params_count:
        print(f"✓ 所有数据类型数量一致: {rgb_count} 个样本")
    else:
        print(f"⚠ 数据类型数量不一致:")
        print(f"  RGB: {rgb_count}, Depth: {depth_count}, SEG: {seg_count}, Params: {params_count}")
    
    print(f"\n{'='*60}")
    print("分析完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

