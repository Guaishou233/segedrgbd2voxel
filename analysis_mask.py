#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义分割mask可视化工具
将类别ID映射为明显颜色，便于观察分割效果
"""

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def visualize_mask(mask_path, output_path=None, color_map=None):
    """
    将语义分割mask可视化，将类别ID映射为明显颜色
    
    Args:
        mask_path: 输入mask路径（PNG格式，像素值为类别ID）
        output_path: 输出可视化图片路径，如果为None则自动生成
        color_map: 颜色映射字典，格式为 {class_id: (R, G, B)}，如果为None则使用默认颜色
    
    Returns:
        vis_image: PIL Image对象（RGB彩色图像）
    """
    # 读取mask
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # 默认颜色映射
    if color_map is None:
        color_map = {
            1: (255, 0, 0),      # 红色 - cloths
            2: (0, 0, 255),      # 蓝色 - background
        }
    
    # 创建RGB图像
    height, width = mask.shape
    vis_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 根据类别ID填充颜色
    for class_id, color in color_map.items():
        vis_image[mask == class_id] = color
    
    # 转换为PIL Image
    vis_image = Image.fromarray(vis_image, mode='RGB')
    
    # 保存可视化图片
    if output_path is None:
        output_path = mask_path.replace('.png', '_vis.png')
    vis_image.save(output_path)
    
    return vis_image


def visualize_mask_with_overlay(image_path, mask_path, output_path=None, alpha=0.5, color_map=None):
    """
    将语义分割mask叠加到原图上进行可视化
    
    Args:
        image_path: 原始图像路径
        mask_path: mask路径（PNG格式，像素值为类别ID）
        output_path: 输出可视化图片路径，如果为None则自动生成
        alpha: mask的透明度（0-1之间），0表示完全透明，1表示完全不透明
        color_map: 颜色映射字典，格式为 {class_id: (R, G, B)}，如果为None则使用默认颜色
    
    Returns:
        overlay_image: PIL Image对象（RGB彩色图像）
    """
    # 读取原图和mask
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # 确保尺寸一致
    if image.shape[:2] != mask.shape:
        mask_pil = Image.fromarray(mask, mode='L')
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        mask = np.array(mask_pil)
    
    # 默认颜色映射
    if color_map is None:
        color_map = {
            1: (255, 0, 0),      # 红色 - cloths
            2: (0, 0, 255),      # 蓝色 - background
        }
    
    # 创建彩色mask
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color
    
    # 叠加到原图
    overlay_image = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    
    # 转换为PIL Image
    overlay_image = Image.fromarray(overlay_image, mode='RGB')
    
    # 保存可视化图片
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(mask_path)
        output_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    overlay_image.save(output_path)
    
    return overlay_image


def visualize_dataset(annotations_dir, images_dir=None, output_dir=None, color_map=None):
    """
    批量可视化数据集中的所有标注mask
    
    Args:
        annotations_dir: 标注mask目录
        images_dir: 原始图像目录（可选，如果提供则生成叠加可视化）
        output_dir: 输出可视化图片目录，如果为None则在annotations_dir下创建visualization子目录
        color_map: 颜色映射字典，格式为 {class_id: (R, G, B)}，如果为None则使用默认颜色
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(annotations_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有mask文件
    mask_files = glob.glob(os.path.join(annotations_dir, "*.png"))
    mask_files.sort()
    
    print(f"找到 {len(mask_files)} 个标注文件")
    print(f"可视化结果将保存到: {output_dir}")
    
    for mask_path in tqdm(mask_files, desc="生成可视化"):
        mask_filename = os.path.basename(mask_path)
        
        # 生成纯mask可视化
        vis_output = os.path.join(output_dir, mask_filename.replace('.png', '_vis.png'))
        visualize_mask(mask_path, vis_output, color_map)
        
        # 如果提供了图像目录，生成叠加可视化
        if images_dir is not None:
            # 尝试找到对应的原始图像
            image_filename = mask_filename.replace('.png', '.jpg')
            if not os.path.exists(os.path.join(images_dir, image_filename)):
                image_filename = mask_filename.replace('.png', '.jpeg')
            
            image_path = os.path.join(images_dir, image_filename)
            if os.path.exists(image_path):
                overlay_output = os.path.join(output_dir, mask_filename.replace('.png', '_overlay.png'))
                visualize_mask_with_overlay(image_path, mask_path, overlay_output, alpha=0.5, color_map=color_map)
    
    print(f"可视化完成！结果保存在: {output_dir}")


def analyze_mask_statistics(mask_path):
    """
    分析mask的统计信息
    
    Args:
        mask_path: mask文件路径
    
    Returns:
        stats: 包含统计信息的字典
    """
    mask = np.array(Image.open(mask_path).convert('L'))
    
    unique_values, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    stats = {
        "unique_classes": unique_values.tolist(),
        "class_counts": dict(zip(unique_values.tolist(), counts.tolist())),
        "class_percentages": {int(cls): float(count / total_pixels * 100) 
                             for cls, count in zip(unique_values, counts)},
        "total_pixels": int(total_pixels),
        "shape": mask.shape
    }
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="语义分割mask可视化工具")
    parser.add_argument("--annotations_dir", type=str, 
                        default="/data/tangqiansong/rgb2voxel/data/cloths_segmentation_dataset/annotations",
                        help="标注mask目录")
    parser.add_argument("--images_dir", type=str,
                        default="/data/tangqiansong/rgb2voxel/data/cloths_segmentation_dataset/images",
                        help="原始图像目录（可选，用于叠加可视化）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出可视化图片目录，如果为None则在annotations_dir下创建visualization子目录")
    parser.add_argument("--single_mask", type=str, default=None,
                        help="只可视化单个mask文件（用于测试）")
    parser.add_argument("--single_image", type=str, default=None,
                        help="单个mask对应的原始图像（用于叠加可视化）")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="叠加可视化的透明度（0-1之间）")
    
    args = parser.parse_args()
    
    # 定义颜色映射（cloths=红色，background=蓝色）
    color_map = {
        1: (255, 0, 0),      # 红色 - cloths
        2: (0, 0, 255),      # 蓝色 - background
    }
    
    if args.single_mask:
        # 单文件模式
        print(f"可视化单个mask: {args.single_mask}")
        visualize_mask(args.single_mask, color_map=color_map)
        
        if args.single_image:
            print(f"生成叠加可视化: {args.single_image}")
            visualize_mask_with_overlay(args.single_image, args.single_mask, 
                                      alpha=args.alpha, color_map=color_map)
        
        # 显示统计信息
        stats = analyze_mask_statistics(args.single_mask)
        print("\nMask统计信息:")
        print(f"  类别: {stats['unique_classes']}")
        print(f"  像素数: {stats['class_counts']}")
        print(f"  百分比: {stats['class_percentages']}")
    else:
        # 批量处理模式
        visualize_dataset(args.annotations_dir, args.images_dir, args.output_dir, color_map)

