#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据集转换为COCO格式，包含深度图信息
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_image_size(image_path):
    """获取图像尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Cannot read image {image_path}: {e}")
        return None, None

def convert_to_coco_format(meta_json_path, rgb_dir, depth_dir, output_dir):
    """
    将数据集转换为COCO格式
    
    Args:
        meta_json_path: META.json文件路径
        rgb_dir: RGB图像目录
        depth_dir: DEPTH图像目录
        output_dir: 输出目录
    """
    # 读取META.json
    print("正在读取META.json...")
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建COCO格式的数据结构
    coco_data = {
        "info": {
            "description": "RGB-D Dataset in COCO format",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 处理每张图像
    print("正在处理图像...")
    images = meta_data.get("images", [])
    
    for idx, img_info in enumerate(tqdm(images, desc="转换图像")):
        rgb_filename = img_info.get("rgb_image", "")
        depth_filename = img_info.get("depth_image", "")
        
        if not rgb_filename:
            continue
        
        # 构建完整路径
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        depth_path = os.path.join(depth_dir, depth_filename) if depth_filename else None
        
        # 检查RGB图像是否存在
        if not os.path.exists(rgb_path):
            print(f"Warning: RGB image not found: {rgb_path}")
            continue
        
        # 获取图像尺寸
        width, height = get_image_size(rgb_path)
        if width is None or height is None:
            continue
        
        # 构建COCO格式的图像信息
        coco_image = {
            "id": idx + 1,  # COCO格式中ID从1开始
            "width": width,
            "height": height,
            "file_name": rgb_filename,
            # 添加深度图相关信息
            "depth_image": depth_filename,
            "depth_path": f"DEPTH/{depth_filename}" if depth_filename else None,
            # 保留原始元数据
            "timestamp": img_info.get("timestamp"),
            "camera_serial": img_info.get("camera_serial"),
            "intrinsic": img_info.get("intrinsic"),
            "extrinsic": img_info.get("extrinsic")
        }
        
        coco_data["images"].append(coco_image)
    
    # 保存COCO格式的JSON文件
    output_json_path = os.path.join(output_dir, "annotations.json")
    print(f"\n正在保存到 {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成！共处理 {len(coco_data['images'])} 张图像")
    print(f"输出文件: {output_json_path}")

if __name__ == "__main__":
    # 设置路径
    base_dir = "/data/tangqiansong/rgb2voxel/data"
    dataset_dir = os.path.join(base_dir, "dataset")
    meta_json_path = os.path.join(dataset_dir, "META.json")
    rgb_dir = os.path.join(dataset_dir, "RGB")
    depth_dir = os.path.join(dataset_dir, "DEPTH")
    output_dir = os.path.join(base_dir, "coco_sytle_dataset")
    
    # 执行转换
    convert_to_coco_format(meta_json_path, rgb_dir, depth_dir, output_dir)

