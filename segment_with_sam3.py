#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用SAM3对RGB图像进行语义分割
识别robot arm和others两个类别
"""

import os
import json
import shutil
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    print("警告: 无法导入sam3，请确保已安装SAM3")
    SAM3_AVAILABLE = False


def load_sam3_model(model_path=None):
    """加载SAM3模型"""
    if not SAM3_AVAILABLE:
        raise ImportError("SAM3未安装，请先安装SAM3")
    
    # 尝试从ModelScope加载
    if model_path is None:
        try:
            from modelscope import snapshot_download
            model_path = snapshot_download('facebook/sam3', cache_dir='./checkpoints')
            print(f"从ModelScope下载模型到: {model_path}")
        except Exception as e:
            print(f"从ModelScope下载失败: {e}")
            # 尝试默认路径（优先使用本地模型）
            possible_paths = [
                '/data/tangqiansong/Sam3/model',  # 绝对路径（优先）
                '../Sam3/model',  # 相对路径
                './checkpoints/sam3',
                './checkpoints',
                '~/.cache/modelscope/hub/facebook/sam3'
            ]
            model_path = None
            for path in possible_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    # 检查是否有模型文件
                    model_files = [f for f in os.listdir(expanded_path) if f.endswith(('.pth', '.pt', '.safetensors'))]
                    if model_files:
                        model_path = expanded_path
                        print(f"找到模型文件在: {model_path}")
                        break
            if model_path is None:
                model_path = './checkpoints'
                print(f"使用默认路径: {model_path} (如果不存在，将尝试自动下载)")
    
    print(f"加载SAM3模型从: {model_path}")
    
    # 如果model_path是目录，尝试找到模型文件
    checkpoint_path = None
    if os.path.isdir(model_path):
        # 查找模型文件
        for ext in ['.pth', '.pt', '.safetensors']:
            model_files = [f for f in os.listdir(model_path) if f.endswith(ext)]
            if model_files:
                checkpoint_path = os.path.join(model_path, model_files[0])
                print(f"找到模型文件: {checkpoint_path}")
                break
        if checkpoint_path is None:
            # 如果目录中没有模型文件，使用目录路径（让模型自动下载）
            checkpoint_path = model_path
    else:
        checkpoint_path = model_path
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建模型
    try:
        sam3_model = build_sam3_image_model(
            checkpoint_path=checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else None,
            device=device,
            eval_mode=True,
            load_from_HF=True  # 如果checkpoint_path不存在，从HuggingFace加载
        )
        # 确保模型在正确的设备上
        sam3_model = sam3_model.to(device)
        sam3_model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试不指定checkpoint_path，让模型自动下载...")
        sam3_model = build_sam3_image_model(
            checkpoint_path=None,
            device=device,
            eval_mode=True,
            load_from_HF=True
        )
        # 确保模型在正确的设备上
        sam3_model = sam3_model.to(device)
        sam3_model.eval()
    
    # 创建处理器
    processor = Sam3Processor(sam3_model, resolution=1008, device=device)
    
    return processor, device


def segment_image(processor, image_path, text_prompt="robot arm", device='cuda'):
    """
    使用SAM3对图像进行语义分割
    
    Args:
        processor: SAM3Processor实例
        image_path: 图像路径
        text_prompt: 文本提示词
        device: 设备
    
    Returns:
        mask: 分割mask (numpy array, 0=others, 1=robot arm)
        has_robot_arm: 是否检测到robot arm
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    h, w = image.size[1], image.size[0]  # PIL Image是(width, height)
    
    try:
        # 设置图像
        state = processor.set_image(image)
        
        # 设置文本提示并推理
        state = processor.set_text_prompt(text_prompt, state)
        
        # 获取masks和scores
        if 'masks' in state and len(state['masks']) > 0:
            # 获取所有mask (state['masks']是tensor，形状为[N, H, W])
            masks = state['masks']  # torch.Tensor, shape: [N, H, W]
            scores = state.get('scores', [])
            
            # 合并所有检测到的robot arm mask
            robot_arm_mask = np.zeros((h, w), dtype=np.uint8)
            
            # 转换为numpy数组
            if isinstance(masks, torch.Tensor):
                masks_np = masks.cpu().numpy()
            else:
                masks_np = np.array(masks)
            
            # 确保masks是3D的 [N, H, W]
            if len(masks_np.shape) == 2:
                masks_np = masks_np[np.newaxis, :, :]
            
            # 合并所有mask
            for i in range(masks_np.shape[0]):
                mask_np = masks_np[i]
                
                # 确保mask是2D的
                if len(mask_np.shape) > 2:
                    mask_np = mask_np.squeeze()
                
                # mask已经是原始图像尺寸，直接使用
                # 二值化 (masks已经是二值化的，但为了安全起见)
                mask_binary = (mask_np > 0.5).astype(np.uint8)
                robot_arm_mask = np.maximum(robot_arm_mask, mask_binary)
            
            # 创建完整的语义分割mask: 1=robot arm, 0=others
            seg_mask = robot_arm_mask.copy()
            has_robot_arm = seg_mask.sum() > 0
            
            return seg_mask, has_robot_arm
        else:
            # 没有检测到robot arm
            return np.zeros((h, w), dtype=np.uint8), False
            
    except Exception as e:
        print(f"分割图像时出错 {image_path}: {e}")
        import traceback
        traceback.print_exc()
        # 如果出错，返回全零mask
        return np.zeros((h, w), dtype=np.uint8), False


def process_dataset(seg_raw_dir, output_dir, test_mode=False, test_num=5):
    """
    处理数据集：对RGB图像进行语义分割，构建新数据集
    
    Args:
        seg_raw_dir: 原始数据集目录
        output_dir: 输出数据集目录
        test_mode: 是否为测试模式（只处理少量图像）
        test_num: 测试模式下处理的图像数量
    """
    rgb_dir = os.path.join(seg_raw_dir, 'RGB')
    depth_dir = os.path.join(seg_raw_dir, 'DEPTH')
    params_dir = os.path.join(seg_raw_dir, 'params')
    
    # 创建输出目录结构
    output_rgb_dir = os.path.join(output_dir, 'RGB')
    output_depth_dir = os.path.join(output_dir, 'DEPTH')
    output_seg_dir = os.path.join(output_dir, 'SEG')
    output_params_dir = os.path.join(output_dir, 'params')
    
    for dir_path in [output_rgb_dir, output_depth_dir, output_seg_dir, output_params_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 加载SAM3模型
    print("正在加载SAM3模型...")
    try:
        processor, device = load_sam3_model()
    except Exception as e:
        print(f"加载SAM3模型失败: {e}")
        print("尝试使用备用方法...")
        # 备用方法：直接使用模型路径
        processor, device = load_sam3_model(model_path='./checkpoints/sam3')
    
    # 获取所有RGB图像
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    if test_mode:
        rgb_files = rgb_files[:test_num]
        print(f"测试模式: 只处理前 {len(rgb_files)} 张RGB图像")
    else:
        print(f"找到 {len(rgb_files)} 张RGB图像")
    
    # 统计信息
    total_samples = 0
    kept_samples = 0
    skipped_samples = 0
    
    # 处理每张图像
    for rgb_file in tqdm(rgb_files, desc="处理图像"):
        total_samples += 1
        
        # 构建文件路径
        rgb_path = os.path.join(rgb_dir, rgb_file)
        
        # 对应的depth和params文件
        base_name = rgb_file.replace('.png', '')
        depth_file = rgb_file  # DEPTH和RGB文件名相同
        params_file = base_name + '.json'
        
        depth_path = os.path.join(depth_dir, depth_file)
        params_path = os.path.join(params_dir, params_file)
        
        # 检查文件是否存在
        if not os.path.exists(depth_path):
            print(f"警告: 深度图不存在 {depth_path}")
            skipped_samples += 1
            continue
        
        if not os.path.exists(params_path):
            print(f"警告: 参数文件不存在 {params_path}")
            skipped_samples += 1
            continue
        
        # 进行语义分割
        seg_mask, has_robot_arm = segment_image(processor, rgb_path, text_prompt="robot arm", device=device)
        
        # 如果没有检测到robot arm，跳过该样本
        if not has_robot_arm:
            skipped_samples += 1
            continue
        
        # 保存数据
        # 1. 复制RGB图像
        output_rgb_path = os.path.join(output_rgb_dir, rgb_file)
        shutil.copy2(rgb_path, output_rgb_path)
        
        # 2. 复制深度图
        output_depth_path = os.path.join(output_depth_dir, depth_file)
        shutil.copy2(depth_path, output_depth_path)
        
        # 3. 保存语义分割mask
        seg_file = base_name + '.png'
        output_seg_path = os.path.join(output_seg_dir, seg_file)
        seg_image = Image.fromarray(seg_mask * 255)  # 转换为0-255范围
        seg_image.save(output_seg_path)
        
        # 4. 复制并更新参数文件（添加语义分割信息）
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # 添加语义分割相关信息
        params['has_segmentation'] = True
        params['segmentation_classes'] = ['others', 'robot_arm']
        params['segmentation_file'] = seg_file
        
        output_params_path = os.path.join(output_params_dir, params_file)
        with open(output_params_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        kept_samples += 1
    
    # 打印统计信息
    print("\n" + "="*50)
    print("处理完成!")
    print(f"总样本数: {total_samples}")
    print(f"保留样本数: {kept_samples}")
    print(f"跳过样本数: {skipped_samples}")
    print(f"输出目录: {output_dir}")
    print("="*50)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用SAM3对RGB图像进行语义分割')
    parser.add_argument('--test', action='store_true', help='测试模式：只处理前5张图像')
    parser.add_argument('--test_num', type=int, default=5, help='测试模式下处理的图像数量')
    parser.add_argument('--input_dir', type=str, default='/data/tangqiansong/rgb2voxel/seg_raw', 
                        help='输入数据集目录')
    parser.add_argument('--output_dir', type=str, default='/data/tangqiansong/rgb2voxel/segmented_dataset',
                        help='输出数据集目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SAM3语义分割数据集构建")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.test:
        print(f"测试模式: 只处理前 {args.test_num} 张图像")
    print("="*60)
    
    process_dataset(args.input_dir, args.output_dir, test_mode=args.test, test_num=args.test_num)

