#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义分割模块
使用SAM3对多视角RGB图像进行语义分割，支持可配置的调色板
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import glob
from typing import Dict, List, Tuple, Optional, Any

from color_palette import ColorPalette, create_palette_from_targets
from config_manager import ConfigManager

# 添加 Sam3 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_dir = os.path.join(current_dir, "Sam3")
if sam3_dir not in sys.path:
    sys.path.insert(0, sam3_dir)

# 检查是否可以使用transformers库
try:
    from transformers import Sam3Processor, Sam3Model
    USE_TRANSFORMERS = True
except ImportError:
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        USE_TRANSFORMERS = False
    except ImportError:
        raise ImportError("需要安装transformers或sam3库")

# 检查pycocotools
try:
    import pycocotools.mask as mask_util
    USE_PYCOCOTOOLS = True
except ImportError:
    USE_PYCOCOTOOLS = False
    print("Warning: pycocotools未安装")


class SemanticSegmenter:
    """语义分割器"""
    
    def __init__(self, config: ConfigManager, palette: ColorPalette = None):
        """
        初始化语义分割器
        
        Args:
            config: 配置管理器
            palette: 调色板（可选）
        """
        self.config = config
        self.device = config.get('segmentation.device', 'cuda')
        self.model_path = config.get('segmentation.model_path')
        self.threshold = config.get('segmentation.threshold', 0.3)
        self.mask_threshold = config.get('segmentation.mask_threshold', 0.3)
        self.overwrite = config.get('segmentation.overwrite', False)
        
        # 获取分割目标
        self.targets = config.get('segmentation.targets', [])
        self.background = config.get('segmentation.background', {'label': 'background', 'id': 2})
        
        # 创建或使用调色板
        if palette is None:
            color_config = config.get('color_palette', {})
            self.palette = create_palette_from_targets(
                self.targets, self.background, color_config
            )
        else:
            self.palette = palette
        
        # 模型（延迟加载）
        self.model = None
        self.processor = None
    
    def _load_model(self):
        """加载SAM3模型"""
        if self.model is not None:
            return
        
        print(f"正在加载SAM3模型: {self.model_path}")
        
        if USE_TRANSFORMERS:
            self.model = Sam3Model.from_pretrained(self.model_path).to(self.device)
            self.processor = Sam3Processor.from_pretrained(self.model_path)
        else:
            checkpoint_path = None
            if os.path.isdir(self.model_path):
                sam3_pt_path = os.path.join(self.model_path, "sam3.pt")
                if os.path.exists(sam3_pt_path):
                    checkpoint_path = sam3_pt_path
            elif os.path.isfile(self.model_path):
                checkpoint_path = self.model_path
            
            self.model = build_sam3_image_model(
                bpe_path=None,
                device=self.device,
                checkpoint_path=checkpoint_path,
                load_from_HF=(checkpoint_path is None)
            )
            self.processor = Sam3Processor(self.model)
        
        self.model.eval()
        print("SAM3模型加载完成！")
    
    def segment_image(self, image_path: str, 
                      target_prompts: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        对单张图像进行语义分割
        
        Args:
            image_path: 图像路径
            target_prompts: 目标文本提示列表（如果为None，使用配置中的targets）
        
        Returns:
            seg_mask: 分割mask，值为类别ID
            annotations: 分割结果的详细信息
        """
        self._load_model()
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # 创建分割mask（默认为背景）
        bg_id = self.background['id']
        seg_mask = np.ones((height, width), dtype=np.uint8) * bg_id
        
        # 存储所有目标的mask
        all_masks = {}
        annotations = {
            'image_path': image_path,
            'width': width,
            'height': height,
            'targets': []
        }
        
        # 获取目标提示
        if target_prompts is None:
            targets_to_segment = self.targets
        else:
            targets_to_segment = [
                {'label': p, 'prompt': p, 'id': i+1} 
                for i, p in enumerate(target_prompts)
            ]
        
        # 对每个目标进行分割
        for target in targets_to_segment:
            prompt = target.get('prompt', target['label'])
            class_id = target['id']
            
            masks, boxes, scores = self._segment_with_prompt(image, prompt)
            
            if len(masks) > 0:
                # 合并所有检测到的mask
                combined_mask = self._combine_masks(
                    masks, boxes, scores, (width, height)
                )
                all_masks[class_id] = combined_mask
                
                # 记录注释信息
                annotations['targets'].append({
                    'label': target['label'],
                    'prompt': prompt,
                    'class_id': class_id,
                    'num_detections': len(masks),
                    'avg_score': float(np.mean([
                        s.item() if isinstance(s, torch.Tensor) else s 
                        for s in scores
                    ]))
                })
        
        # 按优先级应用mask（后面的类别覆盖前面的）
        for class_id in sorted(all_masks.keys()):
            mask = all_masks[class_id]
            seg_mask[mask] = class_id
        
        return seg_mask, annotations
    
    def _segment_with_prompt(self, image: Image.Image, 
                             prompt: str) -> Tuple[List, List, List]:
        """使用文本提示进行分割"""
        if USE_TRANSFORMERS:
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            return results["masks"], results["boxes"], results["scores"]
        else:
            inference_state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )
            return output["masks"], output["boxes"], output["scores"]
    
    def _combine_masks(self, masks: List, boxes: List, scores: List,
                       image_size: Tuple[int, int]) -> np.ndarray:
        """合并多个mask"""
        width, height = image_size
        combined_mask = np.zeros((height, width), dtype=bool)
        
        for mask, score in zip(masks, scores):
            if isinstance(score, torch.Tensor):
                score = score.item()
            
            if score < self.threshold:
                continue
            
            # 转换mask为numpy
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()
            
            # 二值化
            if mask_np.max() <= 1.0:
                mask_bool = mask_np > self.threshold
            else:
                mask_bool = mask_np > 128
            
            # 调整大小
            if mask_bool.shape != (height, width):
                mask_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))
                mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                mask_bool = np.array(mask_pil) > 128
            
            combined_mask = combined_mask | mask_bool
        
        return combined_mask
    
    def save_mask(self, seg_mask: np.ndarray, output_path: str,
                  save_colored: bool = True) -> None:
        """
        保存分割mask
        
        Args:
            seg_mask: 分割mask
            output_path: 输出路径
            save_colored: 是否同时保存彩色可视化图
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存原始mask（类别ID作为像素值）
        mask_image = Image.fromarray(seg_mask, mode='L')
        mask_image.save(output_path)
        
        # 保存彩色可视化图
        if save_colored:
            colored_output = output_path.replace('.png', '_colored.png')
            colored_mask = self.palette.apply_to_mask(seg_mask)
            colored_image = Image.fromarray(colored_mask, mode='RGB')
            colored_image.save(colored_output)
    
    def process_camera_folder(self, cam_dir: str, cam_serial: str,
                               calib_dir: str = None) -> Dict:
        """
        处理单个相机文件夹
        
        Args:
            cam_dir: 相机文件夹路径
            cam_serial: 相机序列号
            calib_dir: 标定文件夹路径
        
        Returns:
            处理结果统计
        """
        # 创建segmentation文件夹
        seg_dir = os.path.join(cam_dir, "segmentation")
        os.makedirs(seg_dir, exist_ok=True)
        
        # 获取RGB图像列表
        color_dir = os.path.join(cam_dir, "color")
        if not os.path.exists(color_dir):
            print(f"Warning: {color_dir} 不存在")
            return {'success': False, 'error': 'color目录不存在'}
        
        rgb_files = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
        if len(rgb_files) == 0:
            rgb_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
        
        if len(rgb_files) == 0:
            print(f"Warning: {color_dir} 中没有找到RGB图像")
            return {'success': False, 'error': '无RGB图像'}
        
        # 加载时间戳
        timestamps_dict = self._load_timestamps(cam_dir)
        
        # 处理结果统计
        stats = {
            'success': True,
            'total_images': len(rgb_files),
            'processed_images': 0,
            'skipped_images': 0,
            'failed_images': 0
        }
        
        # 处理每张RGB图像
        print(f"  处理 {len(rgb_files)} 张RGB图像...")
        
        for idx, rgb_path in enumerate(tqdm(rgb_files, desc=f"  分割 {cam_serial}", leave=False)):
            rgb_filename = os.path.basename(rgb_path)
            rgb_name_without_ext = os.path.splitext(rgb_filename)[0]
            
            # 输出文件路径
            seg_filename = rgb_name_without_ext + ".png"
            seg_path = os.path.join(seg_dir, seg_filename)
            
            # 检查是否需要跳过
            if not self.overwrite and os.path.exists(seg_path):
                stats['skipped_images'] += 1
                continue
            
            try:
                # 执行分割
                seg_mask, annotations = self.segment_image(rgb_path)
                
                # 保存mask
                self.save_mask(seg_mask, seg_path, save_colored=True)
                
                stats['processed_images'] += 1
                
            except Exception as e:
                print(f"Warning: 处理 {rgb_path} 失败: {e}")
                stats['failed_images'] += 1
                continue
        
        return stats
    
    def _load_timestamps(self, cam_dir: str) -> Optional[Dict]:
        """加载时间戳"""
        timestamps_path = os.path.join(cam_dir, "timestamps.npy")
        if not os.path.exists(timestamps_path):
            return None
        
        try:
            timestamps_data = np.load(timestamps_path, allow_pickle=True)
            if isinstance(timestamps_data, np.ndarray):
                if timestamps_data.ndim == 0:
                    timestamps_dict = timestamps_data.item()
                    if isinstance(timestamps_dict, dict):
                        return timestamps_dict
                else:
                    timestamps_list = timestamps_data.flatten().tolist()
                    return {"color": timestamps_list, "depth": timestamps_list}
        except Exception:
            pass
        
        return None
    
    def process_task(self, task_dir: str, calib_dir: str = None) -> Dict:
        """
        处理单个任务
        
        Args:
            task_dir: 任务目录路径
            calib_dir: 标定目录路径
        
        Returns:
            处理结果统计
        """
        task_name = os.path.basename(task_dir)
        print(f"\n处理任务: {task_name}")
        
        # 查找所有相机文件夹
        cam_folders = [
            d for d in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, d)) and d.startswith("cam_")
        ]
        
        if len(cam_folders) == 0:
            print(f"  Warning: 在 {task_dir} 中没有找到相机文件夹")
            return {'success': False, 'error': '无相机文件夹'}
        
        print(f"  找到 {len(cam_folders)} 个相机文件夹")
        
        # 处理结果
        task_stats = {
            'success': True,
            'task_name': task_name,
            'cameras': {}
        }
        
        # 处理每个相机
        for cam_folder in cam_folders:
            cam_path = os.path.join(task_dir, cam_folder)
            cam_serial = cam_folder.replace("cam_", "")
            
            print(f"  处理相机: {cam_serial}")
            
            cam_stats = self.process_camera_folder(cam_path, cam_serial, calib_dir)
            task_stats['cameras'][cam_serial] = cam_stats
        
        return task_stats
    
    def get_palette(self) -> ColorPalette:
        """获取调色板"""
        return self.palette


def segment_dataset(config: ConfigManager, palette: ColorPalette = None) -> Dict:
    """
    对整个数据集进行语义分割
    
    Args:
        config: 配置管理器
        palette: 调色板（可选）
    
    Returns:
        处理结果统计
    """
    segmenter = SemanticSegmenter(config, palette)
    
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
        calib_dir = config.get_calib_dir(task_name)
        
        task_stats = segmenter.process_task(str(task_dir), str(calib_dir) if calib_dir else None)
        results['tasks'][task_name] = task_stats
    
    # 返回调色板
    results['palette'] = segmenter.get_palette()
    
    return results


if __name__ == "__main__":
    # 测试语义分割模块
    from config_manager import load_config
    
    config = load_config()
    results = segment_dataset(config)
    
    print("\n分割完成！")
    for task_name, task_stats in results.get('tasks', {}).items():
        print(f"  任务 {task_name}:")
        for cam_serial, cam_stats in task_stats.get('cameras', {}).items():
            print(f"    相机 {cam_serial}: {cam_stats}")

