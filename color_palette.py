#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调色板管理模块
负责管理语义分割的颜色映射，支持保存和加载颜色配置
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class ColorPalette:
    """调色板管理器"""
    
    # 预定义的颜色集合（用于自动扩展）
    DEFAULT_COLORS = [
        [255, 0, 0],      # 红色
        [0, 255, 0],      # 绿色
        [0, 0, 255],      # 蓝色
        [255, 255, 0],    # 黄色
        [255, 0, 255],    # 品红
        [0, 255, 255],    # 青色
        [255, 128, 0],    # 橙色
        [128, 0, 255],    # 紫色
        [255, 192, 203],  # 粉色
        [0, 128, 0],      # 深绿
        [128, 0, 0],      # 深红
        [0, 0, 128],      # 深蓝
        [128, 128, 0],    # 橄榄色
        [0, 128, 128],    # 青绿色
        [128, 0, 128],    # 紫红色
        [192, 192, 192],  # 银色
        [128, 128, 128],  # 灰色
        [64, 64, 64],     # 深灰色
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化调色板
        
        Args:
            config: 配置字典，包含颜色定义
        """
        self.color_map: Dict[int, Dict[str, Any]] = {}
        self.auto_extend_colors = self.DEFAULT_COLORS.copy()
        
        if config is not None:
            self._load_from_config(config)
    
    def _load_from_config(self, config: Dict[str, Any]) -> None:
        """从配置加载颜色映射"""
        colors = config.get('colors', {})
        for class_id, color_info in colors.items():
            class_id = int(class_id)
            if isinstance(color_info, dict):
                self.color_map[class_id] = {
                    'name': color_info.get('name', f'class_{class_id}'),
                    'color': tuple(color_info.get('color', self._get_auto_color(class_id)))
                }
            elif isinstance(color_info, (list, tuple)):
                self.color_map[class_id] = {
                    'name': f'class_{class_id}',
                    'color': tuple(color_info)
                }
        
        # 加载自动扩展颜色
        auto_colors = config.get('auto_extend_colors', [])
        if auto_colors:
            self.auto_extend_colors = [list(c) for c in auto_colors]
    
    def _get_auto_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取自动分配的颜色"""
        idx = (class_id - 1) % len(self.auto_extend_colors)
        return tuple(self.auto_extend_colors[idx])
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        获取类别对应的颜色
        
        Args:
            class_id: 类别ID
        
        Returns:
            RGB颜色元组 (R, G, B)
        """
        if class_id in self.color_map:
            return self.color_map[class_id]['color']
        
        # 自动分配颜色
        color = self._get_auto_color(class_id)
        self.color_map[class_id] = {
            'name': f'class_{class_id}',
            'color': color
        }
        return color
    
    def get_name(self, class_id: int) -> str:
        """获取类别名称"""
        if class_id in self.color_map:
            return self.color_map[class_id]['name']
        return f'class_{class_id}'
    
    def set_color(self, class_id: int, color: Tuple[int, int, int], 
                  name: str = None) -> None:
        """
        设置类别的颜色
        
        Args:
            class_id: 类别ID
            color: RGB颜色元组
            name: 类别名称（可选）
        """
        if name is None:
            name = self.get_name(class_id)
        
        self.color_map[class_id] = {
            'name': name,
            'color': tuple(color)
        }
    
    def add_class(self, class_id: int, name: str, 
                  color: Tuple[int, int, int] = None) -> None:
        """
        添加新的类别
        
        Args:
            class_id: 类别ID
            name: 类别名称
            color: RGB颜色元组（可选，如果不提供则自动分配）
        """
        if color is None:
            color = self._get_auto_color(class_id)
        
        self.color_map[class_id] = {
            'name': name,
            'color': tuple(color)
        }
    
    def get_all_classes(self) -> Dict[int, Dict[str, Any]]:
        """获取所有类别信息"""
        return self.color_map.copy()
    
    def to_numpy_colormap(self, max_classes: int = 256) -> np.ndarray:
        """
        转换为numpy格式的颜色映射表
        
        Args:
            max_classes: 最大类别数
        
        Returns:
            np.ndarray: 形状为 (max_classes, 3) 的颜色映射表
        """
        colormap = np.zeros((max_classes, 3), dtype=np.uint8)
        
        for class_id in range(max_classes):
            colormap[class_id] = self.get_color(class_id)
        
        return colormap
    
    def save(self, output_path: str) -> None:
        """
        保存调色板到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        data = {
            'color_map': {},
            'class_names': {},
            'description': '语义分割颜色映射配置'
        }
        
        for class_id, info in self.color_map.items():
            data['color_map'][str(class_id)] = {
                'name': info['name'],
                'color': list(info['color']),
                'hex': '#{:02x}{:02x}{:02x}'.format(*info['color'])
            }
            data['class_names'][str(class_id)] = info['name']
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'ColorPalette':
        """
        从JSON文件加载调色板
        
        Args:
            path: JSON文件路径
        
        Returns:
            ColorPalette实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        palette = cls()
        color_map = data.get('color_map', {})
        
        for class_id, info in color_map.items():
            class_id = int(class_id)
            palette.color_map[class_id] = {
                'name': info.get('name', f'class_{class_id}'),
                'color': tuple(info.get('color', [128, 128, 128]))
            }
        
        return palette
    
    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        将调色板应用到语义分割mask，生成彩色可视化图像
        
        Args:
            mask: 语义分割mask，形状为 (H, W)，值为类别ID
        
        Returns:
            np.ndarray: 彩色图像，形状为 (H, W, 3)
        """
        height, width = mask.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            color = self.get_color(int(class_id))
            color_image[mask == class_id] = color
        
        return color_image
    
    def __repr__(self) -> str:
        return f"ColorPalette({len(self.color_map)} classes)"
    
    def __str__(self) -> str:
        lines = ["ColorPalette:"]
        for class_id, info in sorted(self.color_map.items()):
            color_str = f"RGB({info['color'][0]}, {info['color'][1]}, {info['color'][2]})"
            lines.append(f"  {class_id}: {info['name']} - {color_str}")
        return "\n".join(lines)


def create_palette_from_targets(targets: List[Dict[str, Any]], 
                                 background: Dict[str, Any],
                                 color_config: Dict[str, Any] = None) -> ColorPalette:
    """
    从分割目标配置创建调色板
    
    Args:
        targets: 分割目标列表，每个元素包含 label, prompt, id
        background: 背景配置，包含 label, id
        color_config: 颜色配置
    
    Returns:
        ColorPalette实例
    """
    palette = ColorPalette(color_config)
    
    # 添加目标类别
    for target in targets:
        class_id = target['id']
        name = target['label']
        
        if class_id not in palette.color_map:
            palette.add_class(class_id, name)
        else:
            # 更新名称
            palette.color_map[class_id]['name'] = name
    
    # 添加背景类别
    bg_id = background['id']
    bg_name = background['label']
    if bg_id not in palette.color_map:
        palette.add_class(bg_id, bg_name, (128, 128, 128))
    else:
        palette.color_map[bg_id]['name'] = bg_name
    
    return palette


if __name__ == "__main__":
    # 测试调色板
    palette = ColorPalette()
    
    # 添加一些类别
    palette.add_class(1, "robot_arm", (255, 0, 0))
    palette.add_class(2, "background", (128, 128, 128))
    palette.add_class(3, "table", (0, 255, 0))
    
    print(palette)
    
    # 保存调色板
    test_path = "/tmp/test_color_map.json"
    palette.save(test_path)
    print(f"\n调色板已保存到: {test_path}")
    
    # 加载调色板
    loaded_palette = ColorPalette.load(test_path)
    print(f"\n加载的调色板: {loaded_palette}")

