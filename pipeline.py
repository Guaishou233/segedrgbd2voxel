#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB2Voxel 完整处理流程
从语义分割到点云生成再到体素化的完整流程
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from config_manager import ConfigManager, load_config
from color_palette import ColorPalette, create_palette_from_targets


# 配置日志
def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class RGB2VoxelPipeline:
    """RGB到体素的完整处理流程"""
    
    def __init__(self, config_path: str = None):
        """
        初始化处理流程
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        
        # 设置日志
        log_level = self.config.get('pipeline.log_level', 'INFO')
        setup_logging(log_level)
        
        self.logger = logging.getLogger(__name__)
        
        # 创建调色板
        self.palette = self._create_palette()
        
        # 处理结果
        self.results = {
            'start_time': None,
            'end_time': None,
            'segmentation': None,
            'pointcloud': None,
            'voxelization': None
        }
    
    def _create_palette(self) -> ColorPalette:
        """创建调色板"""
        targets = self.config.get('segmentation.targets', [])
        background = self.config.get('segmentation.background', 
                                     {'label': 'background', 'id': 2})
        color_config = self.config.get('color_palette', {})
        
        palette = create_palette_from_targets(targets, background, color_config)
        
        self.logger.info(f"创建调色板: {len(palette.color_map)} 个类别")
        for class_id, info in sorted(palette.color_map.items()):
            self.logger.debug(f"  {class_id}: {info['name']} - RGB{info['color']}")
        
        return palette
    
    def save_color_map(self) -> None:
        """保存颜色映射到数据集目录"""
        if not self.config.get('pipeline.save_color_map', True):
            return
        
        dataset_dir = self.config.dataset_dir
        color_map_path = dataset_dir / "color_map.json"
        
        self.palette.save(str(color_map_path))
        self.logger.info(f"颜色映射已保存到: {color_map_path}")
    
    def run_segmentation(self) -> Dict:
        """运行语义分割"""
        self.logger.info("=" * 60)
        self.logger.info("步骤 1: 语义分割")
        self.logger.info("=" * 60)
        
        from segmentation_module import segment_dataset
        
        results = segment_dataset(self.config, self.palette)
        self.results['segmentation'] = results
        
        # 统计信息
        total_processed = 0
        total_skipped = 0
        total_failed = 0
        
        for task_name, task_stats in results.get('tasks', {}).items():
            for cam_serial, cam_stats in task_stats.get('cameras', {}).items():
                total_processed += cam_stats.get('processed_images', 0)
                total_skipped += cam_stats.get('skipped_images', 0)
                total_failed += cam_stats.get('failed_images', 0)
        
        self.logger.info(f"语义分割完成: 处理 {total_processed}, 跳过 {total_skipped}, 失败 {total_failed}")
        
        return results
    
    def run_pointcloud(self) -> Dict:
        """运行点云生成"""
        self.logger.info("=" * 60)
        self.logger.info("步骤 2: 点云生成")
        self.logger.info("=" * 60)
        
        from pointcloud_module import generate_pointclouds
        
        results = generate_pointclouds(self.config, self.palette)
        self.results['pointcloud'] = results
        
        # 统计信息
        total_processed = 0
        total_skipped = 0
        total_failed = 0
        
        for task_name, task_stats in results.get('tasks', {}).items():
            total_processed += task_stats.get('processed_frames', 0)
            total_skipped += task_stats.get('skipped_frames', 0)
            total_failed += task_stats.get('failed_frames', 0)
        
        self.logger.info(f"点云生成完成: 处理 {total_processed}, 跳过 {total_skipped}, 失败 {total_failed}")
        
        return results
    
    def run_voxelization(self) -> Dict:
        """运行体素化"""
        self.logger.info("=" * 60)
        self.logger.info("步骤 3: 体素化")
        self.logger.info("=" * 60)
        
        from voxelization_module import voxelize_dataset
        
        results = voxelize_dataset(self.config, self.palette)
        self.results['voxelization'] = results
        
        # 统计信息
        total_processed = 0
        total_skipped = 0
        total_failed = 0
        
        for task_name, task_stats in results.get('tasks', {}).items():
            total_processed += task_stats.get('processed_files', 0)
            total_skipped += task_stats.get('skipped_files', 0)
            total_failed += task_stats.get('failed_files', 0)
        
        self.logger.info(f"体素化完成: 处理 {total_processed}, 跳过 {total_skipped}, 失败 {total_failed}")
        
        return results
    
    def run(self, steps: List[str] = None) -> Dict:
        """
        运行完整处理流程
        
        Args:
            steps: 要执行的步骤列表，可选 ['segmentation', 'pointcloud', 'voxelization']
                   如果为None或空列表，则执行所有步骤
        
        Returns:
            处理结果字典
        """
        self.results['start_time'] = datetime.now().isoformat()
        
        # 获取要执行的步骤
        if steps is None or len(steps) == 0:
            steps = self.config.get('pipeline.steps', [])
        
        if not steps:
            steps = ['segmentation', 'pointcloud', 'voxelization']
        
        self.logger.info("=" * 60)
        self.logger.info("RGB2Voxel 完整处理流程")
        self.logger.info("=" * 60)
        self.logger.info(f"配置文件: {self.config.config_path}")
        self.logger.info(f"数据集目录: {self.config.dataset_dir}")
        self.logger.info(f"要处理的任务: {self.config.tasks}")
        self.logger.info(f"执行步骤: {steps}")
        self.logger.info("=" * 60)
        
        # 保存颜色映射
        self.save_color_map()
        
        # 执行各步骤
        if 'segmentation' in steps:
            self.run_segmentation()
        
        if 'pointcloud' in steps:
            self.run_pointcloud()
        
        if 'voxelization' in steps:
            self.run_voxelization()
        
        self.results['end_time'] = datetime.now().isoformat()
        
        # 保存处理结果
        self._save_results()
        
        self.logger.info("=" * 60)
        self.logger.info("处理流程完成!")
        self.logger.info("=" * 60)
        
        return self.results
    
    def _save_results(self) -> None:
        """保存处理结果到文件"""
        results_path = self.config.dataset_dir / "pipeline_results.json"
        
        # 转换结果为可序列化格式
        serializable_results = self._make_serializable(self.results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"处理结果已保存到: {results_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == 'palette':
                    continue  # 跳过调色板对象
                result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RGB2Voxel 完整处理流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置运行完整流程
  python pipeline.py
  
  # 使用指定配置文件
  python pipeline.py --config /path/to/config.yaml
  
  # 只运行分割步骤
  python pipeline.py --steps segmentation
  
  # 运行分割和点云生成
  python pipeline.py --steps segmentation pointcloud
  
  # 只运行体素化（假设点云已生成）
  python pipeline.py --steps voxelization
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径（默认使用项目目录下的config.yaml）"
    )
    
    parser.add_argument(
        "--steps", "-s",
        type=str,
        nargs='+',
        choices=['segmentation', 'pointcloud', 'voxelization'],
        default=None,
        help="要执行的步骤（默认执行所有步骤）"
    )
    
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        nargs='+',
        default=None,
        help="要处理的任务名称（覆盖配置文件中的设置）"
    )
    
    parser.add_argument(
        "--overwrite", "-o",
        action="store_true",
        help="覆盖已有的处理结果"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 创建处理流程
    pipeline = RGB2VoxelPipeline(args.config)
    
    # 覆盖配置
    if args.tasks:
        pipeline.config.set('dataset.tasks', args.tasks)
    
    if args.overwrite:
        pipeline.config.set('segmentation.overwrite', True)
        pipeline.config.set('pointcloud.overwrite', True)
        pipeline.config.set('voxelization.overwrite', True)
    
    if args.log_level:
        pipeline.config.set('pipeline.log_level', args.log_level)
        setup_logging(args.log_level)
    
    # 运行流程
    results = pipeline.run(args.steps)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("处理摘要")
    print("=" * 60)
    print(f"开始时间: {results['start_time']}")
    print(f"结束时间: {results['end_time']}")
    
    if results.get('segmentation'):
        seg_results = results['segmentation']
        total = sum(
            sum(cam.get('processed_images', 0) + cam.get('skipped_images', 0) + cam.get('failed_images', 0)
                for cam in task.get('cameras', {}).values())
            for task in seg_results.get('tasks', {}).values()
        )
        print(f"语义分割: {total} 张图像")
    
    if results.get('pointcloud'):
        pc_results = results['pointcloud']
        total = sum(
            task.get('processed_frames', 0) + task.get('skipped_frames', 0) + task.get('failed_frames', 0)
            for task in pc_results.get('tasks', {}).values()
        )
        print(f"点云生成: {total} 帧")
    
    if results.get('voxelization'):
        vox_results = results['voxelization']
        total = sum(
            task.get('processed_files', 0) + task.get('skipped_files', 0) + task.get('failed_files', 0)
            for task in vox_results.get('tasks', {}).values()
        )
        print(f"体素化: {total} 个点云")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

