#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
负责加载和管理配置文件
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置文件
        """
        self.project_root = Path(__file__).parent
        
        if config_path is None:
            config_path = self.project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 处理相对路径
        config = self._process_paths(config)
        
        return config
    
    def _process_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置中的相对路径"""
        # 处理数据集路径
        if 'dataset' in config:
            base_dir = config['dataset'].get('base_dir', '')
            if not os.path.isabs(base_dir):
                config['dataset']['base_dir'] = str(self.project_root / base_dir)
        
        # 处理模型路径
        if 'segmentation' in config:
            model_path = config['segmentation'].get('model_path', '')
            if not os.path.isabs(model_path):
                config['segmentation']['model_path'] = str(self.project_root / model_path)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的路径，如 'dataset.base_dir'
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def dataset_dir(self) -> Path:
        """获取数据集目录"""
        base_dir = self.get('dataset.base_dir')
        dataset_name = self.get('dataset.dataset_name')
        return Path(base_dir) / dataset_name
    
    @property
    def tasks(self) -> List[str]:
        """获取要处理的任务列表"""
        tasks = self.get('dataset.tasks', [])
        if not tasks:
            # 如果未指定任务，自动发现所有任务
            tasks = self._discover_tasks()
        return tasks
    
    def _discover_tasks(self) -> List[str]:
        """自动发现数据集中的所有任务"""
        dataset_dir = self.dataset_dir
        if not dataset_dir.exists():
            return []
        
        tasks = []
        for item in dataset_dir.iterdir():
            if item.is_dir() and item.name.startswith('task_'):
                tasks.append(item.name)
        
        return sorted(tasks)
    
    def get_task_dir(self, task_name: str) -> Path:
        """获取任务目录"""
        return self.dataset_dir / task_name
    
    def get_calib_dir(self, task_name: str) -> Optional[Path]:
        """获取任务对应的标定目录"""
        task_dir = self.get_task_dir(task_name)
        metadata_path = task_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            calib_timestamp = metadata.get('calib')
            if calib_timestamp:
                calib_dir = self.dataset_dir / "calib" / str(calib_timestamp)
                if calib_dir.exists():
                    return calib_dir
        except Exception:
            pass
        
        # 尝试使用最新的标定目录
        calib_base = self.dataset_dir / "calib"
        if calib_base.exists():
            calib_dirs = sorted([d for d in calib_base.iterdir() if d.is_dir()])
            if calib_dirs:
                return calib_dirs[-1]
        
        return None
    
    def save_config(self, path: str = None) -> None:
        """
        保存配置到文件
        
        Args:
            path: 保存路径，如果为None则使用原路径
        """
        if path is None:
            path = self.config_path
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"


def load_config(config_path: str = None) -> ConfigManager:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        ConfigManager实例
    """
    return ConfigManager(config_path)


if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    print(f"数据集目录: {config.dataset_dir}")
    print(f"任务列表: {config.tasks}")
    print(f"分割阈值: {config.get('segmentation.threshold')}")
    print(f"点云深度范围: {config.get('pointcloud.min_depth')} - {config.get('pointcloud.max_depth')}")

