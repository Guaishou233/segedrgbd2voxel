#!/usr/bin/env python3
"""
数据格式转换器

将 dataset2 格式的数据转换为 rgb2voxel/data/dataset 兼容的格式

源数据结构 (dataset2):
- task_X/
  - basic_XXX/
    - Replicator/, Replicator_01/, ... (多个相机)
      - rgb/ (PNG 图像)
      - distance_to_camera/ (NPY 深度数据)
      - instance_segmentation/ (分割图像和映射)
  - camera_info_default_session.json
  - garment_info_default_session.json

目标数据结构 (dataset):
- task_X/
  - cam_XXX/ (多个相机)
    - color/ (JPG 图像)
    - depth/ (PNG 深度图像)
    - segmentation/ (PNG 分割图像)
    - timestamps.npy
  - metadata.json
  - color_map.json
  - POINTCLOUDS/ (空目录)
  - VOXELS/ (空目录)
"""

import os
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import cv2


class Dataset2Converter:
    """将 dataset2 格式转换为 rgb2voxel 兼容格式的转换器"""
    
    def __init__(self, source_dir: str, target_dir: str, task_name: Optional[str] = None):
        """
        初始化转换器
        
        Args:
            source_dir: 源数据目录 (dataset2/task_X)
            target_dir: 目标数据目录 (rgb2voxel/data/dataset2)
            task_name: 任务名称（如果不指定，会自动从源目录提取）
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.task_name = task_name
        
        # 验证源目录
        if not self.source_dir.exists():
            raise ValueError(f"源目录不存在: {self.source_dir}")
        
        # 加载相机信息
        self.camera_info = self._load_camera_info()
        self.garment_info = self._load_garment_info()
        
        # 查找所有 basic_XXX 目录
        self.basic_dirs = self._find_basic_dirs()
        
    def _load_camera_info(self) -> Dict:
        """加载相机信息"""
        camera_info_path = self.source_dir / "camera_info_default_session.json"
        if camera_info_path.exists():
            with open(camera_info_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_garment_info(self) -> Dict:
        """加载物体/服装信息"""
        garment_info_path = self.source_dir / "garment_info_default_session.json"
        if garment_info_path.exists():
            with open(garment_info_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _find_basic_dirs(self) -> List[Path]:
        """查找所有 basic_XXX 目录"""
        basic_dirs = []
        for item in self.source_dir.iterdir():
            if item.is_dir() and item.name.startswith("basic_"):
                basic_dirs.append(item)
        return sorted(basic_dirs)
    
    def _find_replicator_dirs(self, basic_dir: Path) -> List[Tuple[str, Path]]:
        """
        查找 basic 目录下的所有 Replicator 目录
        
        Returns:
            List of (camera_name, path) tuples
        """
        replicator_dirs = []
        for item in basic_dir.iterdir():
            if item.is_dir() and item.name.startswith("Replicator"):
                # 提取相机索引：Replicator -> 0, Replicator_01 -> 1, etc.
                if item.name == "Replicator":
                    cam_idx = 0
                else:
                    try:
                        cam_idx = int(item.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                cam_name = f"cam_{cam_idx:02d}"
                replicator_dirs.append((cam_name, item))
        return sorted(replicator_dirs, key=lambda x: x[0])
    
    def _convert_rgb_to_jpg(self, src_dir: Path, dst_dir: Path) -> List[int]:
        """
        将 RGB PNG 图像转换为 JPG
        
        文件名使用时间戳命名（与 dataset 格式一致）
        
        Returns:
            时间戳列表
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        timestamps = []
        
        if not src_dir.exists():
            return timestamps
        
        # 获取所有 PNG 文件
        png_files = sorted(src_dir.glob("*.png"))
        
        for idx, png_file in enumerate(png_files):
            # 提取时间戳（源文件名通常是时间戳）
            try:
                timestamp = int(png_file.stem)
            except ValueError:
                # 如果文件名不是时间戳，生成一个基于索引的伪时间戳
                timestamp = 1000000000000 + idx * 100
            timestamps.append(timestamp)
            
            # 转换为 JPG
            img = Image.open(png_file)
            # 如果有 alpha 通道，转换为 RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 使用时间戳作为文件名（与 dataset 格式一致）
            jpg_path = dst_dir / f"{timestamp}.jpg"
            img.save(jpg_path, "JPEG", quality=95)
        
        return timestamps
    
    def _convert_depth_to_png(self, src_dir: Path, dst_dir: Path, timestamps: List[int]) -> Dict:
        """
        将深度 NPY 数据转换为 PNG 图像
        
        深度数据通常以米为单位的浮点数，需要转换为 16位 PNG（毫米）
        
        Args:
            src_dir: 源目录
            dst_dir: 目标目录
            timestamps: 时间戳列表（与 RGB 对应）
        
        Returns:
            深度数据统计信息
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        depth_stats = {
            'min_depth': float('inf'),
            'max_depth': 0.0,
            'count': 0
        }
        
        if not src_dir.exists():
            return depth_stats
        
        # 获取所有 NPY 文件
        npy_files = sorted(src_dir.glob("*.npy"))
        
        for idx, npy_file in enumerate(npy_files):
            depth_data = np.load(npy_file)
            
            # 统计深度范围
            valid_depth = depth_data[depth_data > 0]
            if len(valid_depth) > 0:
                depth_stats['min_depth'] = min(depth_stats['min_depth'], float(valid_depth.min()))
                depth_stats['max_depth'] = max(depth_stats['max_depth'], float(valid_depth.max()))
                depth_stats['count'] += 1
            
            # 将深度数据转换为毫米单位的 16 位整数
            depth_mm = (depth_data * 1000).astype(np.uint16)
            
            # 使用时间戳作为文件名
            timestamp = timestamps[idx] if idx < len(timestamps) else (1000000000000 + idx * 100)
            png_path = dst_dir / f"{timestamp}.png"
            cv2.imwrite(str(png_path), depth_mm)
        
        return depth_stats
    
    def _convert_segmentation(self, src_dir: Path, dst_dir: Path, timestamps: List[int]) -> Dict:
        """
        转换分割数据
        
        文件名使用时间戳命名（与 dataset 格式一致）
        
        Args:
            src_dir: 源目录
            dst_dir: 目标目录
            timestamps: 时间戳列表（与 RGB 对应）
        
        Returns:
            分割映射信息
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        segmentation_mapping = {}
        
        if not src_dir.exists():
            return segmentation_mapping
        
        # 复制分割 PNG 图像，使用时间戳命名
        png_files = sorted(src_dir.glob("*.png"))
        for idx, png_file in enumerate(png_files):
            timestamp = timestamps[idx] if idx < len(timestamps) else (1000000000000 + idx * 100)
            dst_path = dst_dir / f"{timestamp}.png"
            shutil.copy2(png_file, dst_path)
        
        # 读取分割映射
        mapping_file = src_dir / "instance_segmentation_mapping_0000.json"
        semantics_file = src_dir / "instance_segmentation_semantics_mapping_0000.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                segmentation_mapping['instance_mapping'] = json.load(f)
        
        if semantics_file.exists():
            with open(semantics_file, 'r') as f:
                segmentation_mapping['semantics_mapping'] = json.load(f)
        
        return segmentation_mapping
    
    def _generate_metadata(self, task_dir: Path, camera_names: List[str]) -> Dict:
        """
        生成 metadata.json
        
        包含相机内参、外参等信息
        
        重要：内参直接使用原始值，不进行任何缩放
        同时添加 intrinsic_scale 字段标记是否需要在点云生成时缩放
        """
        metadata = {
            "finish_time": int(self.camera_info.get("timestamp", 0) * 1000),
            "intrinsic_scale": 1.0,  # 标记内参不需要额外缩放
            "cameras": {}
        }
        
        cameras_list = self.camera_info.get("cameras", [])
        
        for idx, cam_name in enumerate(camera_names):
            if idx < len(cameras_list):
                cam_info = cameras_list[idx]
                
                # 直接使用原始内参矩阵，不进行缩放
                intrinsic_3x3 = cam_info.get("intrinsic_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                intrinsic_3x4 = [
                    [intrinsic_3x3[0][0], intrinsic_3x3[0][1], intrinsic_3x3[0][2], 0.0],
                    [intrinsic_3x3[1][0], intrinsic_3x3[1][1], intrinsic_3x3[1][2], 0.0],
                    [intrinsic_3x3[2][0], intrinsic_3x3[2][1], intrinsic_3x3[2][2], 0.0]
                ]
                
                # 获取外参矩阵
                extrinsic = cam_info.get("extrinsic_matrix", [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                metadata["cameras"][cam_name] = {
                    "serial": str(cam_info.get("camera_id", idx)),
                    "intrinsic": intrinsic_3x4,
                    "extrinsic": extrinsic,
                    "is_dynamic": False,
                    "resolution": cam_info.get("resolution", [640, 480]),
                    "fx": cam_info.get("fx", intrinsic_3x3[0][0]),
                    "fy": cam_info.get("fy", intrinsic_3x3[1][1]),
                    "cx": cam_info.get("cx", intrinsic_3x3[0][2]),
                    "cy": cam_info.get("cy", intrinsic_3x3[1][2])
                }
            else:
                # 使用默认值
                metadata["cameras"][cam_name] = {
                    "serial": str(idx),
                    "intrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                    "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "is_dynamic": False
                }
        
        return metadata
    
    def _generate_color_map(self, segmentation_mappings: Dict) -> Dict:
        """
        从分割映射生成 color_map.json
        """
        color_map = {
            "color_map": {},
            "class_names": {},
            "description": "语义分割颜色映射配置"
        }
        
        # 合并所有相机的分割映射
        all_semantics = {}
        for cam_name, mapping in segmentation_mappings.items():
            semantics = mapping.get('semantics_mapping', {})
            for color_str, info in semantics.items():
                if color_str not in all_semantics:
                    all_semantics[color_str] = info
        
        # 生成颜色映射
        class_id = 1
        seen_classes = set()
        
        for color_str, info in all_semantics.items():
            class_name = info.get('class', 'unknown')
            
            # 跳过背景和未标记类别
            if class_name in ['BACKGROUND', 'UNLABELLED']:
                continue
            
            if class_name in seen_classes:
                continue
            seen_classes.add(class_name)
            
            # 解析颜色字符串 "(R, G, B, A)"
            try:
                color_tuple = eval(color_str)
                color_rgb = list(color_tuple[:3])
            except:
                color_rgb = [128, 128, 128]
            
            # 转换为 hex
            hex_color = "#{:02x}{:02x}{:02x}".format(*color_rgb)
            
            color_map["color_map"][str(class_id)] = {
                "name": class_name,
                "color": color_rgb,
                "hex": hex_color
            }
            color_map["class_names"][str(class_id)] = class_name
            class_id += 1
        
        # 添加背景类
        color_map["color_map"][str(class_id)] = {
            "name": "background",
            "color": [0, 0, 0],
            "hex": "#000000"
        }
        color_map["class_names"][str(class_id)] = "background"
        
        return color_map
    
    def _save_timestamps(self, timestamps: List[int], dst_path: Path):
        """
        保存时间戳为 NPY 文件
        
        格式与 dataset 保持一致：字典格式 {'color': [...], 'depth': [...]}
        """
        if timestamps:
            timestamps_dict = {
                'color': timestamps,
                'depth': timestamps
            }
            np.save(dst_path, timestamps_dict)
        else:
            np.save(dst_path, {'color': [], 'depth': []})
    
    def convert(self):
        """执行转换"""
        print(f"开始转换...")
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")
        
        # 创建目标目录
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个 basic 目录
        for basic_dir in self.basic_dirs:
            task_name = self.task_name or f"task_{basic_dir.name.split('_')[1]}"
            task_dir = self.target_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n处理: {basic_dir.name} -> {task_name}")
            
            # 查找所有 Replicator 目录
            replicator_dirs = self._find_replicator_dirs(basic_dir)
            camera_names = [cam_name for cam_name, _ in replicator_dirs]
            
            segmentation_mappings = {}
            all_depth_stats = []
            
            # 转换每个相机的数据
            for cam_name, replicator_dir in replicator_dirs:
                print(f"  转换相机: {cam_name}")
                cam_dir = task_dir / cam_name
                
                # 转换 RGB 图像（返回时间戳列表）
                rgb_src = replicator_dir / "rgb"
                color_dst = cam_dir / "color"
                timestamps = self._convert_rgb_to_jpg(rgb_src, color_dst)
                print(f"    - 转换了 {len(timestamps)} 张 RGB 图像")
                
                # 转换深度数据（使用相同的时间戳命名）
                depth_src = replicator_dir / "distance_to_camera"
                depth_dst = cam_dir / "depth"
                depth_stats = self._convert_depth_to_png(depth_src, depth_dst, timestamps)
                all_depth_stats.append(depth_stats)
                depth_count = len(list(depth_dst.glob("*.png"))) if depth_dst.exists() else 0
                print(f"    - 转换了 {depth_count} 张深度图像")
                
                # 转换分割数据（使用相同的时间戳命名）
                seg_src = replicator_dir / "instance_segmentation"
                seg_dst = cam_dir / "segmentation"
                seg_mapping = self._convert_segmentation(seg_src, seg_dst, timestamps)
                segmentation_mappings[cam_name] = seg_mapping
                seg_count = len(list(seg_dst.glob("*.png"))) if seg_dst.exists() else 0
                print(f"    - 转换了 {seg_count} 张分割图像")
                
                # 保存时间戳（字典格式）
                self._save_timestamps(timestamps, cam_dir / "timestamps.npy")
            
            # 生成 metadata.json
            metadata = self._generate_metadata(task_dir, camera_names)
            with open(task_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  生成了 metadata.json")
            
            # 生成 color_map.json
            color_map = self._generate_color_map(segmentation_mappings)
            with open(task_dir / "color_map.json", 'w') as f:
                json.dump(color_map, f, indent=2)
            print(f"  生成了 color_map.json")
            
            # 创建空的点云和体素目录
            (task_dir / "POINTCLOUDS").mkdir(exist_ok=True)
            (task_dir / "VOXELS").mkdir(exist_ok=True)
            print(f"  创建了 POINTCLOUDS 和 VOXELS 空目录")
            
            # 初始化 pipeline_results.json
            pipeline_results = {
                "start_time": None,
                "end_time": None,
                "segmentation": {"success": False, "tasks": {}},
                "pointcloud": {"success": False, "tasks": {}},
                "voxelization": {"success": False, "tasks": {}}
            }
            with open(task_dir / "pipeline_results.json", 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            print(f"  生成了 pipeline_results.json")
            
            # 分析深度数据范围
            global_min_depth = min((s['min_depth'] for s in all_depth_stats if s['count'] > 0), default=0)
            global_max_depth = max((s['max_depth'] for s in all_depth_stats if s['count'] > 0), default=10)
            
            if global_min_depth != float('inf') and global_max_depth > 0:
                print(f"\n  深度数据分析:")
                print(f"    - 最小深度: {global_min_depth:.3f} 米")
                print(f"    - 最大深度: {global_max_depth:.3f} 米")
                print(f"\n  【重要】请更新 config.yaml 中的 pointcloud 配置:")
                print(f"    pointcloud.min_depth: {max(0, global_min_depth - 0.1):.1f}")
                print(f"    pointcloud.max_depth: {global_max_depth + 0.5:.1f}")
        
        # 在根目录也创建一个 color_map.json
        if segmentation_mappings:
            root_color_map = self._generate_color_map(segmentation_mappings)
            with open(self.target_dir / "color_map.json", 'w') as f:
                json.dump(root_color_map, f, indent=2)
        
        print(f"\n转换完成!")
        print(f"输出目录: {self.target_dir}")


def main():
    parser = argparse.ArgumentParser(description="将 dataset2 格式转换为 rgb2voxel 兼容格式")
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="源数据目录路径 (dataset2/task_X)"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="目标数据目录路径"
    )
    parser.add_argument(
        "--task-name", "-n",
        type=str,
        default=None,
        help="任务名称（可选，默认从源目录提取）"
    )
    
    args = parser.parse_args()
    
    converter = Dataset2Converter(
        source_dir=args.source,
        target_dir=args.target,
        task_name=args.task_name
    )
    converter.convert()


if __name__ == "__main__":
    main()
