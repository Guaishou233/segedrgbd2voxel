#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成带语义分割信息的点云数据
从 segmented_dataset 生成 pointed_dataset
"""

import os
import json
import logging
import numpy as np
import cv2
try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Open3D is required but not installed. "
        "Please install it using: conda install -c open3d-admin open3d "
        "or pip install open3d (if available for your Python version)"
    )
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SegmentedPointCloudGenerator:
    """生成带语义分割信息的点云"""
    
    def __init__(
        self,
        min_depth_m: float = 0.3,
        max_depth_m: float = 0.8,
        width: int = 640,
        height: int = 360,
        depth_scale: float = 1000.0,  # Depth图像缩放因子 (毫米转米)
        downsample_voxel_size: float = 0.001,  # 体素下采样大小(米)
        filter_num_neighbors: int = 10,
        filter_std_ratio: float = 2.0,
        use_semantic_color: bool = True,  # 是否使用语义标签作为颜色
    ):
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.width = width
        self.height = height
        self.depth_scale = depth_scale
        self.downsample_voxel_size = downsample_voxel_size
        self.filter_num_neighbors = filter_num_neighbors
        self.filter_std_ratio = filter_std_ratio
        self.use_semantic_color = use_semantic_color
        
        # 语义标签到颜色的映射 (RGB, 0-1范围)
        self.semantic_colors = {
            0: [0.5, 0.5, 0.5],      # others/背景 - 灰色
            255: [1.0, 0.0, 0.0],     # robot_arm - 红色
        }
    
    def load_images(
        self, 
        rgb_path: str, 
        depth_path: str, 
        seg_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载RGB、Depth和语义分割图像"""
        # 加载RGB图像
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        
        # 加载Depth图像 (RGB格式存储的灰度图，取第一个通道)
        depth_rgb = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if len(depth_rgb.shape) == 3:
            depth = depth_rgb[:, :, 0].astype(np.float32)  # 取R通道
        else:
            depth = depth_rgb.astype(np.float32)
        
        # 加载语义分割图像
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if len(seg.shape) == 3:
            seg = seg[:, :, 0]  # 如果是3通道，取第一个通道
        
        return rgb, depth, seg
    
    def depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        """将深度值转换为米
        
        深度图像是归一化的0-255值，需要映射到实际深度范围
        默认假设深度值归一化到0-max_depth_m的范围
        """
        # 如果深度值范围是0-255，说明是归一化的，需要映射到实际深度范围
        if depth.max() <= 255.0 and depth.max() > 1.0:
            # 归一化深度值，映射到0-max_depth_m范围
            depth_m = depth / 255.0 * self.max_depth_m
        else:
            # 假设已经是米单位，或者需要除以缩放因子
            depth_m = depth / self.depth_scale
        
        # 过滤无效深度
        depth_m[depth_m < self.min_depth_m] = 0
        depth_m[depth_m > self.max_depth_m] = 0
        return depth_m
    
    def create_pointcloud_with_semantics(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """从RGBD图像和语义分割图创建带语义的点云（使用改进版本）"""
        return self.create_pointcloud_with_semantics_improved(
            rgb, depth, seg, intrinsic, extrinsic
        )
    
    def map_points_to_segmentation(
        self,
        points: np.ndarray,
        seg: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """将3D点映射回2D像素坐标以获取语义标签"""
        labels = np.zeros(len(points), dtype=np.uint8)
        
        # 从3D点反投影到2D像素坐标
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        # 计算像素坐标
        valid_mask = points[:, 2] > 0  # 有效的深度点
        
        if np.any(valid_mask):
            # 如果提供了外参，需要将点转换到相机坐标系
            if extrinsic is not None:
                points_homo = np.hstack([points, np.ones((len(points), 1))])
                points_cam = (np.linalg.inv(extrinsic) @ points_homo.T).T[:, :3]
            else:
                points_cam = points
            
            u = (points_cam[valid_mask, 0] * fx / points_cam[valid_mask, 2] + cx).astype(int)
            v = (points_cam[valid_mask, 1] * fy / points_cam[valid_mask, 2] + cy).astype(int)
            
            # 确保坐标在图像范围内
            u = np.clip(u, 0, seg.shape[1] - 1)
            v = np.clip(v, 0, seg.shape[0] - 1)
            
            # 获取对应的语义标签
            labels[valid_mask] = seg[v, u]
            
        return labels
    
    def process_single_frame(
        self,
        rgb_path: str,
        depth_path: str,
        seg_path: str,
        params_path: str,
        output_path: str
    ) -> Optional[o3d.geometry.PointCloud]:
        """处理单帧数据生成点云"""
        try:
            # 加载图像
            rgb, depth, seg = self.load_images(rgb_path, depth_path, seg_path)
            
            # 加载相机参数
            with open(params_path, 'r') as f:
                params = json.load(f)
            
            # 解析内参和外参
            intrinsic = np.array(params['intrinsic'])[:3, :3]  # 3x3矩阵
            extrinsic = np.array(params['extrinsic'])  # 4x4矩阵
            
            # 创建点云（带语义信息）
            pcd = self.create_pointcloud_with_semantics(
                rgb, depth, seg, intrinsic, extrinsic
            )
            
            # 下采样
            if self.downsample_voxel_size > 0:
                pcd = pcd.voxel_down_sample(self.downsample_voxel_size)
            
            # 统计离群点过滤
            if len(pcd.points) > 0:
                _, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=self.filter_num_neighbors,
                    std_ratio=self.filter_std_ratio
                )
                pcd = pcd.select_by_index(ind)
            
            # 保存点云
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
            
            return pcd
            
        except Exception as e:
            logger.error(f"处理帧失败 {rgb_path}: {str(e)}")
            return None
    
    def create_pointcloud_with_semantics_improved(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """改进版本：直接从2D像素映射语义标签到3D点"""
        
        # 转换深度到米
        depth_m = self.depth_to_meters(depth)
        
        # 创建RGBD图像（使用语义颜色替代RGB颜色）
        if self.use_semantic_color:
            # 创建语义颜色图像
            semantic_rgb = np.zeros_like(rgb)
            for label, color in self.semantic_colors.items():
                mask = seg == label
                semantic_rgb[mask] = [int(c * 255) for c in color]
            
            rgb_o3d = o3d.geometry.Image(semantic_rgb.astype(np.uint8))
        else:
            rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        
        depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            convert_rgb_to_intensity=False
        )
        
        # 设置相机内参
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(
            self.width, self.height,
            intrinsic[0, 0], intrinsic[1, 1],
            intrinsic[0, 2], intrinsic[1, 2]
        )
        
        # 创建点云
        if extrinsic is not None:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic_o3d, extrinsic=extrinsic
            )
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic_o3d
            )
        
        return pcd


def generate_segmented_pointcloud_dataset(
    input_dir: str = "/data/tangqiansong/rgb2voxel/data/segmented_dataset",
    output_dir: str = "/data/tangqiansong/rgb2voxel/data/pointed_dataset",
    min_depth_m: float = 0.3,
    max_depth_m: float = 0.8,
    depth_scale: float = 1000.0,
    downsample_voxel_size: float = 0.001,
    use_semantic_color: bool = True,
    max_samples: Optional[int] = None
):
    """批量生成带语义分割的点云数据集
    
    Args:
        input_dir: 输入数据集目录
        output_dir: 输出点云数据集目录
        min_depth_m: 最小深度值（米）
        max_depth_m: 最大深度值（米）
        depth_scale: 深度缩放因子（毫米转米）
        downsample_voxel_size: 体素下采样大小（米）
        use_semantic_color: 是否使用语义颜色
        max_samples: 最大处理样本数，None表示处理所有样本（用于测试）
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_pcd_dir = os.path.join(output_dir, "POINTCLOUD")
    os.makedirs(output_pcd_dir, exist_ok=True)
    
    # 初始化生成器
    generator = SegmentedPointCloudGenerator(
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        depth_scale=depth_scale,
        downsample_voxel_size=downsample_voxel_size,
        use_semantic_color=use_semantic_color
    )
    
    # 获取所有RGB图像文件
    rgb_dir = os.path.join(input_dir, "RGB")
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    total_files = len(rgb_files)
    logger.info(f"找到 {total_files} 个RGB图像文件")
    
    # 限制样本数量（用于测试）
    if max_samples is not None and max_samples > 0:
        rgb_files = rgb_files[:max_samples]
        logger.info(f"测试模式: 仅处理前 {len(rgb_files)} 个样本")
    
    # 处理每个文件
    success_count = 0
    for rgb_file in tqdm(rgb_files, desc="生成点云"):
        # 构建文件路径
        base_name = os.path.splitext(rgb_file)[0]
        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_path = os.path.join(input_dir, "DEPTH", rgb_file)
        seg_path = os.path.join(input_dir, "SEG", rgb_file)
        params_path = os.path.join(input_dir, "params", base_name + ".json")
        output_path = os.path.join(output_pcd_dir, base_name + ".ply")
        
        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [rgb_path, depth_path, seg_path, params_path]):
            logger.warning(f"跳过 {rgb_file}: 缺少必要文件")
            continue
        
        # 处理单帧
        pcd = generator.process_single_frame(
            rgb_path, depth_path, seg_path, params_path, output_path
        )
        
        if pcd is not None:
            success_count += 1
    
    logger.info(f"成功生成 {success_count}/{len(rgb_files)} 个点云文件")
    if max_samples is not None:
        logger.info(f"（总共 {total_files} 个文件，本次处理了 {len(rgb_files)} 个）")
    logger.info(f"点云保存目录: {output_pcd_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成带语义分割的点云数据集")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/tangqiansong/rgb2voxel/data/segmented_dataset",
        help="输入数据集目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/tangqiansong/rgb2voxel/data/pointed_dataset",
        help="输出点云数据集目录"
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.3,
        help="最小深度(米)"
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=0.8,
        help="最大深度(米)"
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="深度缩放因子 (毫米转米)"
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=0.001,
        help="体素下采样大小(米)"
    )
    parser.add_argument(
        "--use_rgb_color",
        action="store_true",
        help="使用RGB颜色而不是语义颜色"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试），None表示处理所有样本"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="测试模式：仅处理前N个样本（等同于 --max_samples N）"
    )
    
    args = parser.parse_args()
    
    # 如果使用 --test，覆盖 --max_samples
    max_samples = args.test if args.test is not None else args.max_samples
    
    generate_segmented_pointcloud_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        depth_scale=args.depth_scale,
        downsample_voxel_size=args.downsample,
        use_semantic_color=not args.use_rgb_color,
        max_samples=max_samples
    )

