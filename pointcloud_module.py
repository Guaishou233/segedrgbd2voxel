#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云生成模块
从多视角RGBD数据和语义分割结果生成带语义的点云数据（世界坐标系）
"""

import json
import os
import numpy as np
import cv2
import struct
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from color_palette import ColorPalette
from config_manager import ConfigManager

try:
    import open3d as o3d
    USE_OPEN3D = True
except ImportError:
    USE_OPEN3D = False
    raise ImportError("需要安装open3d: pip install open3d")


def load_dict_npy(file_name: str):
    """加载字典格式的numpy文件"""
    return np.load(file_name, allow_pickle=True).item()


class PointCloudGenerator:
    """点云生成器"""
    
    def __init__(self, config: ConfigManager, palette: ColorPalette = None):
        """
        初始化点云生成器
        
        Args:
            config: 配置管理器
            palette: 调色板
        """
        self.config = config
        self.palette = palette
        
        # 点云生成参数
        self.downsample_factor = config.get('pointcloud.downsample_factor', 1.0)
        self.min_depth = config.get('pointcloud.min_depth', 0.0)
        self.max_depth = config.get('pointcloud.max_depth', 1.0)
        self.depth_scale = config.get('pointcloud.depth_scale', 1000.0)
        self.max_time_diff_ms = config.get('pointcloud.max_time_diff_ms', 50)
        self.num_frames = config.get('pointcloud.num_frames', None)
        
        # 过滤参数
        self.voxel_size = config.get('pointcloud.filter.voxel_size', 0.0001)
        self.filter_neighbors = config.get('pointcloud.filter.num_neighbors', 10)
        self.filter_std_ratio = config.get('pointcloud.filter.std_ratio', 2.0)
        self.filter_radius = config.get('pointcloud.filter.radius', 0.01)
        
        # 输出配置
        self.output_folder = config.get('pointcloud.output_folder', 'POINTCLOUDS')
        self.overwrite = config.get('pointcloud.overwrite', False)
        
        # 内参缩放因子（从元数据读取或自动检测）
        # None 表示自动检测
        self.intrinsic_scale = None
    
    def rgbd_to_pointcloud(self, color_path: str, depth_path: str, 
                           seg_path: str, width: int, height: int,
                           intrinsic: np.ndarray, extrinsic: np.ndarray) -> Tuple:
        """
        从RGBD图像和语义分割生成带语义的点云
        
        Args:
            color_path: RGB图像路径
            depth_path: 深度图路径
            seg_path: 语义分割图路径
            width: 图像宽度
            height: 图像高度
            intrinsic: 相机内参矩阵 3x3
            extrinsic: 相机外参矩阵 4x4
        
        Returns:
            pcd: RGB点云
            seg_pcd: 语义点云
        """
        # 使用PIL读取所有图像（避免OpenCV的libpng兼容性问题）
        from PIL import Image as PILImage
        
        # 读取RGB图像
        try:
            color_pil = PILImage.open(color_path).convert('RGB')
            color = np.array(color_pil)
        except Exception as e:
            print(f"    Warning: 无法读取RGB图像: {color_path}, 错误: {e}")
            return None, None
        
        # 读取深度图
        try:
            depth_pil = PILImage.open(depth_path)
            depth = np.array(depth_pil).astype(np.float32)
        except Exception as e:
            print(f"    Warning: 无法读取深度图: {depth_path}, 错误: {e}")
            return None, None
        
        # 读取语义分割图
        if seg_path and os.path.exists(seg_path):
            try:
                seg_pil = PILImage.open(seg_path)
                segmentation = np.array(seg_pil)
                
                # 处理分割图格式
                if len(segmentation.shape) == 2:
                    # 灰度图，需要应用调色板转为RGB
                    if self.palette:
                        segmentation = self.palette.apply_to_mask(segmentation)
                    else:
                        # 转为3通道灰度
                        segmentation = np.stack([segmentation] * 3, axis=-1)
                elif len(segmentation.shape) == 3 and segmentation.shape[2] == 4:
                    # RGBA转RGB
                    segmentation = segmentation[:, :, :3]
            except Exception as e:
                print(f"    Warning: 无法读取分割图: {seg_path}，使用灰色替代")
                segmentation = np.ones_like(color) * 128
        else:
            # 如果没有分割图，使用灰色
            segmentation = np.ones_like(color) * 128
        
        original_height, original_width = depth.shape
        
        # 下采样
        if self.downsample_factor != 1.0:
            new_width = int(width / self.downsample_factor)
            new_height = int(height / self.downsample_factor)
            color = cv2.resize(color, (new_width, new_height))
            depth = cv2.resize(depth, (new_width, new_height))
            segmentation = cv2.resize(segmentation, (new_width, new_height), 
                                      interpolation=cv2.INTER_NEAREST)
            height, width = new_height, new_width
        
        # 深度转换：从毫米到米
        depth = depth / self.depth_scale
        
        # 深度范围过滤
        depth[depth < self.min_depth] = 0
        depth[depth > self.max_depth] = 0
        
        # 创建Open3D的RGBDImage
        color_o3d = o3d.geometry.Image(color.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        segmentation_o3d = o3d.geometry.Image(segmentation.astype(np.uint8))
        
        # depth_trunc 设置为与 max_depth 相同，确保不会截断有效深度
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False
        )
        
        segd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            segmentation_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False
        )
        
        # 设置内参
        # 自动检测内参是否需要缩放：
        # 如果 cx 大于 width 或 cy 大于 height，说明内参是针对更高分辨率的图像
        # 需要进行缩放以匹配当前图像尺寸
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        
        raw_fx = intrinsic[0, 0]
        raw_fy = intrinsic[1, 1]
        raw_cx = intrinsic[0, 2]
        raw_cy = intrinsic[1, 2]
        
        # 检测内参是否需要缩放
        # 如果 cx 明显大于 width 或 cy 明显大于 height，说明需要缩放
        intrinsic_scale = self.intrinsic_scale  # 从元数据获取的缩放因子
        
        if intrinsic_scale is None:
            # 自动检测：如果 cx > width 或 cy > height，计算缩放因子
            if raw_cx > width * 1.2 or raw_cy > height * 1.2:
                # 内参是针对更高分辨率的，需要缩放
                scale_x = width / (raw_cx * 2) if raw_cx > 0 else 1.0
                scale_y = height / (raw_cy * 2) if raw_cy > 0 else 1.0
                intrinsic_scale = min(scale_x, scale_y)
            else:
                # 内参已经匹配当前分辨率
                intrinsic_scale = 1.0
        
        fx = intrinsic_scale * raw_fx / self.downsample_factor
        fy = intrinsic_scale * raw_fy / self.downsample_factor
        cx = intrinsic_scale * raw_cx / self.downsample_factor
        cy = intrinsic_scale * raw_cy / self.downsample_factor
        
        intrinsic_o3d.set_intrinsics(width, height, fx, fy, cx, cy)
        
        # 创建点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic_o3d, extrinsic=extrinsic
        )
        
        seg_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            segd_image, intrinsic_o3d, extrinsic=extrinsic
        )
        
        return pcd, seg_pcd
    
    def merge_pointclouds(self, pcds_list: List, semantics_list: List) -> Tuple:
        """
        合并多个点云并融合语义信息
        
        Args:
            pcds_list: RGB点云列表
            semantics_list: 语义点云列表
        
        Returns:
            merged_pcd: 合并后的RGB点云
            merged_semantics: 合并后的语义点云颜色
        """
        if len(pcds_list) == 0:
            return None, None
        
        # 合并点云
        merged_points = []
        merged_colors = []
        merged_semantics = []
        
        for pcd, seg_pcd in zip(pcds_list, semantics_list):
            if pcd is None or len(pcd.points) == 0:
                continue
            
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) * 255.0
            semantics = np.asarray(seg_pcd.colors) * 255.0
            
            if len(semantics) != len(points):
                semantics = np.ones((len(points), 3), dtype=np.uint8) * 128
            
            merged_points.append(points)
            merged_colors.append(colors)
            merged_semantics.append(semantics)
        
        if len(merged_points) == 0:
            return None, None
        
        merged_points = np.vstack(merged_points)
        merged_colors = np.vstack(merged_colors)
        merged_semantics = np.vstack(merged_semantics)
        
        # 创建合并的点云
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors / 255.0)
        merged_semantics_array = merged_semantics
        
        # 体素下采样
        if self.voxel_size > 0:
            num_before = len(merged_pcd.points)
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            num_after = len(merged_pcd.points)
            
            if num_after < num_before:
                # 重新匹配语义标签
                original_pcd = o3d.geometry.PointCloud()
                original_pcd.points = o3d.utility.Vector3dVector(merged_points)
                original_tree = o3d.geometry.KDTreeFlann(original_pcd)
                
                downsampled_points = np.asarray(merged_pcd.points)
                downsampled_semantics = []
                for point in downsampled_points:
                    [_, idx, _] = original_tree.search_knn_vector_3d(point, 1)
                    downsampled_semantics.append(merged_semantics_array[idx[0]])
                merged_semantics_array = np.array(downsampled_semantics)
        
        # 统计离群点过滤
        if self.filter_neighbors > 0 and self.filter_std_ratio > 0:
            merged_pcd, inlier_indices = merged_pcd.remove_statistical_outlier(
                nb_neighbors=self.filter_neighbors,
                std_ratio=self.filter_std_ratio
            )
            merged_semantics_array = merged_semantics_array[inlier_indices]
        
        # 半径离群点过滤
        if self.filter_radius > 0:
            min_neighbors = max(3, self.filter_neighbors // 2)
            merged_pcd, inlier_indices = merged_pcd.remove_radius_outlier(
                nb_points=min_neighbors,
                radius=self.filter_radius
            )
            merged_semantics_array = merged_semantics_array[inlier_indices]
        
        merged_semantics = o3d.utility.Vector3dVector(merged_semantics_array / 255.0)
        
        return merged_pcd, merged_semantics
    
    def save_pointcloud(self, pcd, semantics, output_path: str) -> bool:
        """
        保存点云为PLY格式（包含语义信息）
        
        Args:
            pcd: Open3D点云对象
            semantics: 语义颜色
            output_path: 输出文件路径
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255.0
        semantics_array = np.asarray(semantics) * 255.0
        
        num_points = len(points)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar semantic_r
property uchar semantic_g
property uchar semantic_b
end_header
"""
            f.write(header.encode('ascii'))
            
            for i in range(num_points):
                f.write(struct.pack('<fff', points[i][0], points[i][1], points[i][2]))
                f.write(struct.pack('<BBB', 
                                   int(colors[i][0]),
                                   int(colors[i][1]),
                                   int(colors[i][2])))
                f.write(struct.pack('<BBB',
                                   int(semantics_array[i][0]),
                                   int(semantics_array[i][1]),
                                   int(semantics_array[i][2])))
        
        return True
    
    def save_metadata(self, matched_frame: Dict, cameras_data: Dict, 
                      output_path: str) -> bool:
        """保存点云的元数据"""
        metadata = {
            'query_timestamp': list(matched_frame.values())[0]['timestamp'] if matched_frame else None,
            'cameras': []
        }
        
        for cam_serial, frame_data in matched_frame.items():
            img_info = frame_data.get('image_info', {})
            camera_info = {
                'serial': cam_serial,
                'image_id': frame_data.get('image_id'),
                'timestamp': frame_data.get('timestamp'),
                'rgb_path': img_info.get('rgb_path', ''),
                'depth_path': img_info.get('depth_path', ''),
                'segmentation_path': img_info.get('segmentation_path', ''),
                'width': img_info.get('width', 640),
                'height': img_info.get('height', 360)
            }
            metadata['cameras'].append(camera_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True
    
    def load_camera_data(self, task_dir: str) -> Dict:
        """
        加载所有相机的数据
        优先从task目录的metadata.json读取相机参数，其次从calib目录读取
        
        Args:
            task_dir: 任务目录路径
        
        Returns:
            cameras_data: 相机数据字典
        """
        cameras_data = {}
        
        # 优先从metadata.json读取相机参数
        metadata_path = os.path.join(task_dir, "metadata.json")
        metadata_cameras = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                metadata_cameras = metadata.get('cameras', {})
                
                # 读取内参缩放因子（如果存在）
                if 'intrinsic_scale' in metadata:
                    self.intrinsic_scale = metadata['intrinsic_scale']
                    print(f"  从metadata.json读取内参缩放因子: {self.intrinsic_scale}")
                
                if metadata_cameras:
                    print(f"  从metadata.json加载相机参数")
            except Exception as e:
                print(f"  Warning: 无法读取metadata.json: {e}")
        
        # 如果metadata.json中没有相机参数，从calib目录读取
        intrinsics_dict = {}
        extrinsics_dict = {}
        if not metadata_cameras:
            task_name = os.path.basename(task_dir)
            calib_dir = self.config.get_calib_dir(task_name)
            
            if calib_dir and os.path.exists(calib_dir):
                intrinsics_path = os.path.join(calib_dir, "intrinsics.npy")
                extrinsics_path = os.path.join(calib_dir, "extrinsics.npy")
                
                if os.path.exists(intrinsics_path):
                    intrinsics_dict = load_dict_npy(intrinsics_path)
                if os.path.exists(extrinsics_path):
                    extrinsics_dict = load_dict_npy(extrinsics_path)
                
                if intrinsics_dict or extrinsics_dict:
                    print(f"  从calib目录加载相机参数")
        
        # 遍历相机文件夹
        for item in os.listdir(task_dir):
            cam_dir = os.path.join(task_dir, item)
            if not os.path.isdir(cam_dir) or not item.startswith('cam_'):
                continue
            
            cam_serial = item.replace('cam_', '')
            cam_key = f"cam_{cam_serial}"  # metadata.json中的key格式
            
            # 获取图像列表
            color_dir = os.path.join(cam_dir, 'color')
            depth_dir = os.path.join(cam_dir, 'depth')
            seg_dir = os.path.join(cam_dir, 'segmentation')
            
            if not os.path.exists(color_dir):
                continue
            
            # 获取所有图像
            color_files = sorted([f for f in os.listdir(color_dir) 
                                 if f.endswith('.jpg') or f.endswith('.png')])
            
            images = []
            for i, color_file in enumerate(color_files):
                timestamp = int(os.path.splitext(color_file)[0])
                depth_file = color_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                seg_file = depth_file
                
                # 获取图像尺寸
                color_path = os.path.join(color_dir, color_file)
                try:
                    from PIL import Image
                    img = Image.open(color_path)
                    width, height = img.size
                except:
                    width, height = 640, 360
                
                images.append({
                    'id': i + 1,
                    'file_name': color_file,
                    'width': width,
                    'height': height,
                    'rgb_path': os.path.join('color', color_file),
                    'depth_path': os.path.join('depth', depth_file),
                    'segmentation_path': os.path.join('segmentation', seg_file),
                    'timestamp': {'color': timestamp, 'depth': timestamp, 'segmentation': timestamp}
                })
            
            # 获取内参和外参（优先从metadata.json）
            intrinsic = None
            extrinsic = None
            
            if cam_key in metadata_cameras:
                # 从metadata.json读取
                cam_info = metadata_cameras[cam_key]
                intrinsic = np.array(cam_info.get('intrinsic'))
                extrinsic = np.array(cam_info.get('extrinsic'))
            else:
                # 从calib目录读取
                intrinsic = intrinsics_dict.get(cam_serial)
                extrinsic = extrinsics_dict.get(cam_serial)
                
                if intrinsic is not None:
                    intrinsic = np.array(intrinsic)
                if extrinsic is not None:
                    extrinsic = np.array(extrinsic)
                    if isinstance(extrinsic, list) or (isinstance(extrinsic, np.ndarray) and len(extrinsic.shape) > 2):
                        extrinsic = extrinsic[0] if len(extrinsic) > 0 else np.eye(4)
            
            # 确保内参是3x3矩阵
            if intrinsic is not None and intrinsic.shape[1] > 3:
                intrinsic = intrinsic[:3, :3]
            
            cameras_data[cam_serial] = {
                'serial': cam_serial,
                'dir': cam_dir,
                'images': images,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic
            }
            
            print(f"  加载相机 {cam_serial}: {len(images)} 张图像")
        
        return cameras_data
    
    def match_timestamps(self, cameras_data: Dict) -> List[Dict]:
        """
        匹配不同相机的时间戳
        
        Args:
            cameras_data: 相机数据字典
        
        Returns:
            matched_frames: 匹配的帧列表
        """
        # 收集每个相机的时间戳
        camera_timestamps = {}
        camera_timestamp_to_image = {}
        
        for cam_serial, cam_data in cameras_data.items():
            timestamps = []
            timestamp_to_image = {}
            
            for img in cam_data['images']:
                ts = img['timestamp']['color']
                if ts is not None:
                    timestamps.append(ts)
                    timestamp_to_image[ts] = img
            
            timestamps = sorted(list(set(timestamps)))
            camera_timestamps[cam_serial] = timestamps
            camera_timestamp_to_image[cam_serial] = timestamp_to_image
        
        if len(camera_timestamps) == 0:
            return []
        
        # 选择帧数最少的相机作为基准
        base_camera = min(camera_timestamps.items(), key=lambda x: len(x[1]))
        base_cam_serial = base_camera[0]
        base_timestamps = base_camera[1]
        
        if len(base_timestamps) == 0:
            return []
        
        print(f"  选择相机 {base_cam_serial} 作为基准（共 {len(base_timestamps)} 帧）")
        
        # 匹配帧
        matched_frames = []
        
        for base_ts in base_timestamps:
            matched_frame = {}
            all_matched = True
            
            # 添加基准相机的帧
            base_img = camera_timestamp_to_image[base_cam_serial][base_ts]
            matched_frame[base_cam_serial] = {
                'image_id': base_img['id'],
                'image_info': base_img,
                'timestamp': base_ts
            }
            
            # 查找其他相机的匹配帧
            for cam_serial, timestamps in camera_timestamps.items():
                if cam_serial == base_cam_serial:
                    continue
                
                if len(timestamps) == 0:
                    all_matched = False
                    break
                
                # 找最接近的时间戳
                closest_ts = min(timestamps, key=lambda x: abs(x - base_ts))
                time_diff = abs(closest_ts - base_ts)
                
                if time_diff <= self.max_time_diff_ms:
                    img_info = camera_timestamp_to_image[cam_serial][closest_ts]
                    matched_frame[cam_serial] = {
                        'image_id': img_info['id'],
                        'image_info': img_info,
                        'timestamp': closest_ts
                    }
                else:
                    all_matched = False
                    break
            
            if all_matched and len(matched_frame) == len(camera_timestamps):
                matched_frames.append(matched_frame)
        
        return matched_frames
    
    def process_task(self, task_dir: str) -> Dict:
        """
        处理单个任务的点云生成
        
        Args:
            task_dir: 任务目录路径
        
        Returns:
            处理结果统计
        """
        task_name = os.path.basename(task_dir)
        print(f"\n处理任务点云: {task_name}")
        
        # 加载相机数据
        print("  加载相机数据...")
        cameras_data = self.load_camera_data(task_dir)
        
        if len(cameras_data) == 0:
            return {'success': False, 'error': '无相机数据'}
        
        print(f"  找到 {len(cameras_data)} 个相机")
        
        # 匹配时间戳
        print("  匹配时间戳...")
        matched_frames = self.match_timestamps(cameras_data)
        print(f"  匹配到 {len(matched_frames)} 个多视角帧")
        
        if len(matched_frames) == 0:
            return {'success': False, 'error': '无匹配帧'}
        
        # 限制帧数
        if self.num_frames is not None:
            matched_frames = matched_frames[:self.num_frames]
        
        # 创建输出目录
        output_dir = os.path.join(task_dir, self.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理统计
        stats = {
            'success': True,
            'total_frames': len(matched_frames),
            'processed_frames': 0,
            'skipped_frames': 0,
            'failed_frames': 0
        }
        
        # 处理每一帧
        print(f"  生成点云...")
        
        for matched_frame in tqdm(matched_frames, desc="  生成点云"):
            try:
                # 获取时间戳
                timestamp = list(matched_frame.values())[0]['timestamp']
                output_path = os.path.join(output_dir, f"{timestamp}.ply")
                
                # 检查是否需要跳过
                if not self.overwrite and os.path.exists(output_path):
                    stats['skipped_frames'] += 1
                    continue
                
                # 为每个相机生成点云
                pcds_list = []
                semantics_list = []
                
                for cam_serial, frame_data in matched_frame.items():
                    cam_data = cameras_data[cam_serial]
                    img_info = frame_data['image_info']
                    
                    # 构建文件路径
                    rgb_path = os.path.join(cam_data['dir'], img_info['rgb_path'])
                    depth_path = os.path.join(cam_data['dir'], img_info['depth_path'])
                    seg_path = os.path.join(cam_data['dir'], img_info.get('segmentation_path', ''))
                    
                    # 检查文件是否存在
                    if not os.path.exists(rgb_path):
                        tqdm.write(f"    跳过 {cam_serial}: RGB文件不存在 {rgb_path}")
                        continue
                    if not os.path.exists(depth_path):
                        tqdm.write(f"    跳过 {cam_serial}: 深度图不存在 {depth_path}")
                        continue
                    
                    intrinsic = cam_data.get('intrinsic')
                    extrinsic = cam_data.get('extrinsic')
                    
                    if intrinsic is None:
                        tqdm.write(f"    跳过 {cam_serial}: 无内参矩阵")
                        continue
                    if extrinsic is None:
                        tqdm.write(f"    跳过 {cam_serial}: 无外参矩阵")
                        continue
                    
                    # 生成点云
                    try:
                        pcd, seg_pcd = self.rgbd_to_pointcloud(
                            rgb_path, depth_path, seg_path,
                            img_info.get('width', 640),
                            img_info.get('height', 360),
                            intrinsic, extrinsic
                        )
                        
                        if pcd is not None and len(pcd.points) > 0:
                            pcds_list.append(pcd)
                            semantics_list.append(seg_pcd)
                    except Exception as e:
                        tqdm.write(f"    相机 {cam_serial} 点云生成失败: {e}")
                        continue
                
                # 合并点云
                if len(pcds_list) > 0:
                    merged_pcd, merged_semantics = self.merge_pointclouds(
                        pcds_list, semantics_list
                    )
                    
                    if merged_pcd is not None and len(merged_pcd.points) > 0:
                        # 保存点云
                        self.save_pointcloud(merged_pcd, merged_semantics, output_path)
                        
                        # 保存元数据
                        metadata_path = os.path.join(output_dir, f"{timestamp}.json")
                        self.save_metadata(matched_frame, cameras_data, metadata_path)
                        
                        stats['processed_frames'] += 1
                    else:
                        stats['failed_frames'] += 1
                else:
                    stats['failed_frames'] += 1
            
            except Exception as e:
                print(f"\n  Error: {e}")
                stats['failed_frames'] += 1
                continue
        
        return stats


def generate_pointclouds(config: ConfigManager, palette: ColorPalette = None) -> Dict:
    """
    生成数据集的点云
    
    Args:
        config: 配置管理器
        palette: 调色板
    
    Returns:
        处理结果统计
    """
    generator = PointCloudGenerator(config, palette)
    
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
        task_stats = generator.process_task(str(task_dir))
        results['tasks'][task_name] = task_stats
    
    return results


if __name__ == "__main__":
    # 测试点云生成模块
    from config_manager import load_config
    
    config = load_config()
    results = generate_pointclouds(config)
    
    print("\n点云生成完成！")
    for task_name, task_stats in results.get('tasks', {}).items():
        print(f"  任务 {task_name}: {task_stats}")

