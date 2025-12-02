#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从多视角RGBD数据生成带语义的点云数据（世界坐标系）
参考rh20t_api/utils/point_cloud.py的点云生成方式
"""

import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import struct
from collections import defaultdict
try:
    import open3d as o3d
    USE_OPEN3D = True
except ImportError:
    USE_OPEN3D = False
    print("Warning: open3d未安装，将无法生成点云")
    exit(1)

# 检查pycocotools是否安装（用于解码COCO格式的RLE mask）
try:
    import pycocotools.mask as mask_util
    USE_PYCOCOTOOLS = True
except ImportError:
    USE_PYCOCOTOOLS = False
    print("Warning: pycocotools未安装，无法解码COCO格式的RLE mask")
    print("  这将导致语义标注无法正确应用到点云中")
    print("  请安装: pip install pycocotools")

def load_dict_npy(file_name: str):
    """加载字典格式的numpy文件"""
    return np.load(file_name, allow_pickle=True).item()

def decode_rle_to_mask(rle, height, width):
    """
    将RLE格式的segmentation解码为二值mask
    
    Args:
        rle: RLE格式的segmentation字典，包含'size'和'counts'
        height: mask的高度
        width: mask的宽度
    
    Returns:
        mask: 二值mask (bool数组)，如果解码失败返回None
    """
    if not USE_PYCOCOTOOLS:
        return None
    
    try:
        # 创建rle的副本，避免修改原始数据
        rle_copy = rle.copy()
        # 确保counts是bytes格式（不修改原始数据）
        if isinstance(rle_copy['counts'], str):
            rle_copy['counts'] = rle_copy['counts'].encode('utf-8')
        
        # 解码RLE
        mask = mask_util.decode(rle_copy)
        
        return mask.astype(bool)
    except Exception as e:
        return None

def create_semantic_mask_from_annotations(annotations, image_id, height, width):
    """
    从COCO格式的annotations创建语义分割mask
    
    Args:
        annotations: 所有annotations列表
        image_id: 图像ID
        height: 图像高度
        width: 图像宽度
    
    Returns:
        semantic_mask: 语义mask，值为1表示robot arm，值为2表示others
    """
    # 创建默认mask（全部为others，类别2）
    semantic_mask = np.ones((height, width), dtype=np.uint8) * 2
    
    # 找到该图像的所有annotations
    image_annotations = [ann for ann in annotations if ann.get('image_id') == image_id]
    
    if len(image_annotations) == 0:
        return semantic_mask
    
    # 合并所有annotations的mask
    robot_arm_mask = np.zeros((height, width), dtype=bool)
    
    for ann in image_annotations:
        category_id = ann.get('category_id', 2)
        segmentation = ann.get('segmentation')
        
        if segmentation is None:
            continue
        
        # 解码RLE mask
        try:
            if isinstance(segmentation, dict):
                # RLE格式
                mask = decode_rle_to_mask(segmentation, height, width)
                if mask is None:
                    continue
            elif isinstance(segmentation, list):
                # 多边形格式（如果有）
                if not USE_PYCOCOTOOLS:
                    continue
                rle = mask_util.frPyObjects(segmentation, height, width)
                mask = mask_util.decode(rle)
                mask = mask.astype(bool)
            else:
                continue
            
            # 根据category_id添加到对应的mask
            if category_id == 1:  # robot arm
                robot_arm_mask = robot_arm_mask | mask
        except Exception as e:
            continue
    
    # 设置robot arm区域为类别1
    semantic_mask[robot_arm_mask] = 1
    
    return semantic_mask

def load_segmentation_mask(seg_path):
    """
    加载语义分割mask图像
    
    Args:
        seg_path: 分割图像路径
    
    Returns:
        semantic_mask: 语义mask，值为1表示robot arm，值为2表示others
    """
    if not os.path.exists(seg_path):
        return None
    
    seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg_img is None:
        return None
    
    # 假设分割图像中，非零值表示robot arm（类别1），零值表示others（类别2）
    # 根据实际数据格式调整
    semantic_mask = np.ones_like(seg_img, dtype=np.uint8) * 2
    semantic_mask[seg_img > 0] = 1
    
    return semantic_mask

def rgbd_to_pointcloud_with_semantics(
    color_image_path: str,
    depth_image_path: str,
    segmentation_path: str,
    width: int,
    height: int,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray = np.eye(4),
    annotations: list = None,
    image_id: int = None,
    downsample_factor: float = 1.0,
    min_depth_m: float = 0.3,
    max_depth_m: float = 1.0,
    depth_scale: float = 1000.0
):
    """
    从RGBD图像和语义分割生成带语义的点云（参考rh20t_api/utils/point_cloud.py）
    
    Args:
        color_image_path: RGB图像路径
        depth_image_path: 深度图路径
        segmentation_path: 语义分割图路径
        width: 图像宽度
        height: 图像高度
        intrinsic: 相机内参矩阵 3x3或3x4
        extrinsic: 相机外参矩阵 4x4（相机到世界）
        annotations: COCO格式的annotations列表（可选）
        image_id: 图像ID（用于从annotations中查找语义信息）
        downsample_factor: 下采样因子
        min_depth_m: 最小有效深度（米）
        max_depth_m: 最大有效深度（米）
        depth_scale: 深度缩放因子（默认1000，表示深度以毫米存储）
    
    Returns:
        pcd: Open3D点云对象（世界坐标系）
        semantics: 语义标签数组（与点云点数相同）
    """
    # 读取RGB和深度图
    color = cv2.cvtColor(cv2.imread(color_image_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    original_height, original_width = depth.shape
    
    # 下采样图像（如果需要）
    if downsample_factor != 1.0:
        new_width = int(width / downsample_factor)
        new_height = int(height / downsample_factor)
        color = cv2.resize(color, (new_width, new_height))
        depth = cv2.resize(depth, (new_width, new_height))
        height, width = new_height, new_width
    
    # 深度值转换：从毫米到米
    depth /= depth_scale
    
    # 深度范围过滤
    depth[depth < min_depth_m] = 0
    depth[depth > max_depth_m] = 0
    
    # 创建Open3D的RGBDImage
    color_o3d = o3d.geometry.Image(color.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,
        convert_rgb_to_intensity=False
    )
    
    # 设置内参（参考rh20t_api，内参需要乘以0.5并除以downsample_factor）
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    fx = 0.5 * intrinsic[0, 0] / downsample_factor
    fy = 0.5 * intrinsic[1, 1] / downsample_factor
    cx = 0.5 * intrinsic[0, 2] / downsample_factor
    cy = 0.5 * intrinsic[1, 2] / downsample_factor
    
    intrinsic_o3d.set_intrinsics(width, height, fx, fy, cx, cy)
    
    # 使用Open3D创建点云（相机坐标系）
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic_o3d, extrinsic=extrinsic
    )
    
    # 提取点云数据（已经在世界坐标系中）
    points_world = np.asarray(pcd.points)
    colors_o3d = np.asarray(pcd.colors) * 255.0  # Open3D颜色范围是0-1，转换为0-255
    
    if len(points_world) == 0:
        return None, None
    
    # 获取语义mask
    semantic_mask = None
    
    # 优先使用分割图像
    if segmentation_path and os.path.exists(segmentation_path):
        semantic_mask = load_segmentation_mask(segmentation_path)
        if semantic_mask is not None and downsample_factor != 1.0:
            semantic_mask = cv2.resize(
                semantic_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
    
    # 如果没有分割图像，尝试从annotations创建
    if semantic_mask is None and annotations is not None and image_id is not None:
        semantic_mask = create_semantic_mask_from_annotations(
            annotations, image_id, original_height, original_width
        )
        if downsample_factor != 1.0:
            semantic_mask = cv2.resize(
                semantic_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
    
    # 如果仍然没有语义mask，默认为others（类别2）
    if semantic_mask is None:
        semantic_mask = np.ones((height, width), dtype=np.uint8) * 2
    
    # 将点云坐标投影回像素坐标，以获取对应的语义标签
    semantics = []
    coord_scale_x = original_width / width if width > 0 and original_width != width else 1.0
    coord_scale_y = original_height / height if height > 0 and original_height != height else 1.0
    
    for i, point in enumerate(points_world):
        # 需要将世界坐标转换回相机坐标进行投影
        # 计算逆变换：P_camera = R^T * (P_world - t)
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        point_cam = R.T @ (point - t)
        
        z = point_cam[2]
        if z <= 0:
            semantics.append(2)  # others
            continue
        
        u = int(fx * point_cam[0] / z + cx)
        v = int(fy * point_cam[1] / z + cy)
        
        # 如果mask尺寸与投影尺寸不同，需要缩放坐标
        if coord_scale_x != 1.0 or coord_scale_y != 1.0:
            u = int(u * coord_scale_x)
            v = int(v * coord_scale_y)
        
        # 检查边界
        if 0 <= u < semantic_mask.shape[1] and 0 <= v < semantic_mask.shape[0]:
            try:
                semantic_label = semantic_mask[v, u]
                semantics.append(semantic_label)
            except IndexError:
                semantics.append(2)  # others
        else:
            semantics.append(2)  # others
    
    semantics = np.array(semantics)
    
    # 更新点云颜色（可选：根据语义标签着色）
    # 这里保持原始RGB颜色，语义信息单独存储
    
    return pcd, semantics

def merge_pointclouds_with_semantics(pcds_list, semantics_list, 
                                     downsample_voxel_size_m=0.0001,
                                     filter_num_neighbor=10,
                                     filter_std_ratio=2.0,
                                     filter_radius_m=0.01):
    """
    合并多个点云并融合语义信息（参考rh20t_api/utils/point_cloud.py）
    
    Args:
        pcds_list: 点云列表
        semantics_list: 语义标签列表
        downsample_voxel_size_m: 体素下采样大小
        filter_num_neighbor: 统计离群点过滤的邻居数
        filter_std_ratio: 统计离群点过滤的标准差比例
        filter_radius_m: 半径离群点过滤的半径
    
    Returns:
        merged_pcd: 合并后的点云
        merged_semantics: 合并后的语义标签
    """
    if len(pcds_list) == 0:
        return None, None
    
    # 合并点云
    merged_points = []
    merged_colors = []
    merged_semantics = []
    
    for pcd, semantics in zip(pcds_list, semantics_list):
        if pcd is None or len(pcd.points) == 0:
            continue
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255.0  # Open3D颜色范围是0-1
        
        # 确保语义标签长度与点数匹配
        if len(semantics) != len(points):
            # 如果长度不匹配，使用默认值
            semantics = np.ones(len(points), dtype=np.uint8) * 2
        
        merged_points.append(points)
        merged_colors.append(colors)
        merged_semantics.append(semantics)
    
    if len(merged_points) == 0:
        return None, None
    
    merged_points = np.vstack(merged_points)
    merged_colors = np.vstack(merged_colors)
    merged_semantics = np.hstack(merged_semantics)
    
    # 创建Open3D点云
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors / 255.0)
    
    # 体素下采样（使用KDTree最近邻来更新语义标签）
    if downsample_voxel_size_m > 0:
        # 保存原始点云和语义标签
        original_points = merged_points.copy()
        original_semantics = merged_semantics.copy()
        
        # 下采样点云
        merged_pcd = merged_pcd.voxel_down_sample(downsample_voxel_size_m)
        downsampled_points = np.asarray(merged_pcd.points)
        
        # 使用KDTree找到最近邻来更新语义标签
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(original_points)
            _, indices = tree.query(downsampled_points, k=1)
            merged_semantics = original_semantics[indices]
        except ImportError:
            # 如果scipy不可用，使用简单方法：取第一个点的语义标签
            # 这会导致语义信息不准确，但至少不会报错
            print("Warning: scipy未安装，体素下采样后的语义标签可能不准确")
            merged_semantics = np.ones(len(downsampled_points), dtype=np.uint8) * 2
    
    # 统计离群点过滤
    if filter_num_neighbor > 0 and filter_std_ratio > 0:
        _, ind = merged_pcd.remove_statistical_outlier(
            nb_neighbors=filter_num_neighbor,
            std_ratio=filter_std_ratio
        )
        merged_pcd = merged_pcd.select_by_index(ind)
        # 更新语义标签
        merged_semantics = merged_semantics[ind]
    
    # 半径离群点过滤
    if filter_radius_m > 0 and filter_num_neighbor > 0:
        _, ind = merged_pcd.remove_radius_outlier(
            nb_points=filter_num_neighbor,
            radius=filter_radius_m
        )
        merged_pcd = merged_pcd.select_by_index(ind)
        # 更新语义标签
        merged_semantics = merged_semantics[ind]
    
    return merged_pcd, merged_semantics

def save_pointcloud_ply_with_semantics(pcd, semantics, output_path):
    """
    保存点云为PLY格式（包含语义信息）
    
    Args:
        pcd: Open3D点云对象
        semantics: 语义标签数组
        output_path: 输出文件路径
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0  # Open3D颜色范围是0-1
    
    num_points = len(points)
    
    with open(output_path, 'wb') as f:
        # PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar semantic
end_header
"""
        f.write(header.encode('ascii'))
        
        # 写入点云数据
        for i in range(num_points):
            # 坐标
            f.write(struct.pack('<fff', points[i][0], points[i][1], points[i][2]))
            # 颜色
            f.write(struct.pack('<BBB',
                               int(colors[i][0]),
                               int(colors[i][1]),
                               int(colors[i][2])))
            # 语义标签
            semantic_label = int(semantics[i]) if i < len(semantics) else 2
            f.write(struct.pack('<B', semantic_label))
    
    return True

def load_camera_annotations(scene_dir):
    """
    加载所有相机的annotations.json文件
    
    Args:
        scene_dir: 场景目录路径
    
    Returns:
        cameras_data: 字典，键为相机序列号，值为相机数据字典
    """
    cameras_data = {}
    
    # 查找所有相机文件夹
    for item in os.listdir(scene_dir):
        cam_dir = os.path.join(scene_dir, item)
        if not os.path.isdir(cam_dir) or not item.startswith('cam_'):
            continue
        
        cam_serial = item.replace('cam_', '')
        annotations_path = os.path.join(cam_dir, 'annotations.json')
        
        if not os.path.exists(annotations_path):
            print(f"Warning: 相机 {cam_serial} 的annotations.json不存在")
            continue
        
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                cam_data = json.load(f)
            
            cameras_data[cam_serial] = {
                'serial': cam_serial,
                'dir': cam_dir,
                'info': cam_data.get('info', {}),
                'images': cam_data.get('images', []),
                'annotations': cam_data.get('annotations', []),
                'categories': cam_data.get('categories', [])
            }
            
            # 提取内参和外参
            if 'camera_intrinsic' in cam_data.get('info', {}):
                cameras_data[cam_serial]['intrinsic'] = np.array(
                    cam_data['info']['camera_intrinsic']
                )[:3, :3]  # 只取3x3部分
            if 'camera_extrinsic' in cam_data.get('info', {}):
                cameras_data[cam_serial]['extrinsic'] = np.array(
                    cam_data['info']['camera_extrinsic']
                )
            
            print(f"加载相机 {cam_serial}: {len(cameras_data[cam_serial]['images'])} 张图像")
        
        except Exception as e:
            print(f"Error loading annotations for {cam_serial}: {e}")
            continue
    
    return cameras_data

def match_timestamps(cameras_data, tolerance_ms=100):
    """
    匹配不同相机的时间戳，找到同一时刻的多视角图像
    
    Args:
        cameras_data: 相机数据字典
        tolerance_ms: 时间戳容差（毫秒）
    
    Returns:
        matched_frames: 匹配的帧列表，每个元素是一个字典，包含所有相机的图像信息
    """
    # 收集所有时间戳
    all_timestamps = []
    for cam_serial, cam_data in cameras_data.items():
        for img in cam_data['images']:
            timestamp = img.get('timestamp', {})
            color_ts = timestamp.get('color', None)
            if color_ts is not None:
                all_timestamps.append((cam_serial, img['id'], color_ts))
    
    # 按时间戳排序
    all_timestamps.sort(key=lambda x: x[2])
    
    # 匹配时间戳
    matched_frames = []
    current_group = []
    current_ts = None
    
    for cam_serial, img_id, ts in all_timestamps:
        if current_ts is None or abs(ts - current_ts) <= tolerance_ms:
            # 属于同一组
            current_group.append((cam_serial, img_id, ts))
            if current_ts is None:
                current_ts = ts
        else:
            # 新的一组
            if len(current_group) > 0:
                # 创建匹配帧
                matched_frame = {}
                for cs, iid, t in current_group:
                    # 找到对应的图像信息
                    img_info = None
                    for img in cameras_data[cs]['images']:
                        if img['id'] == iid:
                            img_info = img
                            break
                    
                    if img_info:
                        matched_frame[cs] = {
                            'image_id': iid,
                            'image_info': img_info,
                            'timestamp': t
                        }
                
                if len(matched_frame) > 0:
                    matched_frames.append(matched_frame)
            
            # 开始新组
            current_group = [(cam_serial, img_id, ts)]
            current_ts = ts
    
    # 处理最后一组
    if len(current_group) > 0:
        matched_frame = {}
        for cs, iid, t in current_group:
            img_info = None
            for img in cameras_data[cs]['images']:
                if img['id'] == iid:
                    img_info = img
                    break
            
            if img_info:
                matched_frame[cs] = {
                    'image_id': iid,
                    'image_info': img_info,
                    'timestamp': t
                }
        
        if len(matched_frame) > 0:
            matched_frames.append(matched_frame)
    
    return matched_frames

def process_multiview_pointcloud(scene_dir, output_dir=None, num_frames=None,
                                 downsample_factor=1.0, min_depth_m=0.3, max_depth_m=1.0,
                                 downsample_voxel_size_m=0.0001, filter_num_neighbor=10,
                                 filter_std_ratio=2.0, filter_radius_m=0.01):
    """
    处理多视角点云生成
    
    Args:
        scene_dir: 场景目录路径
        output_dir: 输出目录路径（如果为None，则在scene_dir下创建POINTCLOUDS_MULTIVIEW）
        num_frames: 处理的帧数（None表示全部）
        downsample_factor: 图像下采样因子
        min_depth_m: 最小有效深度（米）
        max_depth_m: 最大有效深度（米）
        downsample_voxel_size_m: 点云体素下采样大小
        filter_num_neighbor: 统计离群点过滤的邻居数
        filter_std_ratio: 统计离群点过滤的标准差比例
        filter_radius_m: 半径离群点过滤的半径
    """
    print("=" * 60)
    print("多视角语义点云生成")
    print("=" * 60)
    
    # 加载所有相机的annotations
    print("\n正在加载相机数据...")
    cameras_data = load_camera_annotations(scene_dir)
    
    if len(cameras_data) == 0:
        print("Error: 没有找到有效的相机数据")
        return
    
    print(f"找到 {len(cameras_data)} 个相机")
    
    # 匹配时间戳
    print("\n正在匹配时间戳...")
    matched_frames = match_timestamps(cameras_data, tolerance_ms=100)
    print(f"匹配到 {len(matched_frames)} 个多视角帧")
    
    if len(matched_frames) == 0:
        print("Error: 没有匹配到多视角帧")
        return
    
    # 限制处理数量
    if num_frames is not None:
        matched_frames = matched_frames[:num_frames]
        print(f"处理前 {num_frames} 帧...")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(scene_dir, "POINTCLOUDS_MULTIVIEW")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每一帧
    print(f"\n开始生成点云...")
    processed_count = 0
    failed_count = 0
    
    for frame_idx, matched_frame in enumerate(tqdm(matched_frames, desc="生成点云")):
        try:
            # 为每个相机生成点云
            pcds_list = []
            semantics_list = []
            
            for cam_serial, frame_data in matched_frame.items():
                cam_data = cameras_data[cam_serial]
                img_info = frame_data['image_info']
                image_id = frame_data['image_id']
                
                # 构建文件路径
                rgb_path = os.path.join(cam_data['dir'], img_info['rgb_path'])
                depth_path = os.path.join(cam_data['dir'], img_info['depth_path'])
                seg_path = os.path.join(cam_data['dir'], img_info.get('segmentation_path', ''))
                
                # 检查文件是否存在
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    continue
                
                # 获取内参和外参
                intrinsic = cam_data.get('intrinsic')
                extrinsic = cam_data.get('extrinsic')
                
                if intrinsic is None or extrinsic is None:
                    continue
                
                # 生成点云
                pcd, semantics = rgbd_to_pointcloud_with_semantics(
                    color_image_path=rgb_path,
                    depth_image_path=depth_path,
                    segmentation_path=seg_path if os.path.exists(seg_path) else None,
                    width=img_info.get('width', 640),
                    height=img_info.get('height', 360),
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    annotations=cam_data.get('annotations', []),
                    image_id=image_id,
                    downsample_factor=downsample_factor,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m
                )
                
                if pcd is not None and len(pcd.points) > 0:
                    pcds_list.append(pcd)
                    semantics_list.append(semantics)
            
            # 合并所有相机的点云
            if len(pcds_list) > 0:
                merged_pcd, merged_semantics = merge_pointclouds_with_semantics(
                    pcds_list, semantics_list,
                    downsample_voxel_size_m=downsample_voxel_size_m,
                    filter_num_neighbor=filter_num_neighbor,
                    filter_std_ratio=filter_std_ratio,
                    filter_radius_m=filter_radius_m
                )
                
                if merged_pcd is not None and len(merged_pcd.points) > 0:
                    # 保存点云
                    timestamp = list(matched_frame.values())[0]['timestamp']
                    output_path = os.path.join(output_dir, f"{timestamp}.ply")
                    save_pointcloud_ply_with_semantics(merged_pcd, merged_semantics, output_path)
                    processed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        
        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 帧")
    print(f"失败: {failed_count} 帧")
    print(f"点云保存在: {output_dir}")

def load_pointcloud_with_semantics(ply_path):
    """
    加载带语义信息的点云
    
    Returns:
        points: Nx3 点云坐标
        colors: Nx3 RGB颜色
        semantics: Nx1 语义标签
    """
    points = []
    colors = []
    semantics = []
    
    with open(ply_path, 'rb') as f:
        # 读取header
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if b'end_header' in line:
                break
        
        # 解析header获取点数
        num_points = 0
        for line in header_lines:
            if b'element vertex' in line:
                num_points = int(line.split()[-1])
                break
        
        # 读取点云数据
        for _ in range(num_points):
            # x, y, z (float)
            x, y, z = struct.unpack('<fff', f.read(12))
            # r, g, b (uchar)
            r, g, b = struct.unpack('<BBB', f.read(3))
            # semantic (uchar)
            semantic = struct.unpack('<B', f.read(1))[0]
            
            points.append([x, y, z])
            colors.append([r, g, b])
            semantics.append(semantic)
    
    return np.array(points), np.array(colors), np.array(semantics)

def visualize_pointclouds_interactive(scene_dir, num_samples=10):
    """
    交互式可视化点云数据（使用Open3D）
    
    Args:
        scene_dir: 场景目录路径
        num_samples: 可视化的样本数量
    """
    if not USE_OPEN3D:
        print("Error: Open3D未安装，无法使用交互式可视化")
        return
    
    pointcloud_dir = os.path.join(scene_dir, "POINTCLOUDS_MULTIVIEW")
    
    if not os.path.exists(pointcloud_dir):
        print(f"Error: 点云目录不存在: {pointcloud_dir}")
        return
    
    # 获取点云文件列表
    ply_files = sorted([f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')])
    
    if len(ply_files) == 0:
        print("没有找到点云文件")
        return
    
    # 限制数量
    ply_files = ply_files[:num_samples]
    
    print(f"\n开始交互式可视化 {len(ply_files)} 个点云...")
    print("=" * 60)
    print("操作说明：")
    print("  - 鼠标左键拖拽：旋转视角")
    print("  - 鼠标右键拖拽：平移视角")
    print("  - 滚轮：缩放")
    print("  - 'Q' 或关闭窗口：退出当前点云，查看下一个")
    print("  - 'R'：重置视角")
    print("  - 'S'：切换显示模式（RGB颜色 / 语义颜色）")
    print("=" * 60)
    print("可视化说明：")
    print("  - 红色点 = robot arm (类别1)")
    print("  - 灰色点 = others (类别2)")
    print("=" * 60)
    
    # 为每个点云创建交互式可视化
    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(pointcloud_dir, ply_file)
        
        try:
            # 加载点云
            points, colors, semantics = load_pointcloud_with_semantics(ply_path)
            
            if len(points) == 0:
                print(f"Warning: {ply_file} 是空点云，跳过")
                continue
            
            # 统计信息
            robot_arm_count = np.sum(semantics == 1)
            others_count = np.sum(semantics == 2)
            
            print(f"\n点云 {i+1}/{len(ply_files)}: {ply_file}")
            print(f"  总点数: {len(points):,}")
            print(f"  Robot arm点数: {robot_arm_count:,} ({robot_arm_count/len(points)*100:.1f}%)")
            print(f"  Others点数: {others_count:,} ({others_count/len(points)*100:.1f}%)")
            
            # 下采样以加快可视化（如果点太多）
            max_points = 500000
            if len(points) > max_points:
                print(f"  点云较大，下采样到 {max_points:,} 个点以加快可视化...")
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
                colors = colors[indices]
                semantics = semantics[indices]
            
            # 创建RGB点云
            pcd_rgb = o3d.geometry.PointCloud()
            pcd_rgb.points = o3d.utility.Vector3dVector(points)
            pcd_rgb.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D颜色范围是0-1
            
            # 创建语义点云（红色=robot arm, 灰色=others）
            semantic_colors = np.zeros((len(semantics), 3))
            robot_arm_mask = semantics == 1
            others_mask = semantics == 2
            semantic_colors[robot_arm_mask] = [1.0, 0.0, 0.0]  # 红色 - robot arm
            semantic_colors[others_mask] = [0.5, 0.5, 0.5]    # 灰色 - others
            
            pcd_semantic = o3d.geometry.PointCloud()
            pcd_semantic.points = o3d.utility.Vector3dVector(points)
            pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
            
            # 创建交互式可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"点云可视化 {i+1}/{len(ply_files)}: {ply_file}",
                            width=1920, height=1080)
            
            # 默认显示语义颜色
            current_pcd = pcd_semantic
            show_semantic = True
            vis.add_geometry(current_pcd)
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.point_size = 1.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
            
            # 设置初始视角
            view_control = vis.get_view_control()
            view_control.set_zoom(0.7)
            
            # 使用pynput库监听键盘事件（如果可用）
            keyboard_listener = None
            keyboard_actions = {'toggle_color': False, 'reset_view': False}
            
            try:
                from pynput import keyboard
                
                def on_press(key):
                    try:
                        if hasattr(key, 'char') and key.char:
                            if key.char.lower() == 's':
                                keyboard_actions['toggle_color'] = True
                            elif key.char.lower() == 'r':
                                keyboard_actions['reset_view'] = True
                    except:
                        pass
                
                keyboard_listener = keyboard.Listener(on_press=on_press)
                keyboard_listener.start()
                print("  键盘快捷键已启用（S=切换颜色，R=重置视角）")
            except ImportError:
                print("  提示：安装pynput库可启用键盘快捷键: pip install pynput")
            
            # 运行可视化器（阻塞，直到窗口关闭）
            print(f"\n正在显示点云 {i+1}/{len(ply_files)}...")
            print("  关闭窗口查看下一个点云")
            if keyboard_listener:
                print("  按 'S' 切换显示模式（RGB/语义）")
                print("  按 'R' 重置视角")
            print("  鼠标左键拖拽：旋转 | 右键拖拽：平移 | 滚轮：缩放")
            
            # 在主循环中处理键盘事件
            import time
            initial_view_params = view_control.convert_to_pinhole_camera_parameters()
            
            while vis.poll_events():
                # 处理键盘动作
                if keyboard_actions['toggle_color']:
                    keyboard_actions['toggle_color'] = False
                    vis.remove_geometry(current_pcd, reset_bounding_box=False)
                    if show_semantic:
                        current_pcd = pcd_rgb
                        show_semantic = False
                        print("  切换到RGB颜色模式")
                    else:
                        current_pcd = pcd_semantic
                        show_semantic = True
                        print("  切换到语义颜色模式（红色=robot arm, 灰色=others）")
                    vis.add_geometry(current_pcd, reset_bounding_box=False)
                
                if keyboard_actions['reset_view']:
                    keyboard_actions['reset_view'] = False
                    view_control.convert_from_pinhole_camera_parameters(initial_view_params)
                    print("  视角已重置")
                
                vis.update_renderer()
                time.sleep(0.01)  # 避免CPU占用过高
            
            # 停止键盘监听器
            if keyboard_listener:
                keyboard_listener.stop()
            
            vis.destroy_window()
            
            # 询问是否继续
            if i < len(ply_files) - 1:
                try:
                    response = input(f"\n是否继续查看下一个点云？(y/n，默认y): ").strip().lower()
                    if response == 'n':
                        print("退出交互式可视化")
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\n退出交互式可视化")
                    break
        
        except Exception as e:
            print(f"Error visualizing {ply_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n交互式可视化完成！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从多视角RGBD数据生成带语义的点云")
    parser.add_argument("--scene_dir", type=str, required=True,
                       help="场景目录路径（包含多个cam_*文件夹）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录路径（默认：scene_dir/POINTCLOUDS_MULTIVIEW）")
    parser.add_argument("--num_frames", type=int, default=None,
                       help="处理的帧数（None表示全部）")
    parser.add_argument("--downsample_factor", type=float, default=1.0,
                       help="图像下采样因子（默认1.0）")
    parser.add_argument("--min_depth", type=float, default=0.3,
                       help="最小有效深度（米，默认0.3）")
    parser.add_argument("--max_depth", type=float, default=1.0,
                       help="最大有效深度（米，默认1.0）")
    parser.add_argument("--visualize", action="store_true",
                       help="进行交互式可视化")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="可视化的样本数量（默认10）")
    
    args = parser.parse_args()
    
    # 生成点云
    process_multiview_pointcloud(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        downsample_factor=args.downsample_factor,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth
    )
    
    # 可视化
    if args.visualize:
        visualize_pointclouds_interactive(args.scene_dir, num_samples=args.num_samples)

