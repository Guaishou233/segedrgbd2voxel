#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从RGBD数据生成带语义的点云数据（世界坐标系）
"""

import json
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import struct
try:
    import open3d as o3d
    USE_OPEN3D = True
except ImportError:
    USE_OPEN3D = False
    print("Warning: open3d未安装，将使用手动方法生成点云")

# 检查pycocotools是否安装（用于解码COCO格式的RLE mask）
try:
    import pycocotools.mask as mask_util
    USE_PYCOCOTOOLS = True
except ImportError:
    USE_PYCOCOTOOLS = False
    print("Warning: pycocotools未安装，无法解码COCO格式的RLE mask")
    print("  这将导致语义标注无法正确应用到点云中")
    print("  请安装: pip install pycocotools")

def clean_json_data(obj):
    """
    清理JSON数据，将bytes对象转换为字符串，以便JSON序列化
    
    Args:
        obj: 要清理的对象（可以是dict, list, 或其他类型）
    
    Returns:
        清理后的对象
    """
    if isinstance(obj, dict):
        return {key: clean_json_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    elif isinstance(obj, bytes):
        # 将bytes转换为字符串（假设是UTF-8编码）
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            # 如果无法解码为UTF-8，使用base64编码
            import base64
            return base64.b64encode(obj).decode('ascii')
    elif isinstance(obj, (np.integer, np.floating)):
        # 将numpy数值类型转换为Python原生类型
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # 将numpy数组转换为列表
        return obj.tolist()
    else:
        return obj

def load_depth_image(depth_path, depth_scale=1000.0):
    """
    加载深度图，返回深度值（单位：米）
    
    Args:
        depth_path: 深度图路径
        depth_scale: 深度缩放因子（默认1000，表示深度以毫米存储）
    
    Returns:
        depth_meters: 深度值（米）
    """
    # 使用cv2读取，保持原始数据类型
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    # 转换为米
    depth_meters = depth / depth_scale
    depth_meters[depth == 0] = 0
    
    return depth_meters

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

def create_semantic_mask_from_annotations(annotations, image_id, height, width, categories):
    """
    从COCO格式的annotations创建语义分割mask
    
    Args:
        annotations: 所有annotations列表
        image_id: 图像ID
        height: 图像高度
        width: 图像宽度
        categories: categories列表，用于映射category_id到类别
    
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
                    # 解码失败，可能是pycocotools未安装
                    if not USE_PYCOCOTOOLS:
                        print(f"Warning: Failed to decode annotation {ann.get('id')}: pycocotools未安装，无法解码RLE mask")
                    else:
                        print(f"Warning: Failed to decode annotation {ann.get('id')}: RLE解码失败")
                    continue
            elif isinstance(segmentation, list):
                # 多边形格式（如果有）
                if not USE_PYCOCOTOOLS:
                    print(f"Warning: Failed to decode annotation {ann.get('id')}: pycocotools未安装，无法解码多边形格式")
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
            print(f"Warning: Failed to decode annotation {ann.get('id')}: {e}")
            continue
    
    # 设置robot arm区域为类别1
    semantic_mask[robot_arm_mask] = 1
    
    return semantic_mask

def pixel_to_camera_coords(u, v, depth, intrinsic):
    """
    将像素坐标转换为相机坐标系
    
    Args:
        u, v: 像素坐标
        depth: 深度值（米）
        intrinsic: 相机内参矩阵 3x3
    
    Returns:
        x, y, z: 相机坐标系下的坐标
    """
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return x, y, z

def camera_to_world_coords(points_camera, extrinsic):
    """
    将相机坐标系下的点转换为世界坐标系
    
    Args:
        points_camera: Nx3 数组，相机坐标系下的点
        extrinsic: 4x4 外参矩阵（相机到世界）
    
    Returns:
        points_world: Nx3 数组，世界坐标系下的点
    """
    # 提取旋转矩阵和平移向量
    R = np.array(extrinsic[:3, :3])
    t = np.array(extrinsic[:3, 3])
    
    # 转换：P_world = R * P_camera + t
    points_world = (R @ points_camera.T).T + t
    
    return points_world

def generate_pointcloud(rgb_path, depth_path, annotations, image_id, intrinsic, extrinsic, 
                       categories, depth_scale=1000.0, max_depth=1.0, min_depth=0.3, 
                       downsample_factor=1.0, use_open3d=True):
    """
    生成带语义的点云（使用Open3D或手动方法）
    
    Args:
        rgb_path: RGB图像路径
        depth_path: 深度图路径
        annotations: COCO格式的annotations列表
        image_id: 图像ID
        intrinsic: 相机内参矩阵 3x3
        extrinsic: 相机外参矩阵 4x4
        categories: categories列表
        depth_scale: 深度缩放因子（默认1000，表示深度以毫米存储）
        max_depth: 最大有效深度（米）
        min_depth: 最小有效深度（米）
        downsample_factor: 下采样因子（默认1.0，不采样）
        use_open3d: 是否使用Open3D生成点云（推荐）
    
    Returns:
        points: Nx3 点云坐标（世界坐标系）
        colors: Nx3 RGB颜色
        semantics: Nx1 语义标签（1=robot arm, 2=others）
    """
    # 转换为numpy数组
    intrinsic = np.array(intrinsic)
    extrinsic = np.array(extrinsic)
    
    # 使用Open3D方法（推荐）
    if use_open3d and USE_OPEN3D:
        return generate_pointcloud_open3d(
            rgb_path, depth_path, annotations, image_id, 
            intrinsic, extrinsic, categories, 
            depth_scale, max_depth, min_depth, downsample_factor
        )
    else:
        # 回退到手动方法
        return generate_pointcloud_manual(
            rgb_path, depth_path, annotations, image_id, 
            intrinsic, extrinsic, categories, 
            depth_scale, max_depth, min_depth
        )

def generate_pointcloud_open3d(rgb_path, depth_path, annotations, image_id, 
                               intrinsic, extrinsic, categories,
                               depth_scale=1000.0, max_depth=1.0, min_depth=0.3,
                               downsample_factor=1.0):
    """
    使用Open3D生成点云（参考rh20t_api/utils/point_cloud.py）
    """
    # 读取RGB和深度图
    color = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    original_height, original_width = depth.shape
    height, width = original_height, original_width
    
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
    depth[depth < min_depth] = 0
    depth[depth > max_depth] = 0
    
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
    
    # 提取点云数据
    points_camera = np.asarray(pcd.points)
    colors_o3d = np.asarray(pcd.colors) * 255.0  # Open3D颜色范围是0-1，转换为0-255
    
    if len(points_camera) == 0:
        return None, None, None
    
    # 从annotations创建语义mask（基于原始图像尺寸）
    semantic_mask = create_semantic_mask_from_annotations(
        annotations, image_id, original_height, original_width, categories
    )
    
    # 如果图像被下采样了，需要下采样mask
    if downsample_factor != 1.0:
        semantic_mask = cv2.resize(
            semantic_mask.astype(np.uint8), 
            (width, height), 
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        mask_width, mask_height = width, height
    else:
        # 如果downsample_factor=1.0，检查内参是否被缩放
        # 如果内参被缩放了0.5，投影坐标的范围会相应调整
        # 我们需要将mask缩放到与投影坐标匹配的尺寸
        # 由于内参fx被缩放了0.5，投影坐标u的范围会缩小，需要相应缩放mask
        # 但更简单的方法是：保持mask为原始尺寸，然后调整投影坐标
        mask_width, mask_height = original_width, original_height
    
    # 将点云坐标投影回像素坐标，以获取对应的语义标签
    # 注意：Open3D使用的内参是缩放后的（fx, fy, cx, cy），投影坐标基于当前图像尺寸
    # 如果mask是原始尺寸，需要将投影坐标映射到原始尺寸
    semantics = []
    projection_errors = 0
    out_of_bounds = 0
    robot_arm_points_found = 0
    
    # 计算坐标缩放因子（如果mask尺寸与投影尺寸不同）
    coord_scale_x = mask_width / width if width > 0 and mask_width != width else 1.0
    coord_scale_y = mask_height / height if height > 0 and mask_height != height else 1.0
    
    for i, point in enumerate(points_camera):
        # 投影到像素坐标（使用Open3D实际使用的内参）
        z = point[2]
        if z <= 0:
            semantics.append(2)  # others
            continue
        
        u = int(fx * point[0] / z + cx)
        v = int(fy * point[1] / z + cy)
        
        # 如果mask尺寸与投影尺寸不同，需要缩放坐标
        if coord_scale_x != 1.0 or coord_scale_y != 1.0:
            u = int(u * coord_scale_x)
            v = int(v * coord_scale_y)
        
        # 检查边界
        if 0 <= u < mask_width and 0 <= v < mask_height:
            try:
                semantic_label = semantic_mask[v, u]
                semantics.append(semantic_label)
                if semantic_label == 1:
                    robot_arm_points_found += 1
            except IndexError:
                # 如果索引越界，说明mask尺寸不匹配
                projection_errors += 1
                semantics.append(2)  # others
        else:
            out_of_bounds += 1
            semantics.append(2)  # others
    
    semantics = np.array(semantics)
    colors = colors_o3d.astype(np.uint8)
    
    semantics = np.array(semantics)
    colors = colors_o3d.astype(np.uint8)
    
    # 点云已经在世界坐标系中（因为extrinsic已经传入）
    return points_camera, colors, semantics

def generate_pointcloud_manual(rgb_path, depth_path, annotations, image_id, 
                                intrinsic, extrinsic, categories,
                                depth_scale=1000.0, max_depth=1.0, min_depth=0.3):
    """
    手动方法生成点云（回退方案）
    """
    # 加载图像
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_array = np.array(rgb_img)
    
    depth_array = load_depth_image(depth_path, depth_scale)
    height, width = depth_array.shape
    
    # 从annotations创建语义mask
    semantic_mask = create_semantic_mask_from_annotations(annotations, image_id, height, width, categories)
    
    # 检查并调整内参：如果图像被resize了，需要相应缩放内参
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    expected_center_x = width / 2.0
    expected_center_y = height / 2.0
    
    # 如果内参中心点远大于图像中心，说明内参是针对更大尺寸的
    if (abs(cx - 640) < 50 and abs(cy - 360) < 50) and (abs(cx - expected_center_x) > width * 0.2):
        scale = 0.5
        intrinsic_scaled = intrinsic.copy()
        intrinsic_scaled[0, 0] *= scale
        intrinsic_scaled[1, 1] *= scale
        intrinsic_scaled[0, 2] *= scale
        intrinsic_scaled[1, 2] *= scale
        intrinsic = intrinsic_scaled
    
    # 收集点云数据
    points_camera = []
    colors = []
    semantics = []
    
    step = 1
    
    for v in range(0, height, step):
        for u in range(0, width, step):
            depth = depth_array[v, u]
            
            # 过滤无效深度
            if depth <= 0 or depth < min_depth or depth > max_depth:
                continue
            
            # 像素坐标转相机坐标
            x_cam, y_cam, z_cam = pixel_to_camera_coords(u, v, depth, intrinsic)
            
            # 过滤异常点
            if not (np.isfinite(x_cam) and np.isfinite(y_cam) and np.isfinite(z_cam)):
                continue
            
            points_camera.append([x_cam, y_cam, z_cam])
            colors.append(rgb_array[v, u])
            semantics.append(semantic_mask[v, u])
    
    if len(points_camera) == 0:
        return None, None, None
    
    points_camera = np.array(points_camera)
    colors = np.array(colors)
    semantics = np.array(semantics)
    
    # 相机坐标转世界坐标
    points_world = camera_to_world_coords(points_camera, extrinsic)
    
    return points_world, colors, semantics

def save_pointcloud_ply(points, colors, semantics, output_path):
    """
    保存点云为PLY格式
    
    Args:
        points: Nx3 点云坐标
        colors: Nx3 RGB颜色 (0-255)
        semantics: Nx1 语义标签
        output_path: 输出文件路径
    """
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
            f.write(struct.pack('<B', int(semantics[i])))
    
    return True

def process_dataset(meta_json_path, dataset_dir, num_samples=None, visualize=True, interactive=False):
    """
    处理数据集，生成点云
    
    Args:
        meta_json_path: meta_info.json路径
        dataset_dir: 数据集目录
        num_samples: 处理的样本数量（None表示全部）
        visualize: 是否可视化
        interactive: 是否使用交互式可视化（True=交互式，False=保存图片）
    """
    print("正在读取meta_info.json...")
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建点云输出目录
    pointcloud_dir = os.path.join(dataset_dir, "POINTCLOUDS")
    os.makedirs(pointcloud_dir, exist_ok=True)
    
    images = coco_data.get("images", [])
    
    # 限制处理数量
    if num_samples is not None:
        images = images[:num_samples]
        print(f"处理前 {num_samples} 个样本...")
    else:
        print(f"处理全部 {len(images)} 个样本...")
    
    # 路径设置
    rgb_dir = os.path.join(dataset_dir, "RGB")
    depth_dir = os.path.join(dataset_dir, "DEPTH")
    
    processed_count = 0
    failed_count = 0
    
    for img_info in tqdm(images, desc="生成点云"):
        image_id = img_info["id"]
        rgb_filename = img_info["file_name"]
        depth_filename = img_info.get("depth_image", "")
        
        if not depth_filename:
            print(f"Warning: 图像 {image_id} 没有深度图信息")
            failed_count += 1
            continue
        
        # 构建路径
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        depth_path = os.path.join(depth_dir, depth_filename)
        
        # 检查文件是否存在
        if not os.path.exists(rgb_path):
            print(f"Warning: RGB图像不存在: {rgb_path}")
            failed_count += 1
            continue
        if not os.path.exists(depth_path):
            print(f"Warning: 深度图不存在: {depth_path}")
            failed_count += 1
            continue
        
        try:
            # 获取相机参数
            intrinsic = img_info.get("intrinsic", [])
            extrinsic = img_info.get("extrinsic", [])
            
            if not intrinsic or not extrinsic:
                print(f"Warning: 图像 {image_id} 缺少相机参数")
                failed_count += 1
                continue
            
            # 获取annotations和categories
            all_annotations = coco_data.get("annotations", [])
            categories = coco_data.get("categories", [])
            
            # 检查该图像是否有annotations
            image_annotations = [ann for ann in all_annotations if ann.get('image_id') == image_id]
            if len(image_annotations) > 0:
                robot_arm_annotations = [ann for ann in image_annotations if ann.get('category_id') == 1]
                robot_arm_count = len(robot_arm_annotations)
            else:
                robot_arm_count = 0
            
            # 生成点云（使用Open3D方法，参考rh20t_api）
            points, colors, semantics = generate_pointcloud(
                rgb_path, depth_path, all_annotations, image_id, 
                intrinsic, extrinsic, categories,
                depth_scale=1000.0,
                max_depth=1.0,
                min_depth=0.3,
                downsample_factor=1.0,
                use_open3d=True
            )
            
            if points is None or len(points) == 0:
                print(f"Warning: 图像 {image_id} 未能生成有效点云")
                failed_count += 1
                continue
            
            # 验证语义信息
            actual_robot_arm = np.sum(semantics == 1)
            if robot_arm_count > 0 and actual_robot_arm == 0:
                # 诊断问题原因
                diagnostic_msg = f"Warning: 图像 {image_id} 有 {robot_arm_count} 个robot arm annotations，但点云中为0"
                if not USE_PYCOCOTOOLS:
                    diagnostic_msg += " (可能原因: pycocotools未安装，annotation无法解码)"
                else:
                    # 检查深度图在robot arm区域是否有有效深度
                    try:
                        semantic_mask = create_semantic_mask_from_annotations(
                            all_annotations, image_id, 
                            img_info.get('height', 360), 
                            img_info.get('width', 640), 
                            categories
                        )
                        robot_arm_pixels = np.sum(semantic_mask == 1)
                        if robot_arm_pixels > 0:
                            # 检查深度图
                            depth_array = load_depth_image(depth_path, depth_scale=1000.0)
                            robot_arm_depth = depth_array[semantic_mask == 1]
                            valid_depth_count = np.sum((robot_arm_depth > 0.3) & (robot_arm_depth < 1.0))
                            if valid_depth_count == 0:
                                diagnostic_msg += f" (可能原因: robot arm区域({robot_arm_pixels}像素)的深度值无效或超出范围[0.3m, 1.0m])"
                            else:
                                # 深度有效，但点云中没有robot arm点，可能的原因：
                                # 1. Open3D在创建点云时过滤了这些点（NaN/Inf/其他过滤逻辑）
                                # 2. 投影坐标计算错误，导致无法正确映射到semantic_mask
                                # 3. 内参缩放导致坐标不匹配
                                diagnostic_msg += f" (可能原因: 深度有效({valid_depth_count}/{robot_arm_pixels}像素)但点云中无robot arm点。"
                                diagnostic_msg += " 可能原因：1) Open3D过滤了这些点 2) 投影坐标计算错误 3) 内参缩放导致坐标不匹配)"
                        else:
                            diagnostic_msg += " (可能原因: annotation解码失败，语义mask中无robot arm区域)"
                    except Exception as e:
                        diagnostic_msg += f" (诊断失败: {e})"
                print(diagnostic_msg)
            
            # 保存点云
            pointcloud_filename = rgb_filename.replace(".jpg", ".ply").replace(".jpeg", ".ply")
            pointcloud_path = os.path.join(pointcloud_dir, pointcloud_filename)
            save_pointcloud_ply(points, colors, semantics, pointcloud_path)
            
            # 更新meta_info中的点云路径
            img_info["pointcloud"] = pointcloud_filename
            img_info["pointcloud_path"] = f"POINTCLOUDS/{pointcloud_filename}"
            
            processed_count += 1
        
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    # 保存更新后的meta_info.json
    print(f"\n正在更新 {meta_json_path}...")
    # 清理数据中的bytes对象，确保可以JSON序列化
    coco_data_clean = clean_json_data(coco_data)
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data_clean, f, indent=2, ensure_ascii=False)
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 个样本")
    print(f"失败: {failed_count} 个样本")
    print(f"点云保存在: {pointcloud_dir}")
    
    # 可视化
    if visualize and processed_count > 0:
        if interactive:
            print("\n开始交互式可视化...")
            visualize_pointclouds_interactive(dataset_dir, meta_json_path, min(10, processed_count))
        else:
            print("\n开始可视化（保存图片）...")
            visualize_pointclouds(dataset_dir, meta_json_path, min(10, processed_count))

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

def get_camera_view_params(extrinsic):
    """
    从相机外参计算相机视角参数
    
    Args:
        extrinsic: 4x4 外参矩阵（相机到世界）
    
    Returns:
        camera_pos: 相机在世界坐标系中的位置 [x, y, z]
        camera_front: 相机朝向（Z轴方向，看向场景）[x, y, z]
        camera_up: 相机上方向（Y轴方向）[x, y, z]
    """
    extrinsic = np.array(extrinsic)
    
    # 外参是从相机坐标系到世界坐标系的变换：P_world = R * P_camera + t
    # 相机在世界坐标系中的位置：将相机坐标系原点[0,0,0]转换到世界坐标系
    # P_world = R * [0,0,0] + t = t
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # 相机在世界坐标系中的位置就是t（相机坐标系原点在世界坐标系中的位置）
    camera_pos = t.copy()
    
    # 相机朝向：相机Z轴在世界坐标系中的方向（相机看向的方向）
    # 在相机坐标系中，Z轴指向场景前方，转换到世界坐标系
    camera_z_world = R @ np.array([0, 0, 1])
    camera_front = camera_z_world / (np.linalg.norm(camera_z_world) + 1e-8)
    
    # 相机上方向：相机Y轴在世界坐标系中的方向（通常向下，因为图像坐标系Y向下）
    camera_y_world = R @ np.array([0, 1, 0])
    camera_up = camera_y_world / (np.linalg.norm(camera_y_world) + 1e-8)
    
    return camera_pos, camera_front, camera_up

def calculate_view_angles(camera_pos, lookat_point):
    """
    计算matplotlib 3D视角的elev和azim角度（从相机位置看向目标点）
    
    Args:
        camera_pos: 相机位置 [x, y, z]
        lookat_point: 看向的点（通常是点云中心）[x, y, z]
    
    Returns:
        elev: 仰角（度）
        azim: 方位角（度）
    """
    # 计算从相机到目标点的方向向量
    view_dir = lookat_point - camera_pos
    dist = np.linalg.norm(view_dir)
    
    if dist < 1e-8:
        return 30, 45  # 默认角度
    
    view_dir = view_dir / dist
    
    # 计算仰角（elevation）：Z轴方向的角度
    # elev = arcsin(z_component) * 180 / pi
    elev = np.arcsin(np.clip(view_dir[2], -1, 1)) * 180 / np.pi
    
    # 计算方位角（azimuth）：XY平面上的角度
    # azim = arctan2(y, x) * 180 / pi
    azim = np.arctan2(view_dir[1], view_dir[0]) * 180 / np.pi
    
    return elev, azim

def calculate_camera_view_angle_for_3d(extrinsic):
    """
    根据相机外参矩阵计算相机坐标系下的3D视图角度
    使3D视图与RGB图像对齐
    
    在相机坐标系中：
    - 相机在原点(0,0,0)
    - Z轴向前（相机看向的方向）
    - X轴向右
    - Y轴向下
    
    matplotlib的3D视图默认坐标系：
    - X轴指向右侧
    - Y轴指向前方（在elev=0, azim=0时）
    - Z轴指向上方
    
    我们需要调整elev和azim，使得matplotlib的视图与相机坐标系对齐：
    - X轴指向右侧（图像宽度方向）
    - Y轴指向下方（图像高度方向）
    - Z轴指向前方（深度方向）
    
    关键思路：
    由于点云已经在相机坐标系中，我们需要根据相机在世界坐标系中的旋转，
    计算合适的elev和azim，使得从相机视角看过去，点云和RGB图像对齐。
    
    实际上，在相机坐标系中，标准的视角应该是：
    - elev=0（水平看向前方）
    - azim=0（正面视角）
    但是，如果相机在世界坐标系中有旋转，我们需要调整azim来匹配相机的实际朝向。
    
    更准确的方法：
    1. 相机坐标系的X轴（右方向）在世界坐标系中的方向，决定了我们需要如何旋转视图
    2. 相机坐标系的Y轴（下方向）在世界坐标系中的方向，决定了我们需要如何调整elev
    
    Args:
        extrinsic: 4x4 外参矩阵（世界坐标系到相机坐标系）
    
    Returns:
        elev: 仰角（度）
        azim: 方位角（度）
    """
    R = np.array(extrinsic[:3, :3])
    
    # 相机坐标系的轴方向（在世界坐标系中）
    # R的列向量是相机坐标系的轴在世界坐标系中的方向
    x_axis_world = R[:, 0]  # 相机X轴（右方向）在世界坐标系中的方向
    y_axis_world = R[:, 1]  # 相机Y轴（下方向）在世界坐标系中的方向
    z_axis_world = R[:, 2]  # 相机Z轴（前方向）在世界坐标系中的方向
    
    # 在相机坐标系中，我们总是从原点看向Z轴正方向
    # 但是matplotlib的默认视角需要调整，使得X轴指向右侧，Y轴指向下方
    
    # 计算azim：根据相机X轴（右方向）在XY平面上的投影
    # 我们需要使matplotlib的X轴与相机X轴对齐
    x_proj_xy = x_axis_world[:2]
    x_proj_xy_norm = np.linalg.norm(x_proj_xy)
    
    if x_proj_xy_norm > 1e-8:
        # 计算相机X轴在XY平面上的角度
        x_angle_xy = np.arctan2(x_axis_world[1], x_axis_world[0]) * 180 / np.pi
        # matplotlib的默认X轴方向是azim=0时的方向（指向右侧）
        # 我们需要旋转，使相机X轴指向右侧
        # 如果相机X轴已经指向右侧（角度接近0），azim=0
        # 否则需要调整azim，使得相机X轴指向右侧
        azim = -x_angle_xy
    else:
        # X轴垂直，使用Y轴来计算
        y_proj_xy = y_axis_world[:2]
        y_proj_xy_norm = np.linalg.norm(y_proj_xy)
        if y_proj_xy_norm > 1e-8:
            y_angle_xy = np.arctan2(y_axis_world[1], y_axis_world[0]) * 180 / np.pi
            azim = -y_angle_xy + 90
        else:
            azim = 0
    
    # 计算elev：根据相机Y轴（下方向）的Z分量
    # 相机Y轴在世界坐标系中的Z分量，反映了相机的上下倾斜
    # 如果Y轴的Z分量为正，说明相机向下倾斜，我们需要向上看（elev为正）
    # 如果Y轴的Z分量为负，说明相机向上倾斜，我们需要向下看（elev为负）
    y_z_component = y_axis_world[2]
    if abs(y_z_component) > 1e-8:
        # 计算Y轴与XY平面的夹角
        y_norm = np.linalg.norm(y_axis_world)
        if y_norm > 1e-8:
            elev = np.arcsin(np.clip(y_z_component / y_norm, -1, 1)) * 180 / np.pi
        else:
            elev = 0
    else:
        elev = 0
    
    return elev, azim

def draw_camera_frame(ax, extrinsic, scale=0.1, linewidth=2):
    """
    在3D轴上绘制相机坐标系（位置和方向）
    
    Args:
        ax: matplotlib 3D轴
        extrinsic: 相机外参矩阵 4x4
        scale: 坐标系箭头长度（米）
        linewidth: 箭头线宽
    """
    R = np.array(extrinsic[:3, :3])
    t = np.array(extrinsic[:3, 3])
    
    # 相机位置
    camera_pos = t
    
    # 相机坐标系的方向向量（在世界坐标系中）
    # 相机坐标系的X, Y, Z轴在世界坐标系中的方向
    x_axis = R[:, 0]  # 相机右方向
    y_axis = R[:, 1]  # 相机下方向
    z_axis = -R[:, 2]  # 相机前方向（OpenCV约定：Z轴向前）
    
    # 绘制相机位置（大点）
    ax.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], 
              c='blue', s=100, marker='o', label='Camera', zorder=10)
    
    # 绘制三个轴（用箭头）
    # X轴（红色，右方向）
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
             x_axis[0] * scale, x_axis[1] * scale, x_axis[2] * scale,
             color='red', arrow_length_ratio=0.3, linewidth=linewidth, label='Camera X (Right)')
    
    # Y轴（绿色，下方向）
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
             y_axis[0] * scale, y_axis[1] * scale, y_axis[2] * scale,
             color='green', arrow_length_ratio=0.3, linewidth=linewidth, label='Camera Y (Down)')
    
    # Z轴（蓝色，前方向）
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
             z_axis[0] * scale, z_axis[1] * scale, z_axis[2] * scale,
             color='blue', arrow_length_ratio=0.3, linewidth=linewidth, label='Camera Z (Forward)')
    
    return camera_pos

def draw_camera_position_2d(ax, extrinsic, color='blue', marker='o', size=50, label='Camera'):
    """
    在2D轴上绘制相机位置
    
    Args:
        ax: matplotlib 2D轴
        extrinsic: 相机外参矩阵 4x4
        color: 标记颜色
        marker: 标记样式
        size: 标记大小
        label: 图例标签
    """
    R = np.array(extrinsic[:3, :3])
    t = np.array(extrinsic[:3, 3])
    camera_pos = t
    
    # 根据当前轴的类型绘制
    if ax.name == 'rectilinear':  # 2D轴
        # 判断是XY、XZ还是YZ投影
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        
        if 'X' in xlabel and 'Y' in ylabel:
            # XY投影
            ax.scatter([camera_pos[0]], [camera_pos[1]], 
                      c=color, s=size, marker=marker, label=label, zorder=10)
        elif 'X' in xlabel and 'Z' in ylabel:
            # XZ投影
            ax.scatter([camera_pos[0]], [camera_pos[2]], 
                      c=color, s=size, marker=marker, label=label, zorder=10)
        elif 'Y' in xlabel and 'Z' in ylabel:
            # YZ投影
            ax.scatter([camera_pos[1]], [camera_pos[2]], 
                      c=color, s=size, marker=marker, label=label, zorder=10)

def project_pointcloud_to_image(points_world, colors, semantics, intrinsic, extrinsic, 
                                 image_width, image_height):
    """
    将世界坐标系下的点云投影到2D图像平面（相机视角）
    
    Args:
        points_world: Nx3 点云坐标（世界坐标系）
        colors: Nx3 RGB颜色
        semantics: Nx1 语义标签
        intrinsic: 相机内参矩阵 3x3
        extrinsic: 相机外参矩阵 4x4
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        projected_image: 投影后的图像 (HxWx3)
        depth_map: 深度图 (HxW)
    """
    # 世界坐标转相机坐标
    R = np.array(extrinsic[:3, :3])
    t = np.array(extrinsic[:3, 3])
    
    # P_camera = R^T * (P_world - t)
    points_camera = ((R.T @ (points_world - t).T).T)
    
    # 提取内参
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # 创建投影图像和深度图
    projected_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    depth_map = np.zeros((image_height, image_width), dtype=np.float32)
    
    # 投影每个点到图像平面
    for i, point_cam in enumerate(points_camera):
        x, y, z = point_cam
        
        # 过滤相机后方的点
        if z <= 0:
            continue
        
        # 投影到像素坐标
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)
        
        # 检查边界
        if 0 <= u < image_width and 0 <= v < image_height:
            # 使用深度测试，只保留最近的点
            if depth_map[v, u] == 0 or z < depth_map[v, u]:
                depth_map[v, u] = z
                # 根据语义标签设置颜色（点云统一使用橙色）
                projected_image[v, u] = [255, 165, 0]  # 橙色
    
    return projected_image, depth_map

def visualize_pointclouds_matplotlib(dataset_dir, meta_json_path, num_samples=10):
    """
    使用matplotlib可视化点云数据并保存为图片（支持无GUI环境，使用相机视角）
    包括2D投影视图和3D视图
    
    Args:
        dataset_dir: 数据集目录
        meta_json_path: meta_info.json路径
        num_samples: 可视化的样本数量
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Warning: matplotlib未安装，无法可视化。请安装: pip install matplotlib")
        return
    
    # 读取meta_info以获取相机参数
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    images_dict = {img['id']: img for img in coco_data.get('images', [])}
    
    pointcloud_dir = os.path.join(dataset_dir, "POINTCLOUDS")
    visualization_dir = os.path.join(dataset_dir, "VISUALIZATIONS")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 获取点云文件列表，优先选择有robot arm的点云
    ply_files = [f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')]
    
    # 检查每个点云是否有robot arm，优先可视化有robot arm的
    ply_files_with_robot_arm = []
    ply_files_without_robot_arm = []
    
    print("正在检查点云文件，优先选择包含robot arm的点云...")
    for ply_file in ply_files:
        ply_path = os.path.join(pointcloud_dir, ply_file)
        try:
            points, colors, semantics = load_pointcloud_with_semantics(ply_path)
            if np.sum(semantics == 1) > 0:
                ply_files_with_robot_arm.append(ply_file)
            else:
                ply_files_without_robot_arm.append(ply_file)
        except:
            ply_files_without_robot_arm.append(ply_file)
    
    # 优先选择有robot arm的点云
    if len(ply_files_with_robot_arm) > 0:
        ply_files = ply_files_with_robot_arm[:num_samples]
        if len(ply_files) < num_samples and len(ply_files_without_robot_arm) > 0:
            # 如果不够，补充一些没有robot arm的
            remaining = num_samples - len(ply_files)
            ply_files.extend(ply_files_without_robot_arm[:remaining])
        print(f"找到 {len(ply_files_with_robot_arm)} 个包含robot arm的点云，选择前 {min(num_samples, len(ply_files_with_robot_arm))} 个进行可视化")
    else:
        ply_files = sorted(ply_files)[:num_samples]
        print(f"未找到包含robot arm的点云，可视化前 {num_samples} 个点云")
    
    if len(ply_files) == 0:
        print("没有找到点云文件")
        return
    
    print(f"\n生成 {len(ply_files)} 个点云的可视化图片...")
    print("提示：")
    print("  - 红色点 = robot arm (类别1)")
    print("  - 灰色点 = others (类别2)")
    
    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(pointcloud_dir, ply_file)
        
        try:
            # 加载点云（包含语义信息）
            points, colors, semantics = load_pointcloud_with_semantics(ply_path)
            
            if len(points) == 0:
                print(f"Warning: {ply_file} 是空点云")
                continue
            
            # 统计信息
            robot_arm_count = np.sum(semantics == 1)
            others_count = np.sum(semantics == 2)
            
            print(f"\n点云 {i+1}/{len(ply_files)}: {ply_file}")
            print(f"  总点数: {len(points):,}")
            print(f"  Robot arm点数: {robot_arm_count:,} ({robot_arm_count/len(points)*100:.1f}%)")
            print(f"  Others点数: {others_count:,} ({others_count/len(points)*100:.1f}%)")
            
            # 下采样以加快可视化（如果点太多）
            max_points = 50000
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
                colors = colors[indices]
                semantics = semantics[indices]
            
            # 获取对应的图像信息和相机参数
            img_info = None
            for img in coco_data.get('images', []):
                expected_pc = img['file_name'].replace('.jpg', '.ply').replace('.jpeg', '.ply')
                if expected_pc == ply_file:
                    img_info = img
                    break
            
            # 获取原始RGB图像路径（用于叠加显示）
            rgb_dir = os.path.join(dataset_dir, "RGB")
            rgb_filename = ply_file.replace('.ply', '.jpg').replace('.ply', '.jpeg')
            rgb_path = os.path.join(rgb_dir, rgb_filename)
            
            # 将点云转换到相机坐标系（用于相机视角可视化）
            points_camera_coords = None
            view_elev = 0
            view_azim = 0
            if img_info and img_info.get('extrinsic'):
                try:
                    extrinsic = np.array(img_info['extrinsic'])
                    R = extrinsic[:3, :3]
                    t = extrinsic[:3, 3]
                    # 世界坐标转相机坐标：P_camera = R^T * (P_world - t)
                    points_camera_coords = ((R.T @ (points - t).T).T)
                    # 计算相机视角角度，使3D视图与RGB图像对齐
                    view_elev, view_azim = calculate_camera_view_angle_for_3d(extrinsic)
                except Exception as e:
                    print(f"  Warning: 无法转换到相机坐标系: {e}")
            
            # 创建可视化
            fig = plt.figure(figsize=(20, 12))
            
            # 如果有相机参数，生成2D投影视图（相机视角）
            if img_info and img_info.get('intrinsic') and img_info.get('extrinsic'):
                try:
                    intrinsic = np.array(img_info['intrinsic'])
                    extrinsic = np.array(img_info['extrinsic'])
                    image_width = img_info.get('width', 640)
                    image_height = img_info.get('height', 360)
                    
                    # 投影点云到图像平面
                    projected_img, depth_map = project_pointcloud_to_image(
                        points, colors, semantics, intrinsic, extrinsic,
                        image_width, image_height
                    )
                    
                    # 显示投影图像（相机视角）
                    ax1 = fig.add_subplot(231)
                    ax1.imshow(projected_img)
                    ax1.set_title('Point Cloud Projection (Camera View)\nRed=Robot Arm, Color=Others')
                    ax1.axis('off')
                    
                    # 显示深度图
                    ax2 = fig.add_subplot(232)
                    depth_vis = depth_map.copy()
                    depth_vis[depth_vis == 0] = np.nan
                    im = ax2.imshow(depth_vis, cmap='jet', interpolation='nearest')
                    ax2.set_title('Depth Map (Camera View)')
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046)
                    
                    # 如果有原始RGB图像，叠加显示
                    if os.path.exists(rgb_path):
                        rgb_img = cv2.imread(rgb_path)
                        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                        # 创建叠加图像：RGB半透明，点云橙色
                        overlay = rgb_img.copy().astype(np.float32)
                        mask = (projected_img.sum(axis=2) > 0)
                        # RGB半透明（alpha=0.5），点云橙色（alpha=0.5）
                        overlay[mask] = (0.5 * overlay[mask] + 0.5 * projected_img[mask].astype(np.float32))
                        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                        
                        ax3 = fig.add_subplot(233)
                        ax3.imshow(overlay)
                        ax3.set_title('Overlay: RGB (50% transparent) + Point Cloud (Orange)')
                        ax3.axis('off')
                    else:
                        ax3 = fig.add_subplot(233)
                        ax3.text(0.5, 0.5, 'RGB image not found', 
                                ha='center', va='center', transform=ax3.transAxes)
                        ax3.axis('off')
                    
                except Exception as e:
                    print(f"  Warning: 无法生成2D投影视图: {e}")
                    # 如果投影失败，使用默认布局
                    ax1 = fig.add_subplot(231)
                    ax1.text(0.5, 0.5, 'Projection failed', 
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.axis('off')
                    ax2 = fig.add_subplot(232)
                    ax2.axis('off')
                    ax3 = fig.add_subplot(233)
                    ax3.axis('off')
            else:
                # 如果没有相机参数，留空
                ax1 = fig.add_subplot(231)
                ax1.text(0.5, 0.5, 'No camera parameters', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.axis('off')
                ax2 = fig.add_subplot(232)
                ax2.axis('off')
                ax3 = fig.add_subplot(233)
                ax3.axis('off')
            
            # 3D语义点云视图（相机坐标系视角）
            ax4 = fig.add_subplot(234, projection='3d')
            robot_arm_mask = semantics == 1
            others_mask = semantics == 2
            
            # 使用相机坐标系的点云进行可视化
            if points_camera_coords is not None:
                vis_points = points_camera_coords
            else:
                vis_points = points
            
            if np.any(robot_arm_mask):
                ax4.scatter(vis_points[robot_arm_mask, 0], 
                           vis_points[robot_arm_mask, 1], 
                           vis_points[robot_arm_mask, 2],
                           c='red', s=0.5, alpha=0.6, label='Robot Arm')
            
            if np.any(others_mask):
                ax4.scatter(vis_points[others_mask, 0], 
                           vis_points[others_mask, 1], 
                           vis_points[others_mask, 2],
                           c='gray', s=0.5, alpha=0.3, label='Others')
            
            # 标注相机位置（原点）和正前方方向（Z轴正方向）
            # 相机位置
            ax4.scatter([0], [0], [0], c='blue', s=200, marker='o', 
                       label='Camera Position', zorder=10, edgecolors='black', linewidths=2)
            
            # 相机正前方方向（Z轴正方向，箭头）
            forward_scale = 0.2  # 箭头长度（米）
            ax4.quiver(0, 0, 0, 0, 0, forward_scale,
                      color='blue', arrow_length_ratio=0.3, linewidth=3, 
                      label='Camera Forward (Z+)', zorder=10)
            
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_zlabel('Z (m)')
            ax4.set_title('3D Semantic Point Cloud (Camera Coordinate System)')
            # 从相机视角看：根据相机实际朝向和旋转设置视角
            ax4.view_init(elev=view_elev, azim=view_azim)
            ax4.legend(loc='upper left', fontsize=8)
            
            # 3D RGB点云视图（相机坐标系视角）
            ax5 = fig.add_subplot(235, projection='3d')
            ax5.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2],
                       c=colors / 255.0, s=0.5, alpha=0.6)
            
            # 标注相机位置（原点）和正前方方向（Z轴正方向）
            # 相机位置
            ax5.scatter([0], [0], [0], c='blue', s=200, marker='o', 
                       label='Camera Position', zorder=10, edgecolors='black', linewidths=2)
            
            # 相机正前方方向（Z轴正方向，箭头）
            forward_scale = 0.2  # 箭头长度（米）
            ax5.quiver(0, 0, 0, 0, 0, forward_scale,
                      color='blue', arrow_length_ratio=0.3, linewidth=3, 
                      label='Camera Forward (Z+)', zorder=10)
            
            ax5.set_xlabel('X (m)')
            ax5.set_ylabel('Y (m)')
            ax5.set_zlabel('Z (m)')
            ax5.set_title('3D RGB Point Cloud (Camera Coordinate System)')
            # 从相机视角看：根据相机实际朝向和旋转设置视角
            ax5.view_init(elev=view_elev, azim=view_azim)
            
            # 2D投影视图 - XY平面（相机坐标系，从上往下看）
            ax6 = fig.add_subplot(236)
            if np.any(robot_arm_mask):
                ax6.scatter(vis_points[robot_arm_mask, 0], vis_points[robot_arm_mask, 1],
                           c='red', s=1, alpha=0.6, label='Robot Arm')
            if np.any(others_mask):
                ax6.scatter(vis_points[others_mask, 0], vis_points[others_mask, 1],
                           c='gray', s=1, alpha=0.3, label='Others')
            
            # 相机在原点
            ax6.scatter([0], [0], c='blue', s=150, marker='o', 
                       label='Camera Position', zorder=10, edgecolors='black', linewidths=2)
            
            # 相机正前方方向（在XY平面上的投影，Z轴正方向指向屏幕外，用箭头表示）
            # 在XY投影中，Z轴正方向是垂直于屏幕向外的，我们用一个小箭头表示
            forward_scale_2d = 0.15  # 箭头长度（米）
            # 在XY投影中，Z轴正方向指向屏幕外，我们用一个小圆圈或箭头标记
            ax6.scatter([0], [0], c='blue', s=50, marker='^', 
                       label='Camera Forward (Z+)', zorder=11, edgecolors='blue', linewidths=1)
            
            ax6.set_xlabel('X (m)')
            ax6.set_ylabel('Y (m)')
            ax6.set_title('XY Projection (Top View, Camera Coordinates)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_aspect('equal')
            
            plt.tight_layout()
            
            # 保存图片
            output_image_path = os.path.join(visualization_dir, ply_file.replace('.ply', '_visualization.png'))
            plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  已保存可视化图片: {output_image_path}")
        
        except Exception as e:
            print(f"Error visualizing {ply_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n可视化完成！图片保存在: {visualization_dir}")

def load_pointcloud_ply_with_open3d(ply_path):
    """
    使用Open3D加载点云（包含语义信息）
    
    Args:
        ply_path: PLY文件路径
    
    Returns:
        pcd: Open3D点云对象
        semantics: 语义标签数组
    """
    if not USE_OPEN3D:
        raise ImportError("Open3D未安装，无法使用此功能")
    
    # 先手动读取语义信息（因为Open3D的PLY读取可能不支持自定义属性）
    points, colors, semantics = load_pointcloud_with_semantics(ply_path)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D颜色范围是0-1
    
    return pcd, semantics

def visualize_pointclouds_open3d(dataset_dir, meta_json_path, num_samples=10):
    """
    使用Open3D可视化点云数据并保存为图片（支持无GUI环境）
    
    Args:
        dataset_dir: 数据集目录
        meta_json_path: meta_info.json路径
        num_samples: 可视化的样本数量
    """
    if not USE_OPEN3D:
        print("Warning: Open3D未安装，无法使用Open3D可视化。请安装: conda install -c open3d-admin open3d")
        return
    
    # 读取meta_info以获取相机参数
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    pointcloud_dir = os.path.join(dataset_dir, "POINTCLOUDS")
    visualization_dir = os.path.join(dataset_dir, "VISUALIZATIONS")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 获取点云文件列表，优先选择有robot arm的点云
    ply_files = [f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')]
    
    # 检查每个点云是否有robot arm，优先可视化有robot arm的
    ply_files_with_robot_arm = []
    ply_files_without_robot_arm = []
    
    print("正在检查点云文件，优先选择包含robot arm的点云...")
    for ply_file in ply_files:
        ply_path = os.path.join(pointcloud_dir, ply_file)
        try:
            points, colors, semantics = load_pointcloud_with_semantics(ply_path)
            if np.sum(semantics == 1) > 0:
                ply_files_with_robot_arm.append(ply_file)
            else:
                ply_files_without_robot_arm.append(ply_file)
        except:
            ply_files_without_robot_arm.append(ply_file)
    
    # 优先选择有robot arm的点云
    if len(ply_files_with_robot_arm) > 0:
        ply_files = ply_files_with_robot_arm[:num_samples]
        if len(ply_files) < num_samples and len(ply_files_without_robot_arm) > 0:
            remaining = num_samples - len(ply_files)
            ply_files.extend(ply_files_without_robot_arm[:remaining])
        print(f"找到 {len(ply_files_with_robot_arm)} 个包含robot arm的点云，选择前 {min(num_samples, len(ply_files_with_robot_arm))} 个进行可视化")
    else:
        ply_files = sorted(ply_files)[:num_samples]
        print(f"未找到包含robot arm的点云，可视化前 {num_samples} 个点云")
    
    if len(ply_files) == 0:
        print("没有找到点云文件")
        return
    
    print(f"\n生成 {len(ply_files)} 个点云的可视化图片（使用Open3D）...")
    print("提示：")
    print("  - 红色点 = robot arm (类别1)")
    print("  - 灰色点 = others (类别2)")
    
    # 创建可视化器（离线渲染）
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    
    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(pointcloud_dir, ply_file)
        
        try:
            # 加载点云
            pcd, semantics = load_pointcloud_ply_with_open3d(ply_path)
            
            if len(pcd.points) == 0:
                print(f"Warning: {ply_file} 是空点云")
                continue
            
            # 统计信息
            robot_arm_count = np.sum(semantics == 1)
            others_count = np.sum(semantics == 2)
            
            print(f"\n点云 {i+1}/{len(ply_files)}: {ply_file}")
            print(f"  总点数: {len(pcd.points):,}")
            print(f"  Robot arm点数: {robot_arm_count:,} ({robot_arm_count/len(semantics)*100:.1f}%)")
            print(f"  Others点数: {others_count:,} ({others_count/len(semantics)*100:.1f}%)")
            
            # 下采样以加快可视化（如果点太多）
            max_points = 500000  # Open3D可以处理更多点
            if len(pcd.points) > max_points:
                pcd = pcd.random_down_sample(max_points / len(pcd.points))
                # 需要重新加载语义信息
                points, colors, semantics = load_pointcloud_with_semantics(ply_path)
                indices = np.random.choice(len(points), min(max_points, len(points)), replace=False)
                semantics = semantics[indices]
            
            # 创建语义颜色映射的点云
            semantic_colors = np.zeros((len(semantics), 3))
            robot_arm_mask = semantics == 1
            others_mask = semantics == 2
            semantic_colors[robot_arm_mask] = [1.0, 0.0, 0.0]  # 红色 - robot arm
            semantic_colors[others_mask] = [0.5, 0.5, 0.5]    # 灰色 - others
            
            # 创建语义点云
            pcd_semantic = o3d.geometry.PointCloud()
            pcd_semantic.points = pcd.points
            pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
            
            # 获取对应的图像信息和相机参数
            img_info = None
            for img in coco_data.get('images', []):
                expected_pc = img['file_name'].replace('.jpg', '.ply').replace('.jpeg', '.ply')
                if expected_pc == ply_file:
                    img_info = img
                    break
            
            # 设置相机视角
            view_control = vis.get_view_control()
            render_option = vis.get_render_option()
            
            # 设置渲染选项
            render_option.point_size = 1.0
            render_option.background_color = np.array([0.0, 0.0, 0.0])  # 黑色背景
            
            # 清除之前的几何体
            vis.clear_geometries()
            
            # 添加点云（使用语义颜色）
            vis.add_geometry(pcd_semantic)
            
            # 如果有相机参数，设置相机视角
            if img_info and img_info.get('extrinsic'):
                try:
                    extrinsic = np.array(img_info['extrinsic'])
                    R = extrinsic[:3, :3]
                    t = extrinsic[:3, 3]
                    
                    # 计算相机视角参数
                    # 相机位置
                    camera_pos = t
                    
                    # 相机朝向（Z轴方向）
                    camera_z = R @ np.array([0, 0, 1])
                    camera_front = camera_z / (np.linalg.norm(camera_z) + 1e-8)
                    
                    # 相机上方向（Y轴方向，取反因为Open3D的Y向上）
                    camera_y = R @ np.array([0, 1, 0])
                    camera_up = -camera_y / (np.linalg.norm(camera_y) + 1e-8)
                    
                    # 计算点云中心
                    points_array = np.asarray(pcd.points)
                    center = points_array.mean(axis=0)
                    
                    # 设置相机参数
                    # Open3D的lookat参数：从相机位置看向点云中心
                    lookat = center
                    up = camera_up
                    
                    # 设置视角
                    view_control.set_lookat(lookat)
                    view_control.set_up(up)
                    view_control.set_front(camera_front)
                    
                    # 设置相机距离（从相机位置到点云中心的距离）
                    distance = np.linalg.norm(camera_pos - center)
                    view_control.set_zoom(0.7)  # 调整缩放
                    
                except Exception as e:
                    print(f"  Warning: 无法设置相机视角: {e}")
                    # 使用默认视角
                    view_control.set_zoom(0.7)
            else:
                # 使用默认视角
                view_control.set_zoom(0.7)
            
            # 更新渲染
            vis.poll_events()
            vis.update_renderer()
            
            # 保存图像
            output_image_path = os.path.join(visualization_dir, ply_file.replace('.ply', '_open3d_visualization.png'))
            vis.capture_screen_image(output_image_path, do_render=True)
            
            print(f"  已保存可视化图片: {output_image_path}")
        
        except Exception as e:
            print(f"Error visualizing {ply_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 关闭可视化器
    vis.destroy_window()
    
    print(f"\n可视化完成！图片保存在: {visualization_dir}")

def visualize_pointclouds(dataset_dir, meta_json_path, num_samples=10):
    """
    可视化点云数据并保存为图片（支持无GUI环境，使用相机视角）
    优先使用Open3D进行可视化
    
    Args:
        dataset_dir: 数据集目录
        meta_json_path: meta_info.json路径
        num_samples: 可视化的样本数量
    """
    # 优先使用Open3D（更好的性能和效果）
    if USE_OPEN3D:
        visualize_pointclouds_open3d(dataset_dir, meta_json_path, num_samples)
    else:
        # 回退到matplotlib
        print("Warning: Open3D未安装，使用matplotlib进行可视化")
        visualize_pointclouds_matplotlib(dataset_dir, meta_json_path, num_samples)

def create_camera_frustum_visualization(intrinsic, extrinsic, image_width, image_height, scale=0.1):
    """
    创建立体锥形的相机视锥可视化（用于Open3D可视化）
    使用Open3D的create_camera_visualization方法创建相机视锥
    
    Args:
        intrinsic: 相机内参矩阵 3x3
        extrinsic: 相机外参矩阵 4x4
        image_width: 图像宽度（像素）
        image_height: 图像高度（像素）
        scale: 视锥的缩放因子（米）
    
    Returns:
        camera_frustum: Open3D LineSet对象，表示相机视锥
    """
    try:
        # 使用Open3D的内置方法创建相机视锥
        # 注意：Open3D的create_camera_visualization需要：
        # - view_width_px, view_height_px: 图像尺寸
        # - intrinsic: 3x3内参矩阵
        # - extrinsic: 4x4外参矩阵
        # - scale: 缩放因子
        
        # 确保内参是3x3矩阵
        intrinsic_array = np.array(intrinsic)
        if intrinsic_array.shape == (3, 3):
            intrinsic_3x3 = intrinsic_array
        elif intrinsic_array.shape == (4, 4):
            intrinsic_3x3 = intrinsic_array[:3, :3]
        else:
            # 尝试重塑为3x3
            intrinsic_3x3 = intrinsic_array.reshape(3, 3) if intrinsic_array.size == 9 else intrinsic_array[:3, :3]
        
        # 确保外参是4x4矩阵
        extrinsic_array = np.array(extrinsic)
        if extrinsic_array.shape == (4, 4):
            extrinsic_4x4 = extrinsic_array
        else:
            # 如果不是4x4，尝试构建
            extrinsic_4x4 = np.eye(4)
            if extrinsic_array.shape == (3, 3):
                extrinsic_4x4[:3, :3] = extrinsic_array
            elif extrinsic_array.shape == (3, 4):
                extrinsic_4x4[:3, :] = extrinsic_array
        
        # 创建相机视锥
        camera_frustum = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=image_width,
            view_height_px=image_height,
            intrinsic=intrinsic_3x3,
            extrinsic=extrinsic_4x4,
            scale=scale
        )
        
        # 设置颜色（使用半透明的蓝色，便于观察）
        camera_frustum.paint_uniform_color([0.2, 0.6, 1.0])  # 浅蓝色
        
        return camera_frustum
    
    except Exception as e:
        print(f"  Warning: 无法使用Open3D的create_camera_visualization: {e}")
        # 回退到手动创建简单的视锥
        return create_camera_frustum_manual(extrinsic, scale)

def create_camera_frustum_manual(extrinsic, scale=0.1):
    """
    手动创建相机视锥（回退方案）
    
    Args:
        extrinsic: 相机外参矩阵 4x4
        scale: 视锥的大小（米）
    
    Returns:
        camera_frustum: Open3D LineSet对象，表示相机视锥
    """
    R = np.array(extrinsic[:3, :3])
    t = np.array(extrinsic[:3, 3])
    
    # 相机位置（视锥的顶点）
    camera_pos = t
    
    # 相机坐标系的方向向量（在世界坐标系中）
    # OpenCV约定：Z轴向前（指向场景），X轴向右，Y轴向下
    x_axis = R[:, 0]  # 相机右方向
    y_axis = R[:, 1]  # 相机下方向
    z_axis = -R[:, 2]  # 相机前方向（指向场景）
    
    # 创建视锥的四个角点（在距离scale的平面上）
    # 假设视场角约为60度，创建一个矩形视锥
    frustum_width = scale * 0.5
    frustum_height = scale * 0.375  # 16:9 宽高比
    
    # 视锥的四个角点（在相机前方）
    corner1 = camera_pos + z_axis * scale + x_axis * frustum_width + y_axis * frustum_height
    corner2 = camera_pos + z_axis * scale - x_axis * frustum_width + y_axis * frustum_height
    corner3 = camera_pos + z_axis * scale - x_axis * frustum_width - y_axis * frustum_height
    corner4 = camera_pos + z_axis * scale + x_axis * frustum_width - y_axis * frustum_height
    
    # 创建点（相机位置 + 四个角点）
    points = np.array([
        camera_pos,  # 0: 相机位置（顶点）
        corner1,     # 1: 右上角
        corner2,     # 2: 左上角
        corner3,     # 3: 左下角
        corner4,     # 4: 右下角
    ])
    
    # 创建连接线（从相机位置到四个角点，以及四个角点之间的连接）
    lines = np.array([
        [0, 1],  # 相机 -> 右上角
        [0, 2],  # 相机 -> 左上角
        [0, 3],  # 相机 -> 左下角
        [0, 4],  # 相机 -> 右下角
        [1, 2],  # 右上角 -> 左上角
        [2, 3],  # 左上角 -> 左下角
        [3, 4],  # 左下角 -> 右下角
        [4, 1],  # 右下角 -> 右上角
    ])
    
    # 创建颜色（浅蓝色）
    colors = np.array([[0.2, 0.6, 1.0]] * len(lines))
    
    # 创建LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def visualize_pointclouds_interactive(dataset_dir, meta_json_path, num_samples=10):
    """
    交互式可视化点云数据（支持鼠标拖拽观察不同角度）
    使用Open3D的交互式可视化器
    
    Args:
        dataset_dir: 数据集目录
        meta_json_path: meta_info.json路径
        num_samples: 可视化的样本数量
    """
    if not USE_OPEN3D:
        print("Error: Open3D未安装，无法使用交互式可视化。请安装: pip install open3d")
        print("回退到静态可视化...")
        visualize_pointclouds(dataset_dir, meta_json_path, num_samples)
        return
    
    # 读取meta_info以获取相机参数
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    pointcloud_dir = os.path.join(dataset_dir, "POINTCLOUDS")
    
    # 获取点云文件列表，优先选择有robot arm的点云
    ply_files = [f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')]
    
    if len(ply_files) == 0:
        print("没有找到点云文件")
        return
    
    # 检查每个点云是否有robot arm，优先可视化有robot arm的
    ply_files_with_robot_arm = []
    ply_files_without_robot_arm = []
    
    print("正在检查点云文件，优先选择包含robot arm的点云...")
    for ply_file in ply_files:
        ply_path = os.path.join(pointcloud_dir, ply_file)
        try:
            points, colors, semantics = load_pointcloud_with_semantics(ply_path)
            if np.sum(semantics == 1) > 0:
                ply_files_with_robot_arm.append(ply_file)
            else:
                ply_files_without_robot_arm.append(ply_file)
        except:
            ply_files_without_robot_arm.append(ply_file)
    
    # 优先选择有robot arm的点云
    if len(ply_files_with_robot_arm) > 0:
        ply_files = ply_files_with_robot_arm[:num_samples]
        if len(ply_files) < num_samples and len(ply_files_without_robot_arm) > 0:
            remaining = num_samples - len(ply_files)
            ply_files.extend(ply_files_without_robot_arm[:remaining])
        print(f"找到 {len(ply_files_with_robot_arm)} 个包含robot arm的点云，选择前 {min(num_samples, len(ply_files_with_robot_arm))} 个进行可视化")
    else:
        ply_files = sorted(ply_files)[:num_samples]
        print(f"未找到包含robot arm的点云，可视化前 {num_samples} 个点云")
    
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
    print("  - 相机视锥：立体锥形显示相机位置和拍摄方向（浅蓝色）")
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
            
            # 获取对应的图像信息和相机参数
            img_info = None
            for img in coco_data.get('images', []):
                expected_pc = img['file_name'].replace('.jpg', '.ply').replace('.jpeg', '.ply')
                if expected_pc == ply_file:
                    img_info = img
                    break
            
            # 创建交互式可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"点云可视化 {i+1}/{len(ply_files)}: {ply_file}", 
                            width=1920, height=1080)
            
            # 默认显示语义颜色
            current_pcd = pcd_semantic
            show_semantic = True
            vis.add_geometry(current_pcd)
            
            # 添加相机位姿可视化（立体锥形视锥）
            camera_frustum = None
            if img_info and img_info.get('extrinsic') and img_info.get('intrinsic'):
                try:
                    extrinsic = np.array(img_info['extrinsic'])
                    intrinsic = np.array(img_info['intrinsic'])
                    
                    # 获取图像尺寸
                    image_width = img_info.get('width', 640)
                    image_height = img_info.get('height', 360)
                    
                    # 创建相机视锥（立体锥形）
                    # scale参数控制视锥的大小，可以根据点云范围调整
                    # 使用一个合理的默认值，大约为点云范围的10-20%
                    if len(points) > 0:
                        points_range = np.max(points, axis=0) - np.min(points, axis=0)
                        max_range = np.max(points_range)
                        frustum_scale = max_range * 0.15  # 视锥大小为点云范围的15%
                    else:
                        frustum_scale = 0.1  # 默认值
                    
                    camera_frustum = create_camera_frustum_visualization(
                        intrinsic, extrinsic, image_width, image_height, scale=frustum_scale
                    )
                    vis.add_geometry(camera_frustum, reset_bounding_box=False)
                    
                    print(f"  相机视锥已显示（立体锥形，scale={frustum_scale:.3f}m）")
                except Exception as e:
                    print(f"  Warning: 无法创建相机视锥可视化: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.point_size = 1.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
            
            # 如果有相机参数，设置初始视角
            view_control = vis.get_view_control()
            initial_view_params = None
            if img_info and img_info.get('extrinsic'):
                try:
                    extrinsic = np.array(img_info['extrinsic'])
                    R = extrinsic[:3, :3]
                    t = extrinsic[:3, 3]
                    
                    # 计算相机视角参数
                    camera_pos = t
                    camera_z = R @ np.array([0, 0, 1])
                    camera_front = camera_z / (np.linalg.norm(camera_z) + 1e-8)
                    camera_y = R @ np.array([0, 1, 0])
                    camera_up = -camera_y / (np.linalg.norm(camera_y) + 1e-8)
                    
                    # 计算点云中心
                    center = points.mean(axis=0)
                    
                    # 设置视角
                    view_control.set_lookat(center)
                    view_control.set_up(camera_up)
                    view_control.set_front(camera_front)
                    view_control.set_zoom(0.7)
                    
                    # 保存初始视角参数（用于重置）
                    initial_view_params = {
                        'lookat': center,
                        'up': camera_up,
                        'front': camera_front,
                        'zoom': 0.7
                    }
                except Exception as e:
                    print(f"  Warning: 无法设置相机视角: {e}")
            
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
            if camera_frustum:
                print("  相机视锥：立体锥形显示相机位置和朝向（浅蓝色）")
            
            # 在主循环中处理键盘事件
            import time
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
                    # 确保相机视锥仍然可见
                    if camera_frustum:
                        vis.update_geometry(camera_frustum)
                
                if keyboard_actions['reset_view']:
                    keyboard_actions['reset_view'] = False
                    if initial_view_params:
                        view_control.set_lookat(initial_view_params['lookat'])
                        view_control.set_up(initial_view_params['up'])
                        view_control.set_front(initial_view_params['front'])
                        view_control.set_zoom(initial_view_params['zoom'])
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
    
    parser = argparse.ArgumentParser(description="生成带语义的点云数据")
    parser.add_argument("--num_samples", type=int, default=-1, 
                       help="处理的样本数量（-1表示处理全部样本，默认-1）")
    parser.add_argument("--visualize", action="store_true",
                       help="进行可视化（默认不进行可视化）")
    parser.add_argument("--interactive", action="store_true",
                       help="使用交互式3D可视化（可以通过鼠标拖拽观察不同角度）")
    
    args = parser.parse_args()
    
    # 设置路径（基于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "data")
    dataset_dir = os.path.join(base_dir, "dataset")
    meta_json_path = os.path.join(dataset_dir, "meta_info.json")
    
    # 执行处理
    # -1 表示处理全部样本，转换为 None
    num_samples = None if args.num_samples == -1 else (args.num_samples if args.num_samples > 0 else None)
    visualize = args.visualize
    interactive = args.interactive
    
    process_dataset(meta_json_path, dataset_dir, num_samples=num_samples, 
                   visualize=visualize, interactive=interactive)

