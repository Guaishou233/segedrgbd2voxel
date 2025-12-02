#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证点云数据并使用Open3D进行交互式可视化
"""

import os
import numpy as np
import struct
import json
import random

# 检查Open3D是否安装
try:
    import open3d as o3d
    USE_OPEN3D = True
except ImportError:
    USE_OPEN3D = False
    print("Warning: Open3D未安装，无法进行交互式可视化。请安装: pip install open3d")

def load_pointcloud_with_semantics(ply_path):
    """
    加载带语义信息的点云（借鉴自generate_semantic_pointcloud.py）
    
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

def create_camera_frustum_visualization(intrinsic, extrinsic, image_width, image_height, scale=0.1):
    """
    创建相机视锥可视化（借鉴自generate_semantic_pointcloud.py）
    """
    if not USE_OPEN3D:
        return None
    
    try:
        intrinsic_array = np.array(intrinsic)
        if intrinsic_array.shape == (3, 3):
            intrinsic_3x3 = intrinsic_array
        elif intrinsic_array.shape == (4, 4):
            intrinsic_3x3 = intrinsic_array[:3, :3]
        else:
            intrinsic_3x3 = intrinsic_array.reshape(3, 3) if intrinsic_array.size == 9 else intrinsic_array[:3, :3]
        
        extrinsic_array = np.array(extrinsic)
        if extrinsic_array.shape == (4, 4):
            extrinsic_4x4 = extrinsic_array
        else:
            extrinsic_4x4 = np.eye(4)
            if extrinsic_array.shape == (3, 3):
                extrinsic_4x4[:3, :3] = extrinsic_array
            elif extrinsic_array.shape == (3, 4):
                extrinsic_4x4[:3, :] = extrinsic_array
        
        camera_frustum = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=image_width,
            view_height_px=image_height,
            intrinsic=intrinsic_3x3,
            extrinsic=extrinsic_4x4,
            scale=scale
        )
        camera_frustum.paint_uniform_color([0.2, 0.6, 1.0])  # 浅蓝色
        return camera_frustum
    except Exception as e:
        print(f"  Warning: 无法创建相机视锥: {e}")
        return None

def visualize_pointcloud_interactive(ply_path, img_info=None, max_points=500000):
    """
    交互式可视化单个点云（借鉴自generate_semantic_pointcloud.py）
    
    Args:
        ply_path: 点云文件路径
        img_info: 图像信息（包含相机参数）
        max_points: 最大点数（超过此数量会下采样）
    """
    if not USE_OPEN3D:
        print("Error: Open3D未安装，无法进行交互式可视化")
        return
    
    try:
        # 加载点云
        points, colors, semantics = load_pointcloud_with_semantics(ply_path)
        
        if len(points) == 0:
            print(f"Warning: {os.path.basename(ply_path)} 是空点云")
            return
        
        # 统计信息
        robot_arm_count = np.sum(semantics == 1)
        others_count = np.sum(semantics == 2)
        
        print(f"\n点云文件: {os.path.basename(ply_path)}")
        print(f"  总点数: {len(points):,}")
        print(f"  Robot arm点数: {robot_arm_count:,} ({robot_arm_count/len(points)*100:.1f}%)")
        print(f"  Others点数: {others_count:,} ({others_count/len(points)*100:.1f}%)")
        
        # 坐标范围
        x_range = [points[:, 0].min(), points[:, 0].max()]
        y_range = [points[:, 1].min(), points[:, 1].max()]
        z_range = [points[:, 2].min(), points[:, 2].max()]
        print(f"  坐标范围:")
        print(f"    X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"    Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"    Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
        print(f"  文件大小: {os.path.getsize(ply_path) / 1024 / 1024:.2f} MB")
        
        # 下采样（如果点太多）
        if len(points) > max_points:
            print(f"  点云较大，下采样到 {max_points:,} 个点以加快可视化...")
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
            semantics = semantics[indices]
        
        # 创建RGB点云
        pcd_rgb = o3d.geometry.PointCloud()
        pcd_rgb.points = o3d.utility.Vector3dVector(points)
        pcd_rgb.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # 创建语义点云（红色=robot arm, 灰色=others）
        semantic_colors = np.zeros((len(semantics), 3))
        robot_arm_mask = semantics == 1
        others_mask = semantics == 2
        semantic_colors[robot_arm_mask] = [1.0, 0.0, 0.0]  # 红色
        semantic_colors[others_mask] = [0.5, 0.5, 0.5]      # 灰色
        
        pcd_semantic = o3d.geometry.PointCloud()
        pcd_semantic.points = o3d.utility.Vector3dVector(points)
        pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
        
        # 创建交互式可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"点云可视化: {os.path.basename(ply_path)}", 
                         width=1920, height=1080)
        
        # 默认显示语义颜色
        current_pcd = pcd_semantic
        show_semantic = True
        vis.add_geometry(current_pcd)
        
        # 添加相机视锥（如果有相机参数）
        camera_frustum = None
        if img_info and img_info.get('extrinsic') and img_info.get('intrinsic'):
            try:
                extrinsic = np.array(img_info['extrinsic'])
                intrinsic = np.array(img_info['intrinsic'])
                image_width = img_info.get('width', 640)
                image_height = img_info.get('height', 360)
                
                # 计算合适的视锥大小
                if len(points) > 0:
                    points_range = np.max(points, axis=0) - np.min(points, axis=0)
                    max_range = np.max(points_range)
                    frustum_scale = max_range * 0.15
                else:
                    frustum_scale = 0.1
                
                camera_frustum = create_camera_frustum_visualization(
                    intrinsic, extrinsic, image_width, image_height, scale=frustum_scale
                )
                if camera_frustum:
                    vis.add_geometry(camera_frustum, reset_bounding_box=False)
                    print(f"  相机视锥已显示（浅蓝色，scale={frustum_scale:.3f}m）")
            except Exception as e:
                print(f"  Warning: 无法创建相机视锥: {e}")
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 1.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # 设置初始视角（如果有相机参数）
        view_control = vis.get_view_control()
        initial_view_params = None
        if img_info and img_info.get('extrinsic'):
            try:
                extrinsic = np.array(img_info['extrinsic'])
                R = extrinsic[:3, :3]
                t = extrinsic[:3, 3]
                
                camera_pos = t
                camera_z = R @ np.array([0, 0, 1])
                camera_front = camera_z / (np.linalg.norm(camera_z) + 1e-8)
                camera_y = R @ np.array([0, 1, 0])
                camera_up = -camera_y / (np.linalg.norm(camera_y) + 1e-8)
                
                center = points.mean(axis=0)
                view_control.set_lookat(center)
                view_control.set_up(camera_up)
                view_control.set_front(camera_front)
                view_control.set_zoom(0.7)
                
                initial_view_params = {
                    'lookat': center,
                    'up': camera_up,
                    'front': camera_front,
                    'zoom': 0.7
                }
            except Exception as e:
                print(f"  Warning: 无法设置相机视角: {e}")
        
        # 键盘事件处理
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
        
        # 显示操作说明
        print("\n操作说明：")
        print("  - 鼠标左键拖拽：旋转视角")
        print("  - 鼠标右键拖拽：平移视角")
        print("  - 滚轮：缩放")
        if keyboard_listener:
            print("  - 'S'：切换显示模式（RGB颜色 / 语义颜色）")
            print("  - 'R'：重置视角")
        print("  - 关闭窗口：退出当前点云")
        print("\n可视化说明：")
        print("  - 红色点 = robot arm (类别1)")
        print("  - 灰色点 = others (类别2)")
        if camera_frustum:
            print("  - 浅蓝色视锥 = 相机位置和朝向")
        print()
        
        # 主循环
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
            time.sleep(0.01)
        
        # 停止键盘监听器
        if keyboard_listener:
            keyboard_listener.stop()
        
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error visualizing {ply_path}: {e}")
        import traceback
        traceback.print_exc()

def verify_and_visualize_pointclouds(dataset_dir, num_samples=10, random_select=False, 
                                     interactive=True, show_stats=True):
    """
    验证点云数据并进行交互式可视化
    
    Args:
        dataset_dir: 数据集目录
        num_samples: 可视化的样本数量（-1表示全部）
        random_select: 是否随机选择样本
        interactive: 是否进行交互式可视化
        show_stats: 是否显示统计信息
    """
    pointcloud_dir = os.path.join(dataset_dir, "POINTCLOUDS")
    meta_json_path = os.path.join(dataset_dir, "meta_info.json")
    
    # 检查目录和文件
    if not os.path.exists(pointcloud_dir):
        print(f"Error: 点云目录不存在: {pointcloud_dir}")
        return
    
    if not os.path.exists(meta_json_path):
        print(f"Error: meta_info.json不存在: {meta_json_path}")
        return
    
    # 读取meta_info
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 获取有点云信息的图像
    images_with_pc = [img for img in coco_data.get('images', []) if 'pointcloud' in img]
    
    if len(images_with_pc) == 0:
        print("没有找到包含点云信息的图像")
        return
    
    print(f"找到 {len(images_with_pc)} 个包含点云信息的图像")
    
    # 选择样本
    if num_samples == -1:
        selected_images = images_with_pc
        print(f"将处理全部 {len(selected_images)} 个样本")
    else:
        if random_select:
            selected_images = random.sample(images_with_pc, min(num_samples, len(images_with_pc)))
            print(f"随机选择了 {len(selected_images)} 个样本")
        else:
            selected_images = images_with_pc[:num_samples]
            print(f"选择了前 {len(selected_images)} 个样本")
    
    # 优先选择包含robot arm的点云
    if interactive and USE_OPEN3D:
        print("\n正在检查点云，优先选择包含robot arm的点云...")
        images_with_robot_arm = []
        images_without_robot_arm = []
        
        for img_info in selected_images:
            pc_filename = img_info.get('pointcloud', '')
            if not pc_filename:
                continue
            
            pc_path = os.path.join(pointcloud_dir, pc_filename)
            if not os.path.exists(pc_path):
                continue
            
            try:
                points, colors, semantics = load_pointcloud_with_semantics(pc_path)
                if np.sum(semantics == 1) > 0:
                    images_with_robot_arm.append(img_info)
                else:
                    images_without_robot_arm.append(img_info)
            except:
                images_without_robot_arm.append(img_info)
        
        # 优先显示有robot arm的点云
        if len(images_with_robot_arm) > 0:
            selected_images = images_with_robot_arm
            if len(images_without_robot_arm) > 0 and len(selected_images) < num_samples:
                remaining = num_samples - len(selected_images)
                selected_images.extend(images_without_robot_arm[:remaining])
            print(f"找到 {len(images_with_robot_arm)} 个包含robot arm的点云")
    
    # 显示统计信息
    if show_stats:
        print(f"\n验证 {len(selected_images)} 个点云文件...\n")
        for i, img_info in enumerate(selected_images):
            pc_filename = img_info.get('pointcloud', '')
            if not pc_filename:
                continue
            
            pc_path = os.path.join(pointcloud_dir, pc_filename)
            if not os.path.exists(pc_path):
                print(f"{i+1}. {pc_filename}: 文件不存在")
                continue
            
            try:
                points, colors, semantics = load_pointcloud_with_semantics(pc_path)
                
                robot_arm_count = np.sum(semantics == 1)
                others_count = np.sum(semantics == 2)
                
                x_range = [points[:, 0].min(), points[:, 0].max()]
                y_range = [points[:, 1].min(), points[:, 1].max()]
                z_range = [points[:, 2].min(), points[:, 2].max()]
                
                print(f"{i+1}. {pc_filename}")
                print(f"   总点数: {len(points):,}")
                print(f"   Robot arm: {robot_arm_count:,} ({robot_arm_count/len(semantics)*100:.1f}%)")
                print(f"   Others: {others_count:,} ({others_count/len(semantics)*100:.1f}%)")
                print(f"   坐标范围: X[{x_range[0]:.3f}, {x_range[1]:.3f}] "
                      f"Y[{y_range[0]:.3f}, {y_range[1]:.3f}] "
                      f"Z[{z_range[0]:.3f}, {z_range[1]:.3f}]")
                print(f"   文件大小: {os.path.getsize(pc_path) / 1024 / 1024:.2f} MB")
                print()
            except Exception as e:
                print(f"{i+1}. {pc_filename}: 错误 - {e}\n")
    
    # 交互式可视化
    if interactive:
        if not USE_OPEN3D:
            print("\nWarning: Open3D未安装，无法进行交互式可视化")
            return
        
        print("\n" + "=" * 60)
        print("开始交互式可视化")
        print("=" * 60)
        
        for i, img_info in enumerate(selected_images):
            pc_filename = img_info.get('pointcloud', '')
            if not pc_filename:
                continue
            
            pc_path = os.path.join(pointcloud_dir, pc_filename)
            if not os.path.exists(pc_path):
                continue
            
            print(f"\n[{i+1}/{len(selected_images)}] 正在显示: {pc_filename}")
            visualize_pointcloud_interactive(pc_path, img_info)
            
            # 询问是否继续
            if i < len(selected_images) - 1:
                try:
                    response = input(f"\n是否继续查看下一个点云？(y/n，默认y): ").strip().lower()
                    if response == 'n':
                        print("退出可视化")
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\n退出可视化")
                    break
        
        print("\n可视化完成！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证点云数据并使用Open3D进行交互式可视化")
    parser.add_argument("--num_samples", type=int, default=10, 
                       help="可视化的样本数量（-1表示全部，默认10）")
    parser.add_argument("--random", action="store_true",
                       help="随机选择样本（默认按顺序选择）")
    parser.add_argument("--no_interactive", action="store_true",
                       help="不进行交互式可视化，只显示统计信息")
    parser.add_argument("--no_stats", action="store_true",
                       help="不显示统计信息，只进行交互式可视化")
    parser.add_argument("--dataset_dir", type=str, default=None,
                       help="数据集目录（默认使用脚本所在目录下的data/dataset）")
    
    args = parser.parse_args()
    
    # 设置数据集目录
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "data")
        dataset_dir = os.path.join(base_dir, "dataset")
    
    verify_and_visualize_pointclouds(
        dataset_dir, 
        num_samples=args.num_samples,
        random_select=args.random,
        interactive=not args.no_interactive,
        show_stats=not args.no_stats
    )

