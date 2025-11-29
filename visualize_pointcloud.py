#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
点云数据可视化工具
用于检查生成的点云数据是否正确
支持同时显示RGB图像、深度图、语义分割图和点云
"""

import os
import json
import numpy as np
from PIL import Image
import argparse
import matplotlib
# 检测是否有GUI环境，如果没有则使用非交互式后端
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import matplotlib.patches as mpatches

# 尝试导入open3d（更好的3D可视化）
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Note: open3d not installed, will use matplotlib for 3D visualization")
    print("Install open3d for better interactive 3D visualization: pip install open3d")


def load_pointcloud(pointcloud_path):
    """
    加载点云数据
    
    Args:
        pointcloud_path: 点云文件路径（.ply或.npy）
    
    Returns:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)，如果没有颜色则返回None
    """
    if pointcloud_path.endswith('.ply'):
        return load_ply(pointcloud_path)
    elif pointcloud_path.endswith('.npy'):
        data = np.load(pointcloud_path)
        if data.shape[1] >= 6:
            # 包含颜色信息
            points = data[:, :3]
            colors = data[:, 3:6].astype(np.uint8)
            return points, colors
        else:
            # 只有坐标
            return data, None
    else:
        raise ValueError(f"不支持的点云格式: {pointcloud_path}")


def load_ply(ply_path):
    """
    加载PLY格式的点云（支持ASCII和二进制格式）
    
    Args:
        ply_path: PLY文件路径
    
    Returns:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)，如果没有颜色则返回None
    """
    # 优先使用open3d加载（支持二进制格式）
    if OPEN3D_AVAILABLE:
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            colors = None
            if pcd.has_colors():
                # open3d返回的颜色是0-1范围，转换为0-255
                colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            return points, colors
        except Exception as e:
            print(f"Warning: Failed to load PLY with open3d: {e}")
            raise ValueError(f"无法加载PLY文件。请安装open3d: pip install open3d 或 conda install -c open3d-admin open3d")
    
    # 尝试使用plyfile库（如果可用）
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        points = np.array([vertex['x'], vertex['y'], vertex['z']]).T
        
        colors = None
        if 'red' in vertex.dtype.names and 'green' in vertex.dtype.names and 'blue' in vertex.dtype.names:
            colors = np.array([vertex['red'], vertex['green'], vertex['blue']]).T
        
        return points, colors
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to load PLY with plyfile: {e}")
    
    # 回退到手动解析（仅支持ASCII格式）
    points = []
    colors = []
    has_colors = False
    is_binary = False
    
    with open(ply_path, 'rb') as f:
        # 读取文件头（前几行应该是ASCII）
        header_lines = []
        while True:
            line = f.readline()
            try:
                line_str = line.decode('ascii').strip()
                header_lines.append(line_str)
                if line_str == 'end_header':
                    break
                if 'binary' in line_str.lower():
                    is_binary = True
            except:
                # 如果无法解码，可能是二进制文件
                is_binary = True
                break
        
        if is_binary:
            raise ValueError(
                "检测到二进制PLY格式。\n"
                "请安装open3d来加载二进制PLY文件：\n"
                "  pip install open3d\n"
                "或\n"
                "  conda install -c open3d-admin open3d\n"
                "或者安装plyfile库：\n"
                "  pip install plyfile"
            )
        
        # 解析文件头
        num_vertices = 0
        for line_str in header_lines:
            if line_str.startswith('element vertex'):
                num_vertices = int(line_str.split()[-1])
            elif 'red' in line_str.lower() or 'green' in line_str.lower() or 'blue' in line_str.lower():
                has_colors = True
        
        # 读取点云数据（ASCII格式）
        for i in range(num_vertices):
            line = f.readline().decode('ascii').strip().split()
            if len(line) >= 3:
                x, y, z = float(line[0]), float(line[1]), float(line[2])
                points.append([x, y, z])
                
                if has_colors and len(line) >= 6:
                    r, g, b = int(line[3]), int(line[4]), int(line[5])
                    colors.append([r, g, b])
    
    points = np.array(points)
    colors = np.array(colors) if colors else None
    
    return points, colors


def analyze_semantic_colors(colors):
    """
    分析点云中的语义颜色分布
    
    Args:
        colors: 点云颜色 (N, 3)，值范围0-255
    
    Returns:
        semantic_stats: 语义标签统计信息
    """
    if colors is None:
        return None
    
    # 定义语义颜色映射（RGB值）
    semantic_colors_map = {
        'others': (127, 127, 127),      # 灰色 (0.5, 0.5, 0.5) * 255
        'robot_arm': (255, 0, 0),      # 红色 (1.0, 0.0, 0.0) * 255
    }
    
    # 容差范围（允许颜色值有小的偏差）
    tolerance = 10
    
    stats = {}
    for label, (r, g, b) in semantic_colors_map.items():
        mask = ((np.abs(colors[:, 0] - r) < tolerance) & 
                (np.abs(colors[:, 1] - g) < tolerance) & 
                (np.abs(colors[:, 2] - b) < tolerance))
        count = np.sum(mask)
        stats[label] = {
            'count': count,
            'percentage': count / len(colors) * 100 if len(colors) > 0 else 0,
            'color': (r/255.0, g/255.0, b/255.0)
        }
    
    # 统计未知颜色
    known_mask = np.zeros(len(colors), dtype=bool)
    for label, (r, g, b) in semantic_colors_map.items():
        mask = ((np.abs(colors[:, 0] - r) < tolerance) & 
                (np.abs(colors[:, 1] - g) < tolerance) & 
                (np.abs(colors[:, 2] - b) < tolerance))
        known_mask |= mask
    
    unknown_count = np.sum(~known_mask)
    stats['unknown'] = {
        'count': unknown_count,
        'percentage': unknown_count / len(colors) * 100 if len(colors) > 0 else 0,
        'color': (0.0, 0.0, 1.0)  # 蓝色
    }
    
    return stats


def visualize_with_matplotlib(points, colors, rgb_image, depth_image, seg_image, 
                              title="Point Cloud Visualization", save_path=None, 
                              semantic_stats=None, params=None):
    """
    使用matplotlib可视化点云和图像（增强版，便于人工核对）
    
    Args:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        rgb_image: RGB图像
        depth_image: 深度图
        seg_image: 语义分割图
        title: 窗口标题
        save_path: 保存路径，如果为None则尝试显示窗口
        semantic_stats: 语义标签统计信息
        params: 相机参数信息
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 创建3x4的网格布局
    # 第一行：RGB、深度图、语义分割图、统计信息
    # 第二行：点云3D视图（3个视角）
    # 第三行：点云2D投影视图
    
    # RGB图像
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(rgb_image)
    ax1.set_title('RGB Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 深度图
    ax2 = plt.subplot(3, 4, 2)
    depth_vis = depth_image.copy()
    if depth_vis.dtype != np.uint8:
        # 归一化到0-255
        if depth_vis.max() > depth_vis.min():
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255
        depth_vis = depth_vis.astype(np.uint8)
    im2 = ax2.imshow(depth_vis, cmap='jet')
    ax2.set_title('Depth Map', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 语义分割图（彩色显示）
    ax3 = plt.subplot(3, 4, 3)
    seg_colored = np.zeros((*seg_image.shape, 3), dtype=np.uint8)
    seg_colored[seg_image == 0] = [127, 127, 127]  # others - 灰色
    seg_colored[seg_image == 255] = [255, 0, 0]    # robot_arm - 红色
    ax3.imshow(seg_colored)
    ax3.set_title('Segmentation Map\n(Red=Robot Arm, Gray=Others)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1.0, 0.0, 0.0), label='Robot Arm'),
        Patch(facecolor=(0.5, 0.5, 0.5), label='Others')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # 统计信息面板
    ax4 = plt.subplot(3, 4, 4)
    ax4.axis('off')
    
    # 点云统计
    stats_text = f"点云统计信息\n"
    stats_text += f"{'='*30}\n"
    stats_text += f"总点数: {len(points):,}\n"
    stats_text += f"X范围: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m\n"
    stats_text += f"Y范围: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m\n"
    stats_text += f"Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m\n"
    stats_text += f"\n"
    
    # 语义标签统计
    if semantic_stats:
        stats_text += f"语义标签统计\n"
        stats_text += f"{'='*30}\n"
        for label, info in semantic_stats.items():
            stats_text += f"{label}:\n"
            stats_text += f"  点数: {info['count']:,}\n"
            stats_text += f"  比例: {info['percentage']:.2f}%\n"
    
    # 相机参数信息
    if params:
        stats_text += f"\n相机信息\n"
        stats_text += f"{'='*30}\n"
        if 'cam_id' in params:
            stats_text += f"相机ID: {params['cam_id']}\n"
        if 'segmentation_classes' in params:
            stats_text += f"类别: {', '.join(params['segmentation_classes'])}\n"
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 点云可视化 - 3D视图（主视图）
    ax5 = plt.subplot(3, 4, (5, 6), projection='3d')
    
    # 如果点太多，进行下采样以提高性能
    max_points = 100000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_vis = points[indices]
        colors_vis = colors[indices] if colors is not None else None
        print(f"Point cloud has {len(points)} points, downsampling to {max_points} for display")
    else:
        points_vis = points
        colors_vis = colors
    
    # 绘制点云
    if colors_vis is not None:
        # 归一化颜色到0-1范围
        colors_normalized = colors_vis.astype(np.float32) / 255.0
        scatter = ax5.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                             c=colors_normalized, s=0.3, alpha=0.8, edgecolors='none')
    else:
        # 使用深度作为颜色
        z_values = points_vis[:, 2]
        scatter = ax5.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                             c=z_values, cmap='viridis', s=0.3, alpha=0.8)
    
    ax5.set_xlabel('X (m)', fontsize=10)
    ax5.set_ylabel('Y (m)', fontsize=10)
    ax5.set_zlabel('Z (m)', fontsize=10)
    ax5.set_title(f'Point Cloud 3D View\n({len(points):,} points)', fontsize=11, fontweight='bold')
    
    # 设置相等的坐标轴比例
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax5.set_xlim(mid_x - max_range, mid_x + max_range)
    ax5.set_ylim(mid_y - max_range, mid_y + max_range)
    ax5.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 点云2D投影视图 - XY平面（俯视图）
    ax6 = plt.subplot(3, 4, 7)
    if colors_vis is not None:
        colors_normalized = colors_vis.astype(np.float32) / 255.0
        ax6.scatter(points_vis[:, 0], points_vis[:, 1], c=colors_normalized, s=0.1, alpha=0.6)
    else:
        ax6.scatter(points_vis[:, 0], points_vis[:, 1], c=points_vis[:, 2], cmap='viridis', s=0.1, alpha=0.6)
    ax6.set_xlabel('X (m)', fontsize=9)
    ax6.set_ylabel('Y (m)', fontsize=9)
    ax6.set_title('Top View (XY)', fontsize=10, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal', adjustable='box')
    
    # 点云2D投影视图 - XZ平面（侧视图）
    ax7 = plt.subplot(3, 4, 8)
    if colors_vis is not None:
        colors_normalized = colors_vis.astype(np.float32) / 255.0
        ax7.scatter(points_vis[:, 0], points_vis[:, 2], c=colors_normalized, s=0.1, alpha=0.6)
    else:
        ax7.scatter(points_vis[:, 0], points_vis[:, 2], c=points_vis[:, 1], cmap='viridis', s=0.1, alpha=0.6)
    ax7.set_xlabel('X (m)', fontsize=9)
    ax7.set_ylabel('Z (m)', fontsize=9)
    ax7.set_title('Side View (XZ)', fontsize=10, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal', adjustable='box')
    
    # 点云2D投影视图 - YZ平面（前视图）
    ax8 = plt.subplot(3, 4, 9)
    if colors_vis is not None:
        colors_normalized = colors_vis.astype(np.float32) / 255.0
        ax8.scatter(points_vis[:, 1], points_vis[:, 2], c=colors_normalized, s=0.1, alpha=0.6)
    else:
        ax8.scatter(points_vis[:, 1], points_vis[:, 2], c=points_vis[:, 0], cmap='viridis', s=0.1, alpha=0.6)
    ax8.set_xlabel('Y (m)', fontsize=9)
    ax8.set_ylabel('Z (m)', fontsize=9)
    ax8.set_title('Front View (YZ)', fontsize=10, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.set_aspect('equal', adjustable='box')
    
    # 语义标签分布饼图
    ax9 = plt.subplot(3, 4, 10)
    if semantic_stats:
        labels = list(semantic_stats.keys())
        sizes = [semantic_stats[label]['count'] for label in labels]
        colors_pie = [semantic_stats[label]['color'] for label in labels]
        ax9.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax9.set_title('Semantic Label\nDistribution', fontsize=10, fontweight='bold')
    else:
        ax9.text(0.5, 0.5, 'No semantic\ninformation', 
                ha='center', va='center', transform=ax9.transAxes)
        ax9.axis('off')
    
    # 深度分布直方图
    ax10 = plt.subplot(3, 4, 11)
    ax10.hist(points[:, 2], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax10.set_xlabel('Depth (Z, m)', fontsize=9)
    ax10.set_ylabel('Point Count', fontsize=9)
    ax10.set_title('Depth Distribution', fontsize=10, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    
    # 点云密度热图（XY平面）
    ax11 = plt.subplot(3, 4, 12)
    # 创建2D直方图
    H, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=50)
    im11 = ax11.imshow(H.T, origin='lower', cmap='hot', aspect='auto',
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax11.set_xlabel('X (m)', fontsize=9)
    ax11.set_ylabel('Y (m)', fontsize=9)
    ax11.set_title('Point Density\nHeatmap (XY)', fontsize=10, fontweight='bold')
    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 如果有保存路径，保存图片；否则尝试显示
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to: {save_path}")
        plt.close()
    else:
        # 尝试显示，如果失败则保存到默认位置
        try:
            plt.show()
        except Exception as e:
            print(f"Cannot display window (likely no GUI): {e}")
            default_save_path = "pointcloud_visualization.png"
            plt.savefig(default_save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"Visualization saved to: {default_save_path}")
            plt.close()


def visualize_with_open3d(points, colors, rgb_image, depth_image, seg_image):
    """
    使用open3d进行交互式3D可视化
    
    Args:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        rgb_image: RGB图像
        depth_image: 深度图
        seg_image: 语义分割图
    """
    if not OPEN3D_AVAILABLE:
        print("open3d not installed, using matplotlib for visualization")
        visualize_with_matplotlib(points, colors, rgb_image, depth_image, seg_image)
        return
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # 归一化颜色到0-1范围
        colors_normalized = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    # 显示点云
    print("Displaying point cloud (drag mouse to rotate, scroll to zoom)")
    print("Press 'Q' or close window to exit")
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="Point Cloud Visualization",
                                      width=1024, 
                                      height=768)
    
    # 同时显示图像（使用matplotlib）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    depth_vis = depth_image.copy()
    if depth_vis.dtype != np.uint8:
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8) * 255
        depth_vis = depth_vis.astype(np.uint8)
    axes[1].imshow(depth_vis, cmap='jet')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    axes[2].imshow(seg_image, cmap='gray')
    axes[2].set_title('Segmentation Map')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 在服务器环境下，open3d可能也无法显示，所以也保存图片
    try:
        plt.show()
    except Exception as e:
        print(f"Cannot display window: {e}")
        default_save_path = "pointcloud_images.png"
        plt.savefig(default_save_path, dpi=150, bbox_inches='tight')
        print(f"Images saved to: {default_save_path}")
        plt.close()


def visualize_dataset_sample(dataset_dir, sample_name=None, use_open3d=False, output_dir=None):
    """
    可视化数据集中的一个样本
    
    Args:
        dataset_dir: 数据集目录
        sample_name: 样本名称（不含扩展名），如果为None则使用第一个找到的样本
        use_open3d: 是否使用open3d进行3D可视化
    """
    # 支持两种数据集结构：
    # 1. pointed_dataset/POINTCLOUD/ (点云数据集)
    # 2. segmented_dataset/ (原始数据集，需要从上级目录查找)
    
    # 检查是否是pointed_dataset结构
    if os.path.exists(os.path.join(dataset_dir, 'POINTCLOUD')):
        # pointed_dataset结构
        pointcloud_dir = os.path.join(dataset_dir, 'POINTCLOUD')
        # 从segmented_dataset获取原始图像
        parent_dir = os.path.dirname(dataset_dir)
        segmented_dir = os.path.join(parent_dir, 'segmented_dataset')
        rgb_dir = os.path.join(segmented_dir, 'RGB')
        depth_dir = os.path.join(segmented_dir, 'DEPTH')
        seg_dir = os.path.join(segmented_dir, 'SEG')
        params_dir = os.path.join(segmented_dir, 'params')
    else:
        # segmented_dataset结构（向后兼容）
        rgb_dir = os.path.join(dataset_dir, 'RGB')
        depth_dir = os.path.join(dataset_dir, 'DEPTH')
        seg_dir = os.path.join(dataset_dir, 'SEG')
        params_dir = os.path.join(dataset_dir, 'params')
        pointcloud_dir = os.path.join(dataset_dir, 'POINTCLOUD')
    
    # 如果没有指定样本，使用第一个
    if sample_name is None:
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        if len(rgb_files) == 0:
            print(f"Error: No RGB images found in {rgb_dir}")
            return
        sample_name = rgb_files[0].replace('.png', '')
        print(f"Using first sample: {sample_name}")
    
    # 构建文件路径
    rgb_path = os.path.join(rgb_dir, sample_name + '.png')
    depth_path = os.path.join(depth_dir, sample_name + '.png')
    seg_path = os.path.join(seg_dir, sample_name + '.png')
    params_path = os.path.join(params_dir, sample_name + '.json')
    
    # 查找点云文件（可能是.ply或.npy）
    pointcloud_path = None
    if os.path.exists(os.path.join(pointcloud_dir, sample_name + '.ply')):
        pointcloud_path = os.path.join(pointcloud_dir, sample_name + '.ply')
    elif os.path.exists(os.path.join(pointcloud_dir, sample_name + '.npy')):
        pointcloud_path = os.path.join(pointcloud_dir, sample_name + '.npy')
    
    # 检查文件是否存在
    missing_files = []
    for name, path in [('RGB', rgb_path), ('DEPTH', depth_path), 
                       ('SEG', seg_path), ('params', params_path)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("Error: The following files do not exist:")
        for f in missing_files:
            print(f"  {f}")
        return
    
    if pointcloud_path is None:
        print(f"Error: Point cloud file not found (looking for {sample_name}.ply or {sample_name}.npy)")
        return
    
    print(f"Loading sample: {sample_name}")
    print(f"Point cloud file: {pointcloud_path}")
    
    # 加载数据
    print("Loading data...")
    rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
    depth_image = np.array(Image.open(depth_path))
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]
    
    seg_image = np.array(Image.open(seg_path))
    if len(seg_image.shape) == 3:
        seg_image = seg_image[:, :, 0]
    
    # 加载点云
    points, colors = load_pointcloud(pointcloud_path)
    print(f"Point cloud has {len(points)} points")
    if colors is not None:
        print(f"Point cloud contains color information")
    
    # 分析语义颜色统计
    semantic_stats = None
    if colors is not None:
        semantic_stats = analyze_semantic_colors(colors)
        print("\n语义标签统计:")
        for label, info in semantic_stats.items():
            print(f"  {label}: {info['count']:,} 点 ({info['percentage']:.2f}%)")
    
    # 加载参数文件获取信息
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    title = f"Point Cloud Visualization - {sample_name}"
    
    # 确定保存路径
    save_path = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{sample_name}_visualization.png")
    elif 'DISPLAY' not in os.environ:
        # 没有GUI环境，自动保存到当前目录
        save_path = f"{sample_name}_visualization.png"
    
    # 可视化
    if use_open3d and OPEN3D_AVAILABLE and 'DISPLAY' in os.environ:
        visualize_with_open3d(points, colors, rgb_image, depth_image, seg_image)
        # 同时生成matplotlib版本用于保存
        visualize_with_matplotlib(points, colors, rgb_image, depth_image, seg_image, 
                                  title, save_path, semantic_stats, params)
    else:
        visualize_with_matplotlib(points, colors, rgb_image, depth_image, seg_image, 
                                  title, save_path, semantic_stats, params)


def list_samples(dataset_dir):
    """
    列出数据集中的所有样本
    
    Args:
        dataset_dir: 数据集目录
    """
    rgb_dir = os.path.join(dataset_dir, 'RGB')
    if not os.path.exists(rgb_dir):
        print(f"Error: RGB directory does not exist: {rgb_dir}")
        return
    
    rgb_files = sorted([f.replace('.png', '') for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    print(f"\nDataset: {dataset_dir}")
    print(f"Found {len(rgb_files)} samples:\n")
    
    for i, sample_name in enumerate(rgb_files[:20]):  # 只显示前20个
        print(f"  {i+1}. {sample_name}")
    
    if len(rgb_files) > 20:
        print(f"  ... and {len(rgb_files) - 20} more samples")
    
    print(f"\nUse --sample parameter to specify which sample to visualize")
    print(f"Example: --sample {rgb_files[0] if rgb_files else 'sample_name'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Visualization Tool')
    parser.add_argument('--dataset_dir', type=str, 
                        default='/data/tangqiansong/rgb2voxel/data/pointed_dataset',
                        help='Dataset directory')
    parser.add_argument('--sample', type=str, default=None,
                        help='Sample name to visualize (without extension), if not specified use the first one')
    parser.add_argument('--list', action='store_true',
                        help='List all available samples')
    parser.add_argument('--open3d', action='store_true',
                        help='Use open3d for 3D visualization (requires open3d installation)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for saving visualization images (if not specified, will try to display or save to current directory)')
    
    args = parser.parse_args()
    
    # 检测GUI环境
    has_gui = 'DISPLAY' in os.environ
    if not has_gui:
        print("No GUI environment detected (no DISPLAY). Images will be saved to files.")
        if args.output_dir is None:
            print("Using current directory for output. Use --output_dir to specify a different location.")
    
    if args.list:
        list_samples(args.dataset_dir)
    else:
        visualize_dataset_sample(args.dataset_dir, args.sample, 
                                use_open3d=args.open3d, output_dir=args.output_dir)

