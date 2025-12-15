#!/usr/bin/env python3
"""
可视化点云和相机位置

生成图像显示：
1. 点云的整体分布
2. 6个相机的位置和朝向
3. 验证相机是否正确围绕物体
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import open3d as o3d


def load_ply_points(ply_path: str, max_points: int = 50000):
    """加载 PLY 点云"""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 随机采样
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    return points, colors


def visualize_cameras_and_pointcloud(metadata_path: str, ply_path: str, output_path: str):
    """
    可视化相机位置和点云
    """
    # 加载相机信息
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    cameras = metadata.get('cameras', {})
    
    # 加载点云
    print(f"加载点云: {ply_path}")
    points, colors = load_ply_points(ply_path, max_points=30000)
    print(f"  点数: {len(points)}")
    print(f"  范围 X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"  范围 Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"  范围 Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    
    # 点云中心
    pc_center = points.mean(axis=0)
    print(f"  中心: [{pc_center[0]:.2f}, {pc_center[1]:.2f}, {pc_center[2]:.2f}]")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 主视图 - 3D
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 绘制点云（采样显示）
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors, s=0.5, alpha=0.3)
    
    # 绘制相机
    cam_colors = plt.cm.tab10(np.linspace(0, 1, len(cameras)))
    camera_positions = []
    
    for idx, (cam_name, cam_info) in enumerate(cameras.items()):
        extrinsic = np.array(cam_info['extrinsic'])
        
        # world-to-camera 格式，需要计算相机位置
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        cam_pos = -R.T @ t  # 相机在世界坐标系中的位置
        camera_positions.append(cam_pos)
        
        # 相机朝向（Z 轴在世界坐标系中的方向）
        cam_dir = R.T @ np.array([0, 0, 1])  # 相机 Z 轴在世界坐标系中的方向
        
        color = cam_colors[idx]
        
        # 绘制相机位置
        ax1.scatter(*cam_pos, c=[color], s=100, marker='^', label=cam_name)
        
        # 绘制相机朝向
        ax1.quiver(*cam_pos, *(cam_dir * 0.5), color=color, arrow_length_ratio=0.2)
        
        # 标注
        ax1.text(cam_pos[0], cam_pos[1], cam_pos[2], f'  {cam_name}', fontsize=8)
    
    camera_positions = np.array(camera_positions)
    cam_center = camera_positions.mean(axis=0)
    
    print(f"\n相机信息:")
    print(f"  相机中心: [{cam_center[0]:.2f}, {cam_center[1]:.2f}, {cam_center[2]:.2f}]")
    print(f"  相机到点云中心距离: {np.linalg.norm(cam_center - pc_center):.2f}")
    
    # 绘制点云中心
    ax1.scatter(*pc_center, c='red', s=200, marker='*', label='PC Center')
    
    # 绘制相机中心
    ax1.scatter(*cam_center, c='blue', s=200, marker='o', label='Cam Center')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View: Cameras and Point Cloud')
    ax1.legend(loc='upper left', fontsize=6)
    
    # XY 平面视图
    ax2 = fig.add_subplot(222)
    ax2.scatter(points[:, 0], points[:, 1], c='gray', s=0.1, alpha=0.1)
    for idx, (cam_name, pos) in enumerate(zip(cameras.keys(), camera_positions)):
        ax2.scatter(pos[0], pos[1], c=[cam_colors[idx]], s=100, marker='^')
        ax2.annotate(cam_name, (pos[0], pos[1]), fontsize=8)
    ax2.scatter(pc_center[0], pc_center[1], c='red', s=200, marker='*')
    ax2.scatter(cam_center[0], cam_center[1], c='blue', s=200, marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (XY)')
    ax2.axis('equal')
    
    # XZ 平面视图
    ax3 = fig.add_subplot(223)
    ax3.scatter(points[:, 0], points[:, 2], c='gray', s=0.1, alpha=0.1)
    for idx, (cam_name, pos) in enumerate(zip(cameras.keys(), camera_positions)):
        ax3.scatter(pos[0], pos[2], c=[cam_colors[idx]], s=100, marker='^')
        ax3.annotate(cam_name, (pos[0], pos[2]), fontsize=8)
    ax3.scatter(pc_center[0], pc_center[2], c='red', s=200, marker='*')
    ax3.scatter(cam_center[0], cam_center[2], c='blue', s=200, marker='o')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Front View (XZ)')
    ax3.axis('equal')
    
    # YZ 平面视图
    ax4 = fig.add_subplot(224)
    ax4.scatter(points[:, 1], points[:, 2], c='gray', s=0.1, alpha=0.1)
    for idx, (cam_name, pos) in enumerate(zip(cameras.keys(), camera_positions)):
        ax4.scatter(pos[1], pos[2], c=[cam_colors[idx]], s=100, marker='^')
        ax4.annotate(cam_name, (pos[1], pos[2]), fontsize=8)
    ax4.scatter(pc_center[1], pc_center[2], c='red', s=200, marker='*')
    ax4.scatter(cam_center[1], cam_center[2], c='blue', s=200, marker='o')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('Side View (YZ)')
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化图像已保存到: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化点云和相机位置")
    parser.add_argument("--metadata", "-m", type=str, 
                        default="/home/qiansongtang/Documents/program/rgb2voxel/data/dataset2/task_2/metadata.json",
                        help="metadata.json 路径")
    parser.add_argument("--ply", "-p", type=str,
                        default=None,
                        help="PLY 点云文件路径（自动查找）")
    parser.add_argument("--output", "-o", type=str,
                        default="/home/qiansongtang/Documents/program/rgb2voxel/data/dataset2/pointcloud_cameras_vis.png",
                        help="输出图像路径")
    
    args = parser.parse_args()
    
    # 自动查找 PLY 文件
    if args.ply is None:
        metadata_dir = Path(args.metadata).parent
        pointcloud_dir = metadata_dir / "POINTCLOUDS"
        ply_files = list(pointcloud_dir.glob("*.ply"))
        if ply_files:
            args.ply = str(ply_files[0])
            print(f"自动找到点云文件: {args.ply}")
        else:
            print("错误：未找到 PLY 文件")
            return
    
    visualize_cameras_and_pointcloud(args.metadata, args.ply, args.output)


if __name__ == "__main__":
    main()

