#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块
提供点云和体素网格的交互式可视化功能
"""

import os
import json
import struct
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import open3d as o3d
    USE_OPEN3D = True
except ImportError:
    USE_OPEN3D = False
    raise ImportError("需要安装open3d: pip install open3d")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    USE_TKINTER = True
except ImportError:
    USE_TKINTER = False
    print("警告: tkinter不可用，将使用命令行方式选择文件")


class PointCloudVisualizer:
    """点云可视化器"""
    
    def __init__(self):
        """初始化点云可视化器"""
        pass
    
    def load_pointcloud(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载带语义信息的点云
        
        Args:
            ply_path: PLY文件路径
        
        Returns:
            points: Nx3 点云坐标
            colors: Nx3 RGB颜色 (0-255)
            semantics: Nx3 语义颜色 (0-255)
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
            has_semantics = False
            for line in header_lines:
                if b'element vertex' in line:
                    num_points = int(line.split()[-1])
                if b'semantic' in line:
                    has_semantics = True
            
            # 读取点云数据
            for _ in range(num_points):
                # x, y, z (float)
                x, y, z = struct.unpack('<fff', f.read(12))
                # r, g, b (uchar)
                r, g, b = struct.unpack('<BBB', f.read(3))
                
                # semantic_r, semantic_g, semantic_b (uchar) - 如果有
                if has_semantics:
                    semantic_r, semantic_g, semantic_b = struct.unpack('<BBB', f.read(3))
                else:
                    semantic_r, semantic_g, semantic_b = 128, 128, 128
                
                # 过滤无效点
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or
                        np.isinf(x) or np.isinf(y) or np.isinf(z)):
                    points.append([x, y, z])
                    colors.append([r, g, b])
                    semantics.append([semantic_r, semantic_g, semantic_b])
        
        return np.array(points), np.array(colors), np.array(semantics)
    
    def visualize(self, ply_path: str, mode: str = 'rgb', 
                  point_size: float = 2.0, 
                  show_coordinate_frame: bool = True) -> None:
        """
        可视化点云
        
        Args:
            ply_path: PLY文件路径
            mode: 显示模式 - 'rgb' 显示原始颜色, 'semantic' 显示语义颜色
            point_size: 点大小
            show_coordinate_frame: 是否显示坐标系
        """
        print(f"加载点云: {ply_path}")
        
        # 加载点云
        points, colors, semantics = self.load_pointcloud(ply_path)
        
        if len(points) == 0:
            print("错误: 点云为空")
            return
        
        print(f"点云大小: {len(points)} 个点")
        print(f"边界框: min={points.min(axis=0)}, max={points.max(axis=0)}")
        
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 根据模式设置颜色
        if mode == 'semantic':
            pcd.colors = o3d.utility.Vector3dVector(semantics / 255.0)
            window_title = f"点云可视化 [语义] - {os.path.basename(ply_path)}"
        else:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            window_title = f"点云可视化 [RGB] - {os.path.basename(ply_path)}"
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, width=1280, height=720)
        
        # 添加点云
        vis.add_geometry(pcd)
        
        # 添加坐标系
        if show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )
            vis.add_geometry(coord_frame)
        
        # 设置渲染选项
        render_opt = vis.get_render_option()
        render_opt.point_size = point_size
        render_opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
        
        # 设置视角
        view_ctrl = vis.get_view_control()
        view_ctrl.set_zoom(0.8)
        
        print("\n可视化控制:")
        print("  鼠标左键: 旋转")
        print("  鼠标滚轮: 缩放")
        print("  鼠标右键: 平移")
        print("  Q/Esc: 退出")
        print("  R: 重置视角")
        print("  +/-: 调整点大小")
        
        # 运行可视化
        vis.run()
        vis.destroy_window()


class VoxelVisualizer:
    """体素网格可视化器"""
    
    def __init__(self):
        """初始化体素可视化器"""
        pass
    
    def load_voxel_data(self, voxel_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        加载体素数据
        
        Args:
            voxel_dir: 体素数据目录
        
        Returns:
            voxel_grid: 体素网格 (bool)
            voxel_colors: 体素颜色
            voxel_semantics: 体素语义
            metadata: 元数据
        """
        # 加载体素网格
        grid_path = os.path.join(voxel_dir, "voxel_grid.npy")
        colors_path = os.path.join(voxel_dir, "voxel_colors.npy")
        semantics_path = os.path.join(voxel_dir, "voxel_semantics.npy")
        metadata_path = os.path.join(voxel_dir, "voxel_metadata.json")
        
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"找不到体素网格文件: {grid_path}")
        
        voxel_grid = np.load(grid_path)
        
        # 加载颜色（可选）
        if os.path.exists(colors_path):
            voxel_colors = np.load(colors_path)
        else:
            voxel_colors = np.ones((*voxel_grid.shape, 3), dtype=np.uint8) * 128
        
        # 加载语义（可选）
        if os.path.exists(semantics_path):
            voxel_semantics = np.load(semantics_path)
        else:
            voxel_semantics = np.ones((*voxel_grid.shape, 3), dtype=np.uint8) * 128
        
        # 加载元数据（可选）
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return voxel_grid, voxel_colors, voxel_semantics, metadata
    
    def voxel_to_mesh(self, voxel_grid: np.ndarray, voxel_colors: np.ndarray,
                      voxel_origin: np.ndarray, voxel_resolution: float,
                      use_cubes: bool = True) -> o3d.geometry.TriangleMesh:
        """
        将体素网格转换为网格模型
        
        Args:
            voxel_grid: 体素占用网格
            voxel_colors: 体素颜色
            voxel_origin: 体素原点
            voxel_resolution: 体素分辨率
            use_cubes: 是否使用立方体（True）或点（False）
        
        Returns:
            mesh: Open3D网格模型
        """
        # 找到所有占用的体素
        occupied = np.argwhere(voxel_grid)
        
        if len(occupied) == 0:
            return None
        
        if use_cubes:
            # 创建立方体网格
            meshes = []
            
            for idx in occupied:
                x_idx, y_idx, z_idx = idx
                
                # 计算体素中心位置
                center = voxel_origin + (np.array([x_idx, y_idx, z_idx]) + 0.5) * voxel_resolution
                
                # 创建立方体
                cube = o3d.geometry.TriangleMesh.create_box(
                    width=voxel_resolution * 0.95,
                    height=voxel_resolution * 0.95,
                    depth=voxel_resolution * 0.95
                )
                
                # 移动到正确位置
                cube.translate(center - np.array([voxel_resolution * 0.475] * 3))
                
                # 设置颜色
                color = voxel_colors[x_idx, y_idx, z_idx] / 255.0
                cube.paint_uniform_color(color)
                
                meshes.append(cube)
            
            # 合并所有立方体
            if len(meshes) > 0:
                combined_mesh = meshes[0]
                for mesh in meshes[1:]:
                    combined_mesh += mesh
                return combined_mesh
            else:
                return None
        else:
            # 使用点云表示
            points = []
            colors = []
            
            for idx in occupied:
                x_idx, y_idx, z_idx = idx
                center = voxel_origin + (np.array([x_idx, y_idx, z_idx]) + 0.5) * voxel_resolution
                points.append(center)
                colors.append(voxel_colors[x_idx, y_idx, z_idx] / 255.0)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            return pcd
    
    def visualize(self, voxel_dir: str, mode: str = 'rgb',
                  render_mode: str = 'cube',
                  show_coordinate_frame: bool = True,
                  show_bounding_box: bool = True) -> None:
        """
        可视化体素网格
        
        Args:
            voxel_dir: 体素数据目录
            mode: 显示模式 - 'rgb' 显示原始颜色, 'semantic' 显示语义颜色
            render_mode: 渲染模式 - 'cube' 立方体, 'point' 点云
            show_coordinate_frame: 是否显示坐标系
            show_bounding_box: 是否显示边界框
        """
        print(f"加载体素数据: {voxel_dir}")
        
        # 加载数据
        voxel_grid, voxel_colors, voxel_semantics, metadata = self.load_voxel_data(voxel_dir)
        
        # 打印信息
        print(f"体素网格大小: {voxel_grid.shape}")
        print(f"占用体素数: {np.sum(voxel_grid)}")
        print(f"占用率: {np.sum(voxel_grid) / voxel_grid.size * 100:.2f}%")
        
        if metadata:
            print(f"物理尺寸: {metadata.get('physical_size', 'N/A')} 米")
            print(f"体素分辨率: {metadata.get('voxel_resolution', 'N/A')} 米")
        
        # 获取元数据
        voxel_resolution = metadata.get('voxel_resolution', 0.02)
        voxel_origin_dict = metadata.get('voxel_origin', {'x': 0, 'y': 0, 'z': 0})
        voxel_origin = np.array([
            voxel_origin_dict.get('x', 0),
            voxel_origin_dict.get('y', 0),
            voxel_origin_dict.get('z', 0)
        ])
        
        # 选择颜色模式
        if mode == 'semantic':
            colors_to_use = voxel_semantics
            window_title = f"体素可视化 [语义] - {os.path.basename(voxel_dir)}"
        else:
            colors_to_use = voxel_colors
            window_title = f"体素可视化 [RGB] - {os.path.basename(voxel_dir)}"
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, width=1280, height=720)
        
        # 根据渲染模式创建几何体
        use_cubes = (render_mode == 'cube')
        
        print(f"\n正在生成{'立方体' if use_cubes else '点云'}网格...")
        geometry = self.voxel_to_mesh(voxel_grid, colors_to_use, 
                                      voxel_origin, voxel_resolution, use_cubes)
        
        if geometry is None:
            print("错误: 没有占用的体素")
            return
        
        vis.add_geometry(geometry)
        
        # 添加坐标系
        if show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )
            vis.add_geometry(coord_frame)
        
        # 添加边界框
        if show_bounding_box:
            physical_size = metadata.get('physical_size', voxel_resolution * voxel_grid.shape[0])
            bbox_min = voxel_origin
            bbox_max = voxel_origin + physical_size
            
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=bbox_min,
                max_bound=bbox_max
            )
            bbox.color = (0.5, 0.5, 0.5)
            vis.add_geometry(bbox)
        
        # 设置渲染选项
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.1])
        
        if not use_cubes:
            render_opt.point_size = 5.0
        
        # 设置视角
        view_ctrl = vis.get_view_control()
        view_ctrl.set_zoom(0.8)
        
        print("\n可视化控制:")
        print("  鼠标左键: 旋转")
        print("  鼠标滚轮: 缩放")
        print("  鼠标右键: 平移")
        print("  Q/Esc: 退出")
        print("  R: 重置视角")
        
        # 运行可视化
        vis.run()
        vis.destroy_window()


def select_pointcloud_file() -> Optional[str]:
    """
    使用文件对话框选择点云文件
    
    Returns:
        选择的文件路径，如果取消则返回None
    """
    if USE_TKINTER:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        file_path = filedialog.askopenfilename(
            title="选择点云文件",
            filetypes=[
                ("PLY文件", "*.ply"),
                ("所有文件", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            return file_path
        return None
    else:
        # 命令行输入
        file_path = input("请输入点云文件路径 (.ply): ").strip()
        if file_path and os.path.exists(file_path):
            return file_path
        return None


def select_voxel_directory() -> Optional[str]:
    """
    使用文件对话框选择体素数据目录
    
    Returns:
        选择的目录路径，如果取消则返回None
    """
    if USE_TKINTER:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        dir_path = filedialog.askdirectory(
            title="选择体素数据目录 (包含 voxel_grid.npy 的文件夹)"
        )
        
        root.destroy()
        
        if dir_path:
            # 验证目录包含必需的文件
            if os.path.exists(os.path.join(dir_path, "voxel_grid.npy")):
                return dir_path
            else:
                print(f"警告: 目录 {dir_path} 不包含 voxel_grid.npy 文件")
                return None
        return None
    else:
        # 命令行输入
        dir_path = input("请输入体素数据目录路径: ").strip()
        if dir_path and os.path.exists(dir_path):
            if os.path.exists(os.path.join(dir_path, "voxel_grid.npy")):
                return dir_path
            else:
                print(f"警告: 目录 {dir_path} 不包含 voxel_grid.npy 文件")
        return None


def interactive_menu():
    """交互式菜单"""
    print("\n" + "=" * 60)
    print("RGB2Voxel 可视化工具")
    print("=" * 60)
    
    while True:
        print("\n请选择操作:")
        print("  1. 可视化点云文件 (PLY)")
        print("  2. 可视化体素网格 (NPY)")
        print("  3. 退出")
        
        choice = input("\n请输入选项 (1/2/3): ").strip()
        
        if choice == '1':
            # 点云可视化
            print("\n选择点云文件...")
            ply_path = select_pointcloud_file()
            
            if ply_path:
                print(f"\n选择的文件: {ply_path}")
                
                # 选择显示模式
                print("\n显示模式:")
                print("  1. RGB颜色")
                print("  2. 语义颜色")
                mode_choice = input("请选择 (1/2, 默认1): ").strip()
                mode = 'semantic' if mode_choice == '2' else 'rgb'
                
                # 可视化
                visualizer = PointCloudVisualizer()
                visualizer.visualize(ply_path, mode=mode)
            else:
                print("未选择文件")
        
        elif choice == '2':
            # 体素可视化
            print("\n选择体素数据目录...")
            voxel_dir = select_voxel_directory()
            
            if voxel_dir:
                print(f"\n选择的目录: {voxel_dir}")
                
                # 选择显示模式
                print("\n显示模式:")
                print("  1. RGB颜色")
                print("  2. 语义颜色")
                mode_choice = input("请选择 (1/2, 默认1): ").strip()
                mode = 'semantic' if mode_choice == '2' else 'rgb'
                
                # 选择渲染模式
                print("\n渲染模式:")
                print("  1. 立方体 (更直观但较慢)")
                print("  2. 点云 (更快)")
                render_choice = input("请选择 (1/2, 默认1): ").strip()
                render_mode = 'point' if render_choice == '2' else 'cube'
                
                # 可视化
                visualizer = VoxelVisualizer()
                visualizer.visualize(voxel_dir, mode=mode, render_mode=render_mode)
            else:
                print("未选择目录")
        
        elif choice == '3':
            print("\n退出程序")
            break
        
        else:
            print("无效选项，请重新选择")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RGB2Voxel 可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python visualization_module.py
  
  # 可视化点云文件
  python visualization_module.py --pointcloud /path/to/pointcloud.ply
  
  # 可视化点云的语义颜色
  python visualization_module.py --pointcloud /path/to/pointcloud.ply --mode semantic
  
  # 可视化体素网格
  python visualization_module.py --voxel /path/to/voxel_dir
  
  # 可视化体素网格的语义颜色，使用点云渲染
  python visualization_module.py --voxel /path/to/voxel_dir --mode semantic --render point
        """
    )
    
    parser.add_argument(
        "--pointcloud", "-p",
        type=str,
        default=None,
        help="点云文件路径 (.ply)"
    )
    
    parser.add_argument(
        "--voxel", "-v",
        type=str,
        default=None,
        help="体素数据目录路径"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=['rgb', 'semantic'],
        default='rgb',
        help="显示模式: rgb(原始颜色) 或 semantic(语义颜色)"
    )
    
    parser.add_argument(
        "--render", "-r",
        type=str,
        choices=['cube', 'point'],
        default='cube',
        help="体素渲染模式: cube(立方体) 或 point(点云)"
    )
    
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="点云渲染时的点大小"
    )
    
    parser.add_argument(
        "--no-coord-frame",
        action="store_true",
        help="不显示坐标系"
    )
    
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="不显示边界框（仅体素可视化）"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定文件，进入交互模式
    if args.pointcloud is None and args.voxel is None:
        interactive_menu()
        return
    
    # 可视化点云
    if args.pointcloud:
        if not os.path.exists(args.pointcloud):
            print(f"错误: 文件不存在: {args.pointcloud}")
            return
        
        visualizer = PointCloudVisualizer()
        visualizer.visualize(
            args.pointcloud,
            mode=args.mode,
            point_size=args.point_size,
            show_coordinate_frame=not args.no_coord_frame
        )
    
    # 可视化体素
    if args.voxel:
        if not os.path.exists(args.voxel):
            print(f"错误: 目录不存在: {args.voxel}")
            return
        
        if not os.path.exists(os.path.join(args.voxel, "voxel_grid.npy")):
            print(f"错误: 目录不包含 voxel_grid.npy: {args.voxel}")
            return
        
        visualizer = VoxelVisualizer()
        visualizer.visualize(
            args.voxel,
            mode=args.mode,
            render_mode=args.render,
            show_coordinate_frame=not args.no_coord_frame,
            show_bounding_box=not args.no_bbox
        )


if __name__ == "__main__":
    main()


