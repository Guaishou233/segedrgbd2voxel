# RGB2Voxel

RGB2Voxel 是一个将多视角 RGBD 图像转换为体素网格的完整处理流程。支持语义分割、点云生成和体素化三个主要步骤。

## 目录

- [功能特性](#功能特性)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [数据集结构](#数据集结构)
- [自定义数据转换器](#自定义数据转换器)
- [Pipeline 使用指南](#pipeline-使用指南)
- [可视化工具](#可视化工具)
- [配置文件说明](#配置文件说明)

## 功能特性

- **语义分割**：使用 SAM3 模型对 RGB 图像进行语义分割
- **点云生成**：融合多视角 RGBD 图像生成带语义标签的点云
- **体素化**：将点云转换为固定大小的体素网格
- **可视化**：支持点云和体素的交互式 3D 可视化

## 安装依赖

### 1. 创建虚拟环境（推荐）

```bash
conda create -n rgb2voxel python=3.9
conda activate rgb2voxel
```

### 2. 安装依赖包

```bash
cd rgb2voxel
pip install -r requirements.txt
```

### 3. 依赖列表

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| numpy | >=1.21.0 | 数值计算 |
| opencv-python | >=4.5.0 | 图像处理 |
| pillow | >=8.0.0 | 图像读写 |
| tqdm | >=4.60.0 | 进度条 |
| pyyaml | >=6.0 | 配置文件解析 |
| open3d | >=0.15.0 | 3D 点云处理 |
| torch | >=1.9.0 | 深度学习 |
| torchvision | >=0.10.0 | 图像模型 |
| transformers | >=4.30.0 | SAM3 模型 |
| pycocotools | >=2.0.6 | COCO 格式支持 |

### 4. 下载 SAM3 模型（如需使用语义分割）

将 SAM3 模型文件放置在 `Sam3/model` 目录下。

## 快速开始

### 1. 准备数据

将你的数据集放置在 `data/` 目录下，确保数据格式符合[数据集结构](#数据集结构)要求。

### 2. 配置参数

编辑 `config.yaml` 文件，设置数据集路径和处理参数：

```yaml
dataset:
  base_dir: "/path/to/rgb2voxel/data"
  dataset_name: "your_dataset"
  tasks: []  # 空列表表示处理所有任务
```

### 3. 运行 Pipeline

```bash
# 执行所有步骤
python pipeline.py

# 只执行点云生成
python pipeline.py --steps pointcloud

# 只执行体素化
python pipeline.py --steps voxelization

# 指定配置文件
python pipeline.py --config /path/to/config.yaml
```

### 4. 可视化结果

```bash
# 交互式可视化
python visualization_module.py

# 直接可视化点云
python visualization_module.py -p data/dataset/task_X/POINTCLOUDS/xxx.ply

# 直接可视化体素
python visualization_module.py -v data/dataset/task_X/VOXELS/xxx/
```

## 数据集结构

### 标准数据集结构

Pipeline 期望的数据集结构如下：

```
data/
└── dataset_name/
    └── task_X/
        ├── cam_00/
        │   ├── color/
        │   │   ├── {timestamp}.jpg      # RGB 图像
        │   │   └── ...
        │   ├── depth/
        │   │   ├── {timestamp}.png      # 16位深度图（毫米）
        │   │   └── ...
        │   ├── segmentation/            # 可选，由 pipeline 生成
        │   │   ├── {timestamp}.png      # 语义分割 mask
        │   │   └── {timestamp}_colored.png
        │   └── timestamps.npy           # 时间戳数组
        ├── cam_01/
        │   └── ...
        ├── metadata.json                # 相机参数
        ├── color_map.json               # 颜色映射
        ├── POINTCLOUDS/                 # 点云输出目录
        │   ├── {timestamp}.ply          # 点云文件
        │   └── {timestamp}.json         # 点云元数据
        └── VOXELS/                      # 体素输出目录
            └── {timestamp}/
                ├── voxel_grid.npy       # 体素占用网格
                ├── voxel_colors.npy     # 体素颜色
                ├── voxel_semantics.npy  # 体素语义
                └── voxel_metadata.json  # 体素元数据
```

### metadata.json 格式

```json
{
  "cameras": {
    "cam_00": {
      "width": 640,
      "height": 480,
      "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "extrinsic": [[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3], [0, 0, 0, 1]]
    }
  },
  "intrinsic_scale": 1.0
}
```

### timestamps.npy 格式

```python
# 字典格式，键为时间戳，值为帧索引
{1765521296783: 0, 1765521296784: 1, ...}
```

## 自定义数据转换器

如果你的数据格式与标准格式不同，需要编写自定义转换器。

### 1. 创建转换器类

在 `data_converter/` 目录下创建新文件，例如 `my_converter.py`：

```python
#!/usr/bin/env python3
"""自定义数据转换器示例"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import cv2


class MyDataConverter:
    """将自定义格式转换为 rgb2voxel 兼容格式"""
    
    def __init__(self, source_dir: str, target_dir: str, task_name: Optional[str] = None):
        """
        初始化转换器
        
        Args:
            source_dir: 源数据目录
            target_dir: 目标数据目录
            task_name: 任务名称
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.task_name = task_name
        
        # 加载相机信息
        self.camera_info = self._load_camera_info()
    
    def _load_camera_info(self) -> Dict:
        """加载相机参数"""
        # 根据你的数据格式实现
        pass
    
    def _convert_rgb(self, src_dir: Path, dst_dir: Path) -> List[int]:
        """
        转换 RGB 图像
        
        Returns:
            时间戳列表
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        timestamps = []
        
        # 遍历源图像
        for img_file in sorted(src_dir.glob("*.png")):
            # 提取时间戳（根据你的命名规则）
            timestamp = int(img_file.stem)
            timestamps.append(timestamp)
            
            # 转换为 JPG
            img = cv2.imread(str(img_file))
            cv2.imwrite(str(dst_dir / f"{timestamp}.jpg"), img, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return timestamps
    
    def _convert_depth(self, src_dir: Path, dst_dir: Path, 
                       timestamps: List[int], cam_idx: int) -> Dict:
        """
        转换深度图
        
        注意事项：
        1. 输出格式：16位 PNG，单位毫米
        2. 如果原始深度是欧几里得深度（射线深度），需要转换为 Z 深度
        3. 深度值 0 表示无效
        
        Args:
            src_dir: 源目录
            dst_dir: 目标目录
            timestamps: 时间戳列表
            cam_idx: 相机索引（用于获取内参）
        
        Returns:
            深度统计信息
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取相机内参
        fx, fy, cx, cy = self._get_intrinsics(cam_idx)
        width, height = 640, 480  # 根据实际情况设置
        
        # 如果深度是欧几里得深度，计算转换因子
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        euclidean_to_z_factor = 1.0 / np.sqrt(x_norm**2 + y_norm**2 + 1)
        
        for idx, depth_file in enumerate(sorted(src_dir.glob("*.npy"))):
            depth = np.load(depth_file)
            
            # 转换欧几里得深度到 Z 深度（如果需要）
            z_depth = depth * euclidean_to_z_factor
            
            # 转换为毫米，保存为 16 位 PNG
            depth_mm = (z_depth * 1000).astype(np.uint16)
            timestamp = timestamps[idx]
            cv2.imwrite(str(dst_dir / f"{timestamp}.png"), depth_mm)
        
        return {'min_depth': float(depth.min()), 'max_depth': float(depth.max())}
    
    def _get_intrinsics(self, cam_idx: int):
        """获取相机内参"""
        cam = self.camera_info['cameras'][cam_idx]
        return cam['fx'], cam['fy'], cam['cx'], cam['cy']
    
    def _generate_metadata(self, task_dir: Path, camera_names: List[str]) -> Dict:
        """
        生成 metadata.json
        
        重要字段：
        - cameras: 各相机的宽高、内参、外参
        - intrinsic_scale: 内参缩放因子（通常为 1.0）
        """
        metadata = {
            "cameras": {},
            "intrinsic_scale": 1.0
        }
        
        for idx, cam_name in enumerate(camera_names):
            cam_info = self.camera_info['cameras'][idx]
            
            # 外参矩阵（世界到相机）
            extrinsic = np.array(cam_info['extrinsic_matrix'])
            
            metadata["cameras"][cam_name] = {
                "width": cam_info['width'],
                "height": cam_info['height'],
                "intrinsic": cam_info['intrinsic_matrix'],
                "extrinsic": extrinsic.tolist()
            }
        
        return metadata
    
    def _save_timestamps(self, timestamps: List[int], output_path: Path):
        """保存时间戳"""
        ts_dict = {ts: idx for idx, ts in enumerate(timestamps)}
        np.save(output_path, ts_dict)
    
    def convert(self):
        """执行转换"""
        print(f"开始转换: {self.source_dir} -> {self.target_dir}")
        
        # 创建目标目录
        task_dir = self.target_dir / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        camera_names = []
        
        # 转换每个相机的数据
        for cam_idx in range(len(self.camera_info['cameras'])):
            cam_name = f"cam_{cam_idx:02d}"
            camera_names.append(cam_name)
            cam_dir = task_dir / cam_name
            
            # 转换 RGB
            timestamps = self._convert_rgb(
                self.source_dir / f"camera_{cam_idx}" / "rgb",
                cam_dir / "color"
            )
            
            # 转换深度
            self._convert_depth(
                self.source_dir / f"camera_{cam_idx}" / "depth",
                cam_dir / "depth",
                timestamps,
                cam_idx
            )
            
            # 保存时间戳
            self._save_timestamps(timestamps, cam_dir / "timestamps.npy")
        
        # 生成 metadata.json
        metadata = self._generate_metadata(task_dir, camera_names)
        with open(task_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 创建空目录
        (task_dir / "POINTCLOUDS").mkdir(exist_ok=True)
        (task_dir / "VOXELS").mkdir(exist_ok=True)
        
        print("转换完成!")


if __name__ == "__main__":
    converter = MyDataConverter(
        source_dir="/path/to/source",
        target_dir="/path/to/rgb2voxel/data/my_dataset",
        task_name="task_1"
    )
    converter.convert()
```

### 2. 深度数据注意事项

**欧几里得深度 vs Z 深度**

- **欧几里得深度（射线深度）**：从相机中心到 3D 点的直线距离
- **Z 深度（平面深度）**：沿相机 Z 轴的距离

Open3D 期望的是 Z 深度，如果你的深度数据是欧几里得深度，需要进行转换：

```python
# 转换公式
z_depth = euclidean_depth / sqrt(x_norm² + y_norm² + 1)

# 其中
x_norm = (u - cx) / fx
y_norm = (v - cy) / fy
```

### 3. 外参矩阵格式

外参矩阵应该是 **世界到相机** (World-to-Camera) 的变换矩阵，格式为 4x4：

```
[R | t]   R: 3x3 旋转矩阵
[0 | 1]   t: 3x1 平移向量
```

如果你的数据使用 OpenGL 坐标系（Z 轴向后），需要翻转 Y 和 Z 轴以兼容 Open3D：

```python
# OpenGL -> OpenCV 坐标系转换
extrinsic[1, :3] *= -1  # 翻转 Y 轴
extrinsic[2, :3] *= -1  # 翻转 Z 轴
extrinsic[:3, 1] *= -1
extrinsic[:3, 2] *= -1
```

## Pipeline 使用指南

### 命令行参数

```bash
python pipeline.py [选项]

选项:
  --config, -c    配置文件路径（默认: config.yaml）
  --steps, -s     要执行的步骤（segmentation, pointcloud, voxelization）
  --tasks, -t     要处理的任务名称
  --help, -h      显示帮助信息
```

### 执行步骤

Pipeline 包含三个主要步骤：

| 步骤 | 输入 | 输出 | 说明 |
|------|------|------|------|
| segmentation | RGB 图像 | 语义分割 mask | 使用 SAM3 模型 |
| pointcloud | RGBD + mask | PLY 点云 | 多视角融合 |
| voxelization | 点云 | NPY 体素 | 固定大小网格 |

### 生成的数据

#### 1. 语义分割（segmentation）

```
cam_XX/segmentation/
├── {timestamp}.png          # 原始 mask（灰度，像素值=类别ID）
└── {timestamp}_colored.png  # 彩色可视化
```

#### 2. 点云（pointcloud）

```
POINTCLOUDS/
├── {timestamp}.ply    # PLY 格式点云
│                      # 包含: x, y, z, r, g, b, semantic_r, semantic_g, semantic_b
└── {timestamp}.json   # 元数据
```

#### 3. 体素（voxelization）

```
VOXELS/{timestamp}/
├── voxel_grid.npy        # 布尔数组 (N, N, N)，True 表示占用
├── voxel_colors.npy      # RGB 颜色 (N, N, N, 3)
├── voxel_semantics.npy   # 语义颜色 (N, N, N, 3)
└── voxel_metadata.json   # 元数据
```

**voxel_metadata.json 示例：**

```json
{
  "timestamp": "1765521296785",
  "voxel_size": 64,
  "physical_size": 6.0,
  "voxel_resolution": 0.09375,
  "voxel_origin": {"x": 3.12, "y": -3.64, "z": -2.91},
  "occupied_voxels": 3701
}
```

## 可视化工具

### 交互式模式

```bash
python visualization_module.py
```

菜单选项：
1. 可视化点云文件 (PLY)
2. 可视化体素目录
3. 可视化单个体素文件 (NPY)
4. 退出

### 命令行模式

```bash
# 可视化点云
python visualization_module.py -p /path/to/pointcloud.ply
python visualization_module.py -p /path/to/pointcloud.ply --mode semantic

# 可视化体素目录
python visualization_module.py -v /path/to/VOXELS/timestamp/
python visualization_module.py -v /path/to/VOXELS/timestamp/ --mode semantic

# 可视化单个 NPY 文件
python visualization_module.py -n /path/to/voxel_colors.npy
python visualization_module.py -n /path/to/voxel_semantics.npy --render point
```

### 可视化控制

| 操作 | 功能 |
|------|------|
| 鼠标左键拖动 | 旋转视角 |
| 鼠标滚轮 | 缩放 |
| 鼠标右键拖动 | 平移 |
| R | 重置视角 |
| Q / Esc | 退出 |
| +/- | 调整点大小 |

### 参数说明

| 参数 | 说明 |
|------|------|
| `--mode`, `-m` | 显示模式：`rgb`（原始颜色）或 `semantic`（语义颜色） |
| `--render`, `-r` | 体素渲染模式：`cube`（立方体）或 `point`（点云） |
| `--point-size` | 点云渲染时的点大小 |
| `--no-coord-frame` | 不显示坐标系 |
| `--no-bbox` | 不显示边界框 |

## 配置文件说明

配置文件 `config.yaml` 分为以下几个部分：

### 数据集配置

```yaml
dataset:
  base_dir: "/path/to/data"     # 数据根目录
  dataset_name: "my_dataset"    # 数据集名称
  tasks: []                     # 要处理的任务，空表示全部
```

### 语义分割配置

```yaml
segmentation:
  model_path: "Sam3/model"      # SAM3 模型路径
  device: "cuda"                # 设备：cuda 或 cpu
  targets:                      # 分割目标
    - label: "clothe"
      prompt: "clothe"
      id: 1
  background:
    label: "background"
    id: 2
  threshold: 0.3
  overwrite: true
```

### 点云生成配置

```yaml
pointcloud:
  min_depth: 0.0                # 最小深度（米）
  max_depth: 10.0               # 最大深度（米）
  depth_scale: 1000.0           # 深度缩放（毫米->米）
  max_time_diff_ms: 50          # 多视角时间戳匹配阈值
  filter:
    voxel_size: 0.0001          # 下采样体素大小
    num_neighbors: 10           # 离群点过滤邻居数
    std_ratio: 2.0              # 标准差比例
```

### 体素化配置

```yaml
voxelization:
  voxel_size: 64                # 体素网格大小 (N x N x N)
  physical_size: 6.0            # 物理空间大小（米）
  overwrite: true
```

**重要**：`physical_size` 需要大于等于点云的最大跨度，否则部分点云会被裁剪。

### Pipeline 配置

```yaml
pipeline:
  steps: []                     # 要执行的步骤，空表示全部
  save_color_map: true          # 保存颜色映射
  show_progress: true           # 显示进度条
  log_level: "INFO"             # 日志级别
```

## 常见问题

### Q: 点云生成失败，输出 0 个点？

**可能原因：**
1. `max_depth` 设置过小，深度被截断
2. 深度数据格式不正确（欧几里得深度未转换）
3. 相机内参或外参配置错误

**解决方法：**
1. 检查深度图的实际范围，调整 `min_depth` 和 `max_depth`
2. 确认深度数据是 Z 深度而非欧几里得深度
3. 验证 `metadata.json` 中的相机参数

### Q: 体素化结果不完整？

**可能原因：**
`physical_size` 小于点云的最大跨度。

**解决方法：**
1. 检查点云的边界范围
2. 增大 `voxelization.physical_size`

### Q: 多视角融合时边界有弧形？

**原因：**
深度数据是欧几里得深度，未转换为 Z 深度。

**解决方法：**
在数据转换器中添加深度类型转换。

## License

MIT License

