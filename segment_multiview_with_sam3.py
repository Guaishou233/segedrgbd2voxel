#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用SAM3对多视角RGB图像进行语义分割，分割robot arm，其他归类为others
处理6cam_dataset格式的多视角数据
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import pycocotools.mask as mask_util
import glob

# 添加 Sam3 目录到 Python 路径，以便能够导入 sam3 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_dir = os.path.join(current_dir, "Sam3")
if sam3_dir not in sys.path:
    sys.path.insert(0, sam3_dir)

# 检查是否可以使用transformers库
try:
    from transformers import Sam3Processor, Sam3Model
    USE_TRANSFORMERS = True
except ImportError:
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        USE_TRANSFORMERS = False
    except ImportError:
        raise ImportError("需要安装transformers或sam3库")

def load_dict_npy(file_name: str):
    """加载字典格式的numpy文件"""
    return np.load(file_name, allow_pickle=True).item()

def numpy_to_list(obj):
    """将numpy数组转换为列表，以便JSON序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj

def encode_mask_to_rle(mask):
    """将mask编码为RLE格式（COCO格式）"""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def segment_image_with_sam3(model, processor, image_path, text_prompt, device):
    """
    使用SAM3对图像进行分割
    
    Args:
        model: SAM3模型
        processor: SAM3处理器
        image_path: 图像路径
        text_prompt: 文本提示（如"robot arm"）
        device: 设备（cuda或cpu）
    
    Returns:
        masks: 分割mask列表
        boxes: 边界框列表
        scores: 置信度分数列表
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    if USE_TRANSFORMERS:
        # 使用transformers库
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 后处理结果
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3,  # 降低阈值以获取更多结果
            mask_threshold=0.3,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results["masks"]
        boxes = results["boxes"]
        scores = results["scores"]
    else:
        # 使用sam3原生库
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
    
    return masks, boxes, scores

def create_segmentation_mask(masks, boxes, scores, image_size, threshold=0.3):
    """
    创建分割mask，robot arm为类别1，其他区域为类别2（others）
    
    Args:
        masks: mask列表
        boxes: 边界框列表
        scores: 置信度分数列表
        image_size: 图像尺寸 (width, height)
        threshold: 置信度阈值
    
    Returns:
        seg_mask: 分割mask，值为1表示robot arm，值为2表示others
    """
    width, height = image_size
    seg_mask = np.ones((height, width), dtype=np.uint8) * 2  # 默认都是others
    
    # 创建robot arm的mask（类别1）
    robot_arm_mask = np.zeros((height, width), dtype=bool)
    
    if len(masks) == 0:
        return seg_mask
    
    if USE_TRANSFORMERS:
        # transformers返回的masks是tensor
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score >= threshold:
                mask_np = mask.cpu().numpy()
                if len(mask_np.shape) > 2:
                    mask_np = mask_np.squeeze()
                
                # 确保是二值mask
                if mask_np.max() <= 1.0:
                    mask_bool = mask_np > threshold
                else:
                    mask_bool = mask_np > 128
                
                if mask_bool.shape != (height, width):
                    # 需要resize
                    mask_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))
                    mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                    mask_bool = np.array(mask_pil) > 128
                
                robot_arm_mask = robot_arm_mask | mask_bool
    else:
        # sam3原生库返回的格式
        for mask, score in zip(masks, scores):
            if score >= threshold:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                
                if len(mask_np.shape) > 2:
                    mask_np = mask_np.squeeze()
                
                if mask_np.max() <= 1.0:
                    mask_bool = mask_np > threshold
                else:
                    mask_bool = mask_np > 128
                
                if mask_bool.shape != (height, width):
                    mask_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))
                    mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                    mask_bool = np.array(mask_pil) > 128
                
                robot_arm_mask = robot_arm_mask | mask_bool
    
    # 设置robot arm区域为类别1
    seg_mask[robot_arm_mask] = 1
    
    return seg_mask

def convert_to_coco_annotations(masks, boxes, scores, image_id, image_size, category_id, threshold=0.3):
    """
    将分割结果转换为COCO格式的annotations
    
    Args:
        masks: mask列表
        boxes: 边界框列表
        scores: 置信度分数列表
        image_id: 图像ID
        image_size: 图像尺寸 (width, height)
        category_id: 类别ID（1表示robot arm）
        threshold: 置信度阈值
    
    Returns:
        annotations: COCO格式的annotation列表
    """
    annotations = []
    width, height = image_size
    
    if len(masks) == 0:
        return annotations
    
    if USE_TRANSFORMERS:
        for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            if score < threshold:
                continue
            
            # 转换mask为numpy数组
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()
            
            # 确保mask是二值化的
            if mask_np.max() <= 1.0:
                mask_bool = mask_np > threshold
            else:
                mask_bool = mask_np > 128
            
            # Resize mask到原始图像尺寸
            if mask_bool.shape != (height, width):
                mask_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))
                mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                mask_bool = np.array(mask_pil) > 128
            
            # 计算边界框
            if isinstance(box, torch.Tensor):
                box_np = box.cpu().numpy()
            else:
                box_np = np.array(box)
            
            # 如果box是xyxy格式，转换为xywh
            if len(box_np) >= 4:
                x1, y1, x2, y2 = box_np[:4]
                # 确保坐标在图像范围内
                x1 = max(0, min(float(x1), width))
                y1 = max(0, min(float(y1), height))
                x2 = max(0, min(float(x2), width))
                y2 = max(0, min(float(y2), height))
                bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                bbox = [0.0, 0.0, float(width), float(height)]
            
            # 计算面积
            area = float(mask_bool.sum())
            
            if area == 0:
                continue
            
            # 编码mask为RLE
            try:
                rle = encode_mask_to_rle(mask_bool)
            except Exception as e:
                print(f"Warning: Failed to encode mask: {e}")
                continue
            
            annotation = {
                "id": len(annotations) + 1,  # 临时ID，后续会重新编号
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": float(score.item() if isinstance(score, torch.Tensor) else score)
            }
            
            annotations.append(annotation)
    else:
        # sam3原生库格式
        for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            if score < threshold:
                continue
            
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()
            
            mask_bool = mask_np > threshold if mask_np.max() <= 1.0 else mask_np > 128
            
            if mask_bool.shape != (height, width):
                mask_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))
                mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                mask_bool = np.array(mask_pil) > 128
            
            if isinstance(box, torch.Tensor):
                box_np = box.cpu().numpy()
            else:
                box_np = np.array(box)
            
            if len(box_np) >= 4:
                x1, y1, x2, y2 = box_np[:4]
                x1 = max(0, min(float(x1), width))
                y1 = max(0, min(float(y1), height))
                x2 = max(0, min(float(x2), width))
                y2 = max(0, min(float(y2), height))
                bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                bbox = [0.0, 0.0, float(width), float(height)]
            
            area = float(mask_bool.sum())
            if area == 0:
                continue
            
            try:
                rle = encode_mask_to_rle(mask_bool)
            except Exception as e:
                print(f"Warning: Failed to encode mask: {e}")
                continue
            
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": float(score.item() if isinstance(score, torch.Tensor) else score)
            }
            
            annotations.append(annotation)
    
    return annotations

def load_camera_calibration(calib_dir, cam_serial):
    """
    加载相机标定信息
    
    Args:
        calib_dir: 标定文件夹路径
        cam_serial: 相机序列号（如'036422060909'）
    
    Returns:
        intrinsic: 相机内参矩阵 (3x3或3x4)
        extrinsic: 相机外参矩阵 (4x4)
    """
    try:
        intrinsics_dict = load_dict_npy(os.path.join(calib_dir, "intrinsics.npy"))
        extrinsics_dict = load_dict_npy(os.path.join(calib_dir, "extrinsics.npy"))
        
        # 获取该相机的内参和外参
        intrinsic = intrinsics_dict.get(cam_serial, None)
        extrinsic = extrinsics_dict.get(cam_serial, None)
        
        if intrinsic is None or extrinsic is None:
            print(f"Warning: 无法找到相机 {cam_serial} 的标定信息")
            return None, None
        
        # 如果extrinsic是列表，取第一个元素
        if isinstance(extrinsic, (list, np.ndarray)) and len(extrinsic) > 0:
            if isinstance(extrinsic[0], np.ndarray):
                extrinsic = extrinsic[0]
        
        return intrinsic, extrinsic
    except Exception as e:
        print(f"Error loading calibration for {cam_serial}: {e}")
        return None, None

def process_cam_folder(cam_dir, cam_serial, calib_dir, model, processor, device):
    """
    处理单个相机文件夹
    
    Args:
        cam_dir: 相机文件夹路径
        cam_serial: 相机序列号
        calib_dir: 标定文件夹路径
        model: SAM3模型
        processor: SAM3处理器
        device: 设备
    
    Returns:
        coco_data: COCO格式的数据字典
    """
    # 创建segmentation文件夹
    seg_dir = os.path.join(cam_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)
    
    # 获取RGB图像列表
    color_dir = os.path.join(cam_dir, "color")
    depth_dir = os.path.join(cam_dir, "depth")
    
    if not os.path.exists(color_dir):
        print(f"Warning: {color_dir} 不存在")
        return None
    
    # 获取所有RGB图像文件
    rgb_files = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
    if len(rgb_files) == 0:
        rgb_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    
    if len(rgb_files) == 0:
        print(f"Warning: {color_dir} 中没有找到RGB图像")
        return None
    
    # 加载相机标定信息
    intrinsic, extrinsic = load_camera_calibration(calib_dir, cam_serial)
    
    # 加载时间戳（如果存在）
    timestamps_dict = None
    timestamps_path = os.path.join(cam_dir, "timestamps.npy")
    if os.path.exists(timestamps_path):
        try:
            timestamps_data = np.load(timestamps_path, allow_pickle=True)
            # 处理不同格式的时间戳
            if isinstance(timestamps_data, np.ndarray):
                if timestamps_data.ndim == 0:
                    # 标量，尝试获取字典
                    timestamps_dict = timestamps_data.item()
                    if not isinstance(timestamps_dict, dict):
                        timestamps_dict = None
                else:
                    # 数组格式，转换为列表
                    timestamps_list = timestamps_data.flatten().tolist()
                    timestamps_dict = {"color": timestamps_list, "depth": timestamps_list}
            else:
                timestamps_dict = timestamps_data if isinstance(timestamps_data, dict) else None
        except Exception as e:
            print(f"  Warning: 无法加载时间戳文件: {e}")
            timestamps_dict = None
    
    # 初始化COCO数据结构
    coco_data = {
        "info": {
            "description": f"Multi-view semantic segmentation dataset for camera {cam_serial}",
            "version": "1.0",
            "camera_serial": cam_serial,
            "camera_intrinsic": numpy_to_list(intrinsic) if intrinsic is not None else None,
            "camera_extrinsic": numpy_to_list(extrinsic) if extrinsic is not None else None,
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "robot arm", "supercategory": "object"},
            {"id": 2, "name": "others", "supercategory": "object"}
        ]
    }
    
    # 处理每张RGB图像
    image_id = 1
    annotation_id = 1
    
    print(f"  正在处理 {len(rgb_files)} 张RGB图像...")
    
    for idx, rgb_path in enumerate(tqdm(rgb_files, desc=f"  分割 {cam_serial}", leave=False)):
        rgb_filename = os.path.basename(rgb_path)
        rgb_name_without_ext = os.path.splitext(rgb_filename)[0]
        
        # 尝试从文件名提取时间戳（文件名通常是时间戳）
        color_timestamp = None
        try:
            # 假设文件名是时间戳（如 1631270649099.jpg）
            color_timestamp = int(rgb_name_without_ext)
        except ValueError:
            pass
        
        # 检查对应的深度图是否存在
        depth_filename = rgb_filename.replace(".jpg", ".png").replace(".jpeg", ".png")
        depth_path = os.path.join(depth_dir, depth_filename)
        depth_exists = os.path.exists(depth_path)
        
        # 获取深度图时间戳（从文件名或timestamps字典）
        depth_timestamp = None
        if depth_exists:
            try:
                depth_name_without_ext = os.path.splitext(depth_filename)[0]
                depth_timestamp = int(depth_name_without_ext)
            except ValueError:
                pass
        
        # 从timestamps字典中获取对应的时间戳（如果存在）
        if timestamps_dict is not None:
            try:
                if "color" in timestamps_dict and idx < len(timestamps_dict["color"]):
                    color_timestamp = timestamps_dict["color"][idx]
                if "depth" in timestamps_dict and idx < len(timestamps_dict["depth"]):
                    depth_timestamp = timestamps_dict["depth"][idx]
            except (TypeError, IndexError, KeyError):
                pass
        
        # segmentation的时间戳与color相同（因为是同时生成的）
        segmentation_timestamp = color_timestamp
        
        # 构建timestamp字典
        timestamp_info = {
            "color": color_timestamp,
            "depth": depth_timestamp,
            "segmentation": segmentation_timestamp
        }
        
        # 获取图像尺寸
        try:
            image = Image.open(rgb_path)
            width, height = image.size
        except Exception as e:
            print(f"Warning: 无法读取图像 {rgb_path}: {e}")
            continue
        
        # 使用SAM3进行分割
        try:
            masks, boxes, scores = segment_image_with_sam3(
                model, processor, rgb_path, "robot arm", device
            )
        except Exception as e:
            print(f"Warning: 分割失败 {rgb_path}: {e}")
            continue
        
        # 创建分割mask
        image_size = (width, height)
        seg_mask = create_segmentation_mask(masks, boxes, scores, image_size)
        
        # 保存分割mask
        seg_filename = rgb_filename.replace(".jpg", ".png").replace(".jpeg", ".png")
        seg_path = os.path.join(seg_dir, seg_filename)
        # 将类别值映射到可视化范围：robot arm(1) -> 255(白色), others(2) -> 128(中灰色)
        # 使用更大的颜色区分度，避免图像看起来全黑
        vis_mask = np.zeros_like(seg_mask, dtype=np.uint8)
        vis_mask[seg_mask == 1] = 255  # robot arm显示为白色
        vis_mask[seg_mask == 2] = 128  # others显示为中灰色（增加区分度）
        seg_image = Image.fromarray(vis_mask, mode='L')
        seg_image.save(seg_path)
        
        # 添加图像信息到COCO数据
        image_info = {
            "id": image_id,
            "file_name": rgb_filename,
            "width": width,
            "height": height,
            "rgb_path": os.path.join("color", rgb_filename),
            "depth_path": os.path.join("depth", depth_filename) if depth_exists else None,
            "segmentation_path": os.path.join("segmentation", seg_filename),
            "timestamp": timestamp_info
        }
        coco_data["images"].append(image_info)
        
        # 转换为COCO格式的annotations（只保存robot arm的annotations）
        annotations = convert_to_coco_annotations(
            masks, boxes, scores, image_id, image_size, category_id=1, threshold=0.3
        )
        
        # 重新分配annotation ID
        for ann in annotations:
            ann["id"] = annotation_id
            annotation_id += 1
            coco_data["annotations"].append(ann)
        
        image_id += 1
    
    return coco_data

def process_multiview_dataset(dataset_base_dir, model_path, device="cuda"):
    """
    处理多视角数据集
    
    Args:
        dataset_base_dir: 数据集基础目录（包含6cam_dataset文件夹）
        model_path: SAM3模型路径
        device: 设备
    """
    dataset_dir = os.path.join(dataset_base_dir, "6cam_dataset")
    calib_dir = os.path.join(dataset_base_dir, "6cam_dataset", "calib")
    
    if not os.path.exists(dataset_dir):
        raise ValueError(f"数据集目录不存在: {dataset_dir}")
    
    # 加载SAM3模型
    print("正在加载SAM3模型...")
    if USE_TRANSFORMERS:
        model = Sam3Model.from_pretrained(model_path).to(device)
        processor = Sam3Processor.from_pretrained(model_path)
    else:
        # 处理模型路径：build_sam3_image_model 的第一个参数是 bpe_path，不是 model_path
        # 如果 model_path 是目录，查找其中的 sam3.pt 文件作为 checkpoint_path
        # 如果 model_path 是文件，直接作为 checkpoint_path
        checkpoint_path = None
        if os.path.isdir(model_path):
            # 查找目录中的 sam3.pt 文件
            sam3_pt_path = os.path.join(model_path, "sam3.pt")
            if os.path.exists(sam3_pt_path):
                checkpoint_path = sam3_pt_path
                print(f"找到本地模型检查点: {checkpoint_path}")
            else:
                print(f"警告: 在 {model_path} 中未找到 sam3.pt，将从 HuggingFace 下载模型")
        elif os.path.isfile(model_path):
            # 如果是文件，直接作为检查点路径
            checkpoint_path = model_path
            print(f"使用模型检查点: {checkpoint_path}")
        else:
            print(f"警告: {model_path} 不存在，将从 HuggingFace 下载模型")
        
        # 调用 build_sam3_image_model
        # bpe_path=None 会使用默认路径，checkpoint_path 用于加载本地模型
        model = build_sam3_image_model(
            bpe_path=None,  # 使用默认的 BPE 路径
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=(checkpoint_path is None)  # 如果没有本地检查点，从 HF 下载
        )
        processor = Sam3Processor(model)
    
    model.eval()
    
    # 查找所有task文件夹
    task_folders = [d for d in os.listdir(dataset_dir) 
                    if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("task_")]
    
    if len(task_folders) == 0:
        print(f"Warning: 在 {dataset_dir} 中没有找到task文件夹")
        return
    
    print(f"找到 {len(task_folders)} 个task文件夹")
    
    # 处理每个task文件夹
    for task_folder in task_folders:
        task_path = os.path.join(dataset_dir, task_folder)
        print(f"\n处理task: {task_folder}")
        
        # 读取metadata.json获取calib信息
        metadata_path = os.path.join(task_path, "metadata.json")
        calib_timestamp = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    calib_timestamp = metadata.get("calib", None)
            except:
                pass
        
        # 确定标定文件夹
        if calib_timestamp is not None:
            task_calib_dir = os.path.join(calib_dir, str(calib_timestamp))
        else:
            # 如果没有找到calib信息，尝试使用最新的标定文件夹
            calib_folders = sorted([d for d in os.listdir(calib_dir) 
                                   if os.path.isdir(os.path.join(calib_dir, d))])
            if len(calib_folders) > 0:
                task_calib_dir = os.path.join(calib_dir, calib_folders[-1])
                print(f"  使用默认标定文件夹: {task_calib_dir}")
            else:
                print(f"  Warning: 无法找到标定文件夹")
                task_calib_dir = None
        
        # 查找所有cam文件夹
        cam_folders = [d for d in os.listdir(task_path) 
                      if os.path.isdir(os.path.join(task_path, d)) and d.startswith("cam_")]
        
        if len(cam_folders) == 0:
            print(f"  Warning: 在 {task_path} 中没有找到cam文件夹")
            continue
        
        print(f"  找到 {len(cam_folders)} 个相机文件夹")
        
        # 处理每个cam文件夹
        for cam_folder in cam_folders:
            cam_path = os.path.join(task_path, cam_folder)
            cam_serial = cam_folder.replace("cam_", "")
            
            print(f"  处理相机: {cam_serial}")
            
            # 处理该相机文件夹
            coco_data = process_cam_folder(
                cam_path, cam_serial, task_calib_dir, model, processor, device
            )
            
            if coco_data is None:
                continue
            
            # 保存COCO格式的JSON文件
            coco_json_path = os.path.join(cam_path, "annotations.json")
            with open(coco_json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)
            
            print(f"    已保存: {coco_json_path}")
            print(f"    图像数量: {len(coco_data['images'])}")
            print(f"    标注数量: {len(coco_data['annotations'])}")
    
    print("\n处理完成！")

if __name__ == "__main__":
    # 设置路径
    base_dir = "/home/qiansongtang/Documents/program/rgb2voxel/data"
    # 使用相对于当前文件的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Sam3", "model")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 执行处理
    process_multiview_dataset(base_dir, model_path, device)

