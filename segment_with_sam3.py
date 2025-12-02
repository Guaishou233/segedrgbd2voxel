#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用SAM3对RGB图像进行语义分割，分割robot arm，其他归类为others
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

def process_dataset(meta_json_path, rgb_dir, model_path, dataset_dir, device="cuda"):
    """
    处理整个数据集，直接保存到原始数据集目录
    
    Args:
        meta_json_path: meta_info.json路径
        rgb_dir: RGB图像目录
        model_path: SAM3模型路径
        dataset_dir: 原始数据集目录
        device: 设备
    """
    print("正在读取meta_info.json...")
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建分割mask输出目录（在原始数据集目录下）
    seg_output_dir = os.path.join(dataset_dir, "SEGMENTATION_MASKS")
    os.makedirs(seg_output_dir, exist_ok=True)
    
    # 加载SAM3模型
    print("正在加载SAM3模型...")
    if USE_TRANSFORMERS:
        model = Sam3Model.from_pretrained(model_path).to(device)
        processor = Sam3Processor.from_pretrained(model_path)
    else:
        model = build_sam3_image_model(model_path)
        processor = Sam3Processor(model)
    
    model.eval()
    
    # 创建categories
    categories = [
        {"id": 1, "name": "robot arm", "supercategory": "object"},
        {"id": 2, "name": "others", "supercategory": "object"}
    ]
    
    # 处理每张图像
    all_annotations = []
    annotation_id = 1
    
    images = coco_data.get("images", [])
    print(f"正在处理 {len(images)} 张图像...")
    
    # 可以限制处理数量用于测试
    # images = images[:10]  # 测试时只处理前10张
    
    for img_info in tqdm(images, desc="分割图像"):
        image_id = img_info["id"]
        rgb_filename = img_info["file_name"]
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        
        if not os.path.exists(rgb_path):
            print(f"Warning: 图像不存在: {rgb_path}")
            continue
        
        try:
            # 使用SAM3进行分割
            masks, boxes, scores = segment_image_with_sam3(
                model, processor, rgb_path, "robot arm", device
            )
            
            # 创建分割mask
            image_size = (img_info["width"], img_info["height"])
            seg_mask = create_segmentation_mask(masks, boxes, scores, image_size)
            
            # 保存分割mask（为了更好的可视化，将值映射：1->255(白色), 2->128(中灰色)）
            seg_filename = rgb_filename.replace(".jpg", ".png").replace(".jpeg", ".png")
            seg_path = os.path.join(seg_output_dir, seg_filename)
            # 将类别值映射到可视化范围：robot arm(1) -> 255(白色), others(2) -> 128(中灰色)
            # 使用更大的颜色区分度，避免图像看起来全黑
            vis_mask = np.zeros_like(seg_mask, dtype=np.uint8)
            vis_mask[seg_mask == 1] = 255  # robot arm显示为白色
            vis_mask[seg_mask == 2] = 128  # others显示为中灰色（增加区分度）
            seg_image = Image.fromarray(vis_mask, mode='L')
            seg_image.save(seg_path)
            
            # 转换为COCO格式的annotations（只保存robot arm的annotations）
            annotations = convert_to_coco_annotations(
                masks, boxes, scores, image_id, image_size, category_id=1, threshold=0.3
            )
            
            # 重新分配annotation ID
            for ann in annotations:
                ann["id"] = annotation_id
                annotation_id += 1
                all_annotations.append(ann)
        
        except Exception as e:
            print(f"Error processing image {rgb_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 更新COCO数据
    coco_data["annotations"] = all_annotations
    coco_data["categories"] = categories
    
    # 直接更新原始数据集的meta_info.json
    print(f"\n正在更新 {meta_json_path}...")
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成！")
    print(f"共处理 {len(images)} 张图像")
    print(f"共生成 {len(all_annotations)} 个annotations")
    print(f"分割mask保存在: {seg_output_dir}")
    print(f"已更新: {meta_json_path}")

if __name__ == "__main__":
    # 设置路径
    base_dir = "/data/tangqiansong/rgb2voxel/data"
    dataset_dir = os.path.join(base_dir, "dataset")
    meta_json_path = os.path.join(dataset_dir, "meta_info.json")
    rgb_dir = os.path.join(dataset_dir, "RGB")
    # 使用相对于当前文件的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Sam3", "model")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 执行处理（直接保存到原始数据集）
    process_dataset(meta_json_path, rgb_dir, model_path, dataset_dir, device)

