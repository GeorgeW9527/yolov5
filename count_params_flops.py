#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计YOLOv5各模型的参数量和FLOPs
"""

import torch
import os
import sys
import yaml
from thop import profile
import io
import contextlib

# 添加YOLOv5根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolo import Model


def count_model_params_flops():
    """统计YOLOv5各模型的参数量和FLOPs"""
    # 模型配置文件路径
    model_configs = [
        'models/yolov5n.yaml',
        'models/yolov5s.yaml',
        'models/yolov5m.yaml',
        'models/yolov5l.yaml',
        'models/yolov5x.yaml'
    ]
    
    # 存储所有模型的结果
    all_results = []
    
    for config_path in model_configs:
        try:
            # 获取模型名称
            model_name = os.path.basename(config_path).replace('.yaml', '')
            
            # 加载模型配置
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 创建模型，并重定向stdout以隐藏模型构建输出
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                model = Model(config)
            
            model.eval()
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 计算FLOPs (640x640输入)
            input_tensor = torch.randn(1, 3, 640, 640)
            
            # 暂时修改model.forward方法以适配thop
            original_forward = model.forward
            
            def modified_forward(x):
                return original_forward(x)[0]  # 只返回预测结果
            
            model.forward = modified_forward
            
            # 使用thop计算FLOPs，并重定向输出
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                flops, params = profile(model, inputs=(input_tensor,), verbose=False)
            
            # 转换为GFLOPs (注意：thop计算的是乘加操作数，YOLOv5通常乘以2来计算FLOPs)
            flops_gflops = flops * 2 / 1e9
            
            # 恢复原始forward方法
            model.forward = original_forward
            
            # 保存结果
            all_results.append({
                'name': model_name,
                'params': total_params,
                'params_grad': total_params_grad,
                'flops': flops_gflops
            })
            
        except Exception as e:
            model_name = os.path.basename(config_path).replace('.yaml', '')
            all_results.append({
                'name': model_name,
                'params': 0,
                'params_grad': 0,
                'flops': 0.0,
                'error': str(e)
            })
    
    # 打印所有结果
    print("\n" + "=" * 60)
    print("{:<10} {:<15} {:<15} {:<15}".format("模型", "参数量(M)", "需要梯度参数(M)", "FLOPs(640x640, GFLOPs)"))
    print("=" * 60)
    
    for result in all_results:
        if 'error' in result:
            print("{:<10} {:<15} {:<15} {:<15}".format(result['name'], "Error", "Error", "Error"))
            print(f"{result['name']} 处理失败: {result['error']}")
        else:
            print("{:<10} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                result['name'],
                result['params'] / 1e6,
                result['params_grad'] / 1e6,
                result['flops']
            ))
    
    print("=" * 60)


if __name__ == '__main__':
    count_model_params_flops()