#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTX 4090 GPU优化配置
专门针对AutoDL RTX 4090环境的性能优化设置
"""

import torch
import os

def setup_rtx4090_optimization():
    """
    配置RTX 4090的最佳性能设置
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过GPU优化")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    if "4090" not in gpu_name:
        print(f"检测到GPU: {gpu_name}")
        print("此优化专为RTX 4090设计，但仍会应用通用优化")
    else:
        print(f"检测到RTX 4090: {gpu_name}")
    
    # 基本GPU信息
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU内存: {gpu_memory:.1f} GB")
    
    # RTX 4090优化设置
    optimizations = []
    
    # 1. 启用TensorFloat-32 (TF32) - RTX 4090原生支持
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    optimizations.append("TF32加速")
    
    # 2. 启用cuDNN benchmark - 自动选择最优算法
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    optimizations.append("cuDNN自动优化")
    
    # 3. 启用Flash Attention (如果支持)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        optimizations.append("Flash Attention")
    except:
        pass
    
    # 4. 内存管理优化
    if gpu_memory >= 20:  # RTX 4090有24GB
        # 使用90%的GPU内存
        torch.cuda.set_per_process_memory_fraction(0.9)
        optimizations.append("大内存模式(90%)")
    else:
        # 保守的内存使用
        torch.cuda.set_per_process_memory_fraction(0.8)
        optimizations.append("标准内存模式(80%)")
    
    # 5. 清理GPU缓存
    torch.cuda.empty_cache()
    optimizations.append("内存清理")
    
    # 6. 设置环境变量优化
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # 启用cuDNN v8 API
    
    print(f"已启用优化: {', '.join(optimizations)}")
    return True

def get_optimal_batch_size(model_size="medium"):
    """
    根据模型大小推荐最佳批次大小
    """
    if not torch.cuda.is_available():
        return 16
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory >= 20:  # RTX 4090
        batch_sizes = {
            "small": 256,   # 简单模型
            "medium": 128,  # LSTM-TCN
            "large": 64,    # 复杂模型
            "xlarge": 32    # 超大模型
        }
    else:
        batch_sizes = {
            "small": 64,
            "medium": 32,
            "large": 16,
            "xlarge": 8
        }
    
    return batch_sizes.get(model_size, 32)

def get_optimal_learning_rate(batch_size, base_lr=0.001):
    """
    根据批次大小调整学习率
    """
    # 线性缩放规则
    return base_lr * (batch_size / 32)

def setup_mixed_precision():
    """
    设置混合精度训练
    """
    if not torch.cuda.is_available():
        return None, None
    
    from torch.cuda.amp import GradScaler, autocast
    
    scaler = GradScaler()
    print("已启用混合精度训练 (AMP)")
    
    return scaler, autocast

def monitor_gpu_usage():
    """
    监控GPU使用情况
    """
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"GPU内存使用:")
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已保留: {reserved:.2f} GB") 
    print(f"  总内存: {total:.2f} GB")
    print(f"  使用率: {(allocated/total)*100:.1f}%")

def benchmark_performance():
    """
    简单的性能基准测试
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，无法进行GPU基准测试")
        return
    
    print("进行GPU性能基准测试...")
    
    # 创建测试张量
    size = 4096
    device = torch.device('cuda')
    
    # 矩阵乘法测试
    import time
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    for _ in range(10):
        _ = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    
    # 测试
    start_time = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    tflops = (2 * size**3) / (avg_time * 1e12)
    
    print(f"矩阵乘法性能: {avg_time*1000:.2f} ms")
    print(f"计算性能: {tflops:.2f} TFLOPS")
    
    # 清理
    del a, b, c
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=== RTX 4090 优化配置 ===")
    
    # 应用优化
    success = setup_rtx4090_optimization()
    
    if success:
        # 显示推荐配置
        print(f"\n推荐配置:")
        print(f"  批次大小 (中等模型): {get_optimal_batch_size('medium')}")
        print(f"  学习率 (批次128): {get_optimal_learning_rate(128):.6f}")
        
        # 设置混合精度
        scaler, autocast = setup_mixed_precision()
        
        # 监控GPU
        print(f"\nGPU状态:")
        monitor_gpu_usage()
        
        # 性能测试
        print(f"\n性能基准:")
        benchmark_performance()
    
    print("\n优化配置完成！")