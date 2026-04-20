# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model#, vision_transformer
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.models.vision_transformer import _load_weights
import timm

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator
import safetensors

import numpy as np

import models
import models_v2
import models_v3
import models_prepruning
from models_prepruning import ViTTokenDropBeforePatchEmbedWrapper
# import lvvit
import tome

import utils
from torchprofile import profile_macs

# ✨ 프로파일링용 임포트 추가
import torch.profiler as profiler
from collections import defaultdict
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("Warning: pynvml not found. Power measurement will be disabled. (pip install pynvml)")

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("Warning: fvcore not found. Install with 'pip install fvcore' for accurate FLOPs.")




def get_macs(model, x=None, img_size=224):
    model.eval()
    if x is None:
        x = torch.rand(1, 3, img_size, img_size).cuda()
    
    # fvcore 사용 (더 정확함, PaPr 논문과 동일)
    if HAS_FVCORE:
        
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        return fca.total()  # FLOPs 단위로 반환
    
    # fvcore 없으면 torchprofile 사용
    macs = profile_macs(model, x)
    return macs

# ========================================================================================
# def speed_test(model, ntest=100, batchsize=1, img_size=224, x=None, **kwargs):
#     if x is None:
#         x = torch.rand(batchsize, 3, img_size, img_size).cuda()
#     else:
#         batchsize = x.shape[0]
#     model.eval()

#     start = time.time()
#     with torch.no_grad():
#         for i in range(ntest):
#             model(x, **kwargs)
#     end = time.time()

#     elapse = end - start
#     speed = batchsize * ntest / elapse

#     return speed
# ========================================================================================

# ========================================================================================
def speed_test(model, ntest=100, batchsize=1, img_size=224, x=None, warmup=10, **kwargs):
    if x is None:
        x = torch.rand(batchsize, 3, img_size, img_size).cuda()
    else:
        batchsize = x.shape[0]
    model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            model(x, **kwargs)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for i in range(ntest):
            output = model(x, **kwargs)
            torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse
    avg_latency = (elapse / ntest) * 1000  # ms per batch
    per_image_latency = avg_latency / batchsize  # ms per image

    return speed, avg_latency, per_image_latency
# ========================================================================================
def measure_power_memory(model, ntest=100, batchsize=1, img_size=224, x=None, warmup=10, **kwargs):
    """
    속도, 메모리(Peak), 전력(Average)을 동시에 측정하는 함수
    """
    if x is None:
        x = torch.rand(batchsize, 3, img_size, img_size).cuda()
    else:
        batchsize = x.shape[0]
    model.eval()

    # NVML 초기화 (전력 측정용)
    handle = None
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            # 현재 사용 중인 GPU 핸들 가져오기 (기본 0번, CUDA_VISIBLE_DEVICES에 따라 다를 수 있음)
            device_index = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            print(f"NVML Init failed: {e}")
            handle = None

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x, **kwargs)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats() # 메모리 통계 초기화
    
    power_readings = []
    start = time.time()
    
    with torch.no_grad():
        for i in range(ntest):
            output = model(x, **kwargs)
            
            # 전력 측정 (매 반복마다 샘플링)
            if handle:
                try:
                    # milliwatts 단위 반환 -> watts로 변환
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_readings.append(power)
                except:
                    pass
            
            # 너무 잦은 동기화는 전력 측정 루프를 느리게 할 수 있으므로 
            # 정확한 Latency 측정이 주 목적이 아닐 때는 제거하거나 배치 단위로 조절 가능
            # 여기서는 Latency도 같이 재기 위해 유지
            torch.cuda.synchronize() 
            
    end = time.time()

    # 메모리 측정 (Peak)
    max_memory = torch.cuda.max_memory_reserved() / (1024 ** 2) # MB 단위
    
    # 전력 계산
    avg_power = sum(power_readings) / len(power_readings) if power_readings else 0.0
    
    # 속도 계산
    elapse = end - start
    speed = batchsize * ntest / elapse
    avg_latency = (elapse / ntest) * 1000 # ms
    
    if HAS_PYNVML:
        pynvml.nvmlShutdown()

    return speed, avg_latency, max_memory, avg_power

# ...existing code...

# def measure_latency_breakdown(model, ntest=100, batchsize=1, img_size=224, warmup=10):
#     """
#     Latency를 두 부분으로 분리하여 측정:
#     1. Pre-Transformer: Patch Embedding + Token Dropping (if any) + Positional Embedding
#     2. Transformer: Transformer Blocks + Norm + Head
#     """
#     import torch.nn.functional as F
#     import numpy as np
#     model.eval()
#     x = torch.rand(batchsize, 3, img_size, img_size).cuda()
    
#     # 모델 구조 파악
#     is_wrapped = hasattr(model, 'm')  # Prepruning wrapper 여부
#     is_papr = hasattr(model, 'vit')   # PaPr wrapper 여부
    
#     if is_papr:
#         base_model = model.vit
#         has_cnn = True
#     elif is_wrapped:
#         base_model = model.m
#         has_cnn = False
#     else:
#         base_model = model
#         has_cnn = False
    
#     # CUDA Events for precise timing
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
    
#     # Warmup
#     with torch.no_grad():
#         for _ in range(warmup):
#             _ = model(x)
#     torch.cuda.synchronize()
    
#     pre_transformer_times = []
#     transformer_times = []
#     total_times = []
    
#     with torch.no_grad():
#         for _ in range(ntest):
#             # ========== Pre-Transformer 측정 ==========
#             start_event.record()
            
#             if is_papr:
#                 # PaPr: CNN + Patch Embed + Token Selection
#                 cnn_features = model.cnn(x)
#                 proposal_map = cnn_features[-1]
                
#                 x_vit = base_model.patch_embed(x)
#                 cls_token = base_model.cls_token.expand(x_vit.shape[0], -1, -1)
                
#                 if hasattr(base_model, 'dist_token') and base_model.dist_token is not None:
#                     dist_token = base_model.dist_token.expand(x_vit.shape[0], -1, -1)
#                     x_vit = torch.cat((cls_token, dist_token, x_vit), dim=1)
#                 else:
#                     x_vit = torch.cat((cls_token, x_vit), dim=1)
                
#                 x_vit = base_model.pos_drop(x_vit + base_model.pos_embed)
                
#                 # PaPr token selection (apply_papr 로직)
#                 if model.ratio < 1.0:
#                     b, n, c = x_vit.shape
#                     num_patches = n - 1
#                     h1 = w1 = int(np.sqrt(num_patches))
#                     nt = int(num_patches * model.ratio)
                    
#                     Fd = proposal_map.mean(dim=1)
#                     P = F.interpolate(Fd.unsqueeze(1), size=(h1, w1), mode="bicubic", align_corners=False)
#                     P = P.view(b, -1)
#                     patch_indices = P.argsort(dim=1, descending=True)[:, :nt]
#                     patch_indices += 1
#                     class_indices = torch.zeros(b, 1, dtype=torch.long, device=patch_indices.device)
#                     M = torch.cat([class_indices, patch_indices], dim=1)
#                     x_inter = x_vit.gather(dim=1, index=M.unsqueeze(-1).expand(b, -1, c))
#                 else:
#                     x_inter = x_vit
                    
#             elif is_wrapped:
#                 # Prepruning Wrapper: forward_features의 앞부분 재현
#                 # (models_prepruning9_1.py의 forward 로직 참조)
#                 x_patch = base_model.patch_embed(x)
#                 cls_tokens = base_model.cls_token.expand(batchsize, -1, -1)
#                 x_with_cls = torch.cat((cls_tokens, x_patch), dim=1)
#                 x_pos = x_with_cls + base_model.pos_embed
#                 x_drop = base_model.pos_drop(x_pos)
                
#                 if hasattr(model, 'drop_ratio') and model.drop_ratio > 0:
#                     B, N, C = x_drop.shape
                    
#                     # 1. Score 계산 (Importance - alpha * Redundancy)
#                     patch_tokens = x_drop[:, 1:, :] # CLS 제외
                    
#                     # Importance (L2 norm)
#                     importance = patch_tokens.norm(dim=-1)
                    
#                     # Redundancy (Cosine Similarity)
#                     normalized = F.normalize(patch_tokens, dim=-1)
#                     sim_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
#                     redundancy = (sim_matrix.sum(dim=-1) - 1) / (sim_matrix.shape[-1] - 1)
                    
#                     # Final Score
#                     alpha = getattr(model, 'alpha', 0.1) # wrapper의 alpha 값 사용 (기본 0.1)
#                     scores = importance - alpha * redundancy
                    
#                     # 2. Selection (Sort & Gather)
#                     num_keep = int((N - 1) * (1 - model.drop_ratio))
#                     idx = torch.argsort(scores, dim=1, descending=True)[:, :num_keep]
#                     idx = idx + 1 # CLS 인덱스 보정 (+1)
                    
#                     cls_idx = torch.zeros(B, 1, dtype=torch.long, device=x.device)
#                     final_idx = torch.cat([cls_idx, idx], dim=1)
#                     # 3. Pruned Tokens 생성
#                     x_inter = x_drop.gather(dim=1, index=final_idx.unsqueeze(-1).expand(-1, -1, C))
#                 # Token dropping 로직 (wrapper의 forward에서 수행)
#                 # 여기서는 wrapper의 실제 로직을 호출해야 정확함
#                 # 간단히 model의 forward 일부만 측정하기 어려우므로,
#                 # 전체 forward를 두 번 측정하는 방식으로 대체
#                 else:
#                     x_inter = x_drop
                
#             else:
#                 # Vanilla ViT
#                 x_patch = base_model.patch_embed(x)
#                 cls_tokens = base_model.cls_token.expand(batchsize, -1, -1)
#                 x_with_cls = torch.cat((cls_tokens, x_patch), dim=1)
#                 x_pos = x_with_cls + base_model.pos_embed
#                 x_inter = base_model.pos_drop(x_pos)
            
#             end_event.record()
#             torch.cuda.synchronize()
#             pre_transformer_times.append(start_event.elapsed_time(end_event))
            
#             # ========== Transformer Blocks 측정 ==========
#             start_event.record()
            
#             x_blocks = base_model.blocks(x_inter)
#             x_norm = base_model.norm(x_blocks)
            
#             # Head
#             if hasattr(base_model, 'forward_head'):
#                 output = base_model.forward_head(x_norm)
#             elif hasattr(base_model, 'fc_norm'):
#                 if base_model.global_pool:
#                     x_pool = x_norm[:, 1:].mean(dim=1) if base_model.global_pool == 'avg' else x_norm[:, 0]
#                 else:
#                     x_pool = x_norm[:, 0]
#                 x_fc = base_model.fc_norm(x_pool)
#                 output = base_model.head(x_fc)
#             else:
#                 output = base_model.head(x_norm[:, 0])
            
#             end_event.record()
#             torch.cuda.synchronize()
#             transformer_times.append(start_event.elapsed_time(end_event))
            
#             # ========== Total 측정 (검증용) ==========
#             start_event.record()
#             _ = model(x)
#             end_event.record()
#             torch.cuda.synchronize()
#             total_times.append(start_event.elapsed_time(end_event))
    
#     # 통계 계산
#     results = {
#         'pre_transformer': {
#             'mean_ms': np.mean(pre_transformer_times),
#             'std_ms': np.std(pre_transformer_times),
#             'per_image_ms': np.mean(pre_transformer_times) / batchsize
#         },
#         'transformer': {
#             'mean_ms': np.mean(transformer_times),
#             'std_ms': np.std(transformer_times),
#             'per_image_ms': np.mean(transformer_times) / batchsize
#         },
#         'total': {
#             'mean_ms': np.mean(total_times),
#             'std_ms': np.std(total_times),
#             'per_image_ms': np.mean(total_times) / batchsize
#         },
#         'pre_transformer_ratio': np.mean(pre_transformer_times) / np.mean(total_times) * 100,
#         'transformer_ratio': np.mean(transformer_times) / np.mean(total_times) * 100
#     }
    
#     return results

# ...existing code...

# ✨ 새로운 프로파일링 함수
def profile_model(model, img_size=224, batch_size=1, trace_file="trace.json"):
    """
    모델의 상세한 프로파일링 수행
    - GPU 시간 분석
    - 메모리 사용량
    - 각 레이어별 시간
    """
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size).cuda()
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            model(x)
        
        torch.cuda.synchronize()
        
        # Profile
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        ) as prof:
            for _ in range(100):
                model(x)
                torch.cuda.synchronize()
        
        # Export Chrome trace
        if trace_file:
            prof.export_chrome_trace(trace_file)
            print(f"\n✅ Chrome trace saved to {trace_file}")
            print(f"   Open chrome://tracing and load this file for visualization\n")
        
        # Print summary
        print("="*100)
        print("TOP 20 OPERATIONS BY GPU TIME")
        print("="*100)
        print(prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=20))
        
        # Analyze by operation type
        print("\n" + "="*100)
        print("GPU TIME BY OPERATION TYPE")
        print("="*100)
        events = prof.key_averages()
        
        # ✨ 버전 호환성: device_time_total 또는 cuda_time_total 사용
        # PyTorch 2.0+ 에서는 device_time_total 사용
        time_attr = 'device_time_total' if hasattr(events[0], 'device_time_total') else 'cuda_time_total'
        
        total_time = sum([getattr(e, time_attr, 0) for e in events])
        
        op_times = defaultdict(float)
        op_counts = defaultdict(int)
        
        for e in events:
            # 연산 종류별로 그룹화
            op_name = e.key
            if '::' in op_name:
                op_type = op_name.split('::')[-1].split('<')[0]
            else:
                op_type = op_name.split('<')[0]
            
            op_times[op_type] += getattr(e, time_attr, 0)
            op_counts[op_type] += 1
        
        sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)[:15]
        
        print(f"{'Operation':<40} {'Count':<8} {'Total Time (ms)':<18} {'Percentage':<12}")
        print("-"*100)
        for op, time_us in sorted_ops:
            time_ms = time_us / 1000
            percentage = (time_us / total_time * 100) if total_time > 0 else 0
            count = op_counts[op]
            print(f"{op:<40} {count:<8} {time_ms:>15.2f} ms   {percentage:>8.1f}%")
        
        # Memory statistics
        print("\n" + "="*100)
        print("MEMORY STATISTICS")
        print("="*100)
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=10))
        
        return prof

# ✨ 레이어별 상세 프로파일링
def profile_layers(model, img_size=224, batch_size=1):
    """
    각 Transformer 블록별 시간 측정
    """
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size).cuda()
    
    # 모델 구조 파악
    is_wrapped = hasattr(model, 'm')
    base_model = model.m if is_wrapped else model
    
    if not hasattr(base_model, 'blocks'):
        print("⚠️  Model doesn't have 'blocks' attribute. Skipping layer profiling.")
        return
    
    print("\n" + "="*100)
    print("LAYER-BY-LAYER PROFILING")
    print("="*100)
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            model(x)
        
        torch.cuda.synchronize()
        
        # Patch embedding 시간
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        if is_wrapped:
            # Prepruning wrapper인 경우
            print("\n[Pre-processing (Patch Embed + Token Drop)]")
            start.record()
            for _ in range(100):
                x_temp = base_model.patch_embed(x)
                # Token dropping logic would go here
            end.record()
            torch.cuda.synchronize()
            print(f"  Average time: {start.elapsed_time(end) / 100:.3f} ms")
        else:
            print("\n[Patch Embedding]")
            start.record()
            for _ in range(100):
                base_model.patch_embed(x)
            end.record()
            torch.cuda.synchronize()
            print(f"  Average time: {start.elapsed_time(end) / 100:.3f} ms")
        
        # 전체 forward로 중간 출력 얻기
        if is_wrapped:
            x_patch = base_model.patch_embed(x)
            # Wrapper의 forward 로직 일부 재현
            # (실제로는 wrapper 코드에 맞게 수정 필요)
            x_inter = x_patch  # Simplified
        else:
            x_inter = base_model.patch_embed(x)
            if hasattr(base_model, 'cls_token'):
                cls_tokens = base_model.cls_token.expand(batch_size, -1, -1)
                x_inter = torch.cat([cls_tokens, x_inter], dim=1)
            if hasattr(base_model, 'pos_embed'):
                x_inter = x_inter + base_model.pos_embed
            if hasattr(base_model, 'pos_drop'):
                x_inter = base_model.pos_drop(x_inter)
        
        # 각 Transformer 블록별 측정
        print(f"\n{'Block':<10} {'Tokens':<10} {'Time (ms)':<12} {'Cumulative (ms)':<18}")
        print("-"*100)
        
        cumulative_time = 0
        for i, block in enumerate(base_model.blocks):
            num_tokens = x_inter.shape[1]
            
            start.record()
            for _ in range(100):
                x_inter = block(x_inter)
            end.record()
            torch.cuda.synchronize()
            
            block_time = start.elapsed_time(end) / 100
            cumulative_time += block_time
            
            print(f"Block {i:<4} {num_tokens:<10} {block_time:>9.3f} ms   {cumulative_time:>12.3f} ms")
        
        # Norm + Head
        print(f"\n[Norm + Head]")
        start.record()
        for _ in range(100):
            x_norm = base_model.norm(x_inter)
            if hasattr(base_model, 'forward_head'):
                base_model.forward_head(x_norm)
            elif hasattr(base_model, 'head'):
                base_model.head(x_norm[:, 0])
        end.record()
        torch.cuda.synchronize()
        print(f"  Average time: {start.elapsed_time(end) / 100:.3f} ms")
        print(f"\nTotal cumulative block time: {cumulative_time:.3f} ms")

# ✨ 종합 벤치마크 함수
def comprehensive_benchmark(model, model_name, img_size=224, batch_sizes=[1, 32, 64, 128], 
                           output_dir="benchmark_results"):
    """
    다양한 batch size에 대한 종합 벤치마크
    """
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*100)
    print(f"COMPREHENSIVE BENCHMARK: {model_name}")
    print("="*100)
    
    results = []
    
    for bs in batch_sizes:
        print(f"\n📊 Batch Size: {bs}")
        print("-"*50)
        
        # Speed test
        speeds = []
        latencies = []
        memories = []
        
        for run in range(5):
            torch.cuda.empty_cache()
            speed, batch_lat, img_lat, peak_mem = speed_test(
                model, ntest=100, batchsize=bs, 
                img_size=img_size, warmup=10
            )
            speeds.append(speed)
            latencies.append(img_lat)
            memories.append(peak_mem)
        
        result = {
            'model': model_name,
            'batch_size': bs,
            'img_size': img_size,
            'throughput_mean': np.mean(speeds),
            'throughput_std': np.std(speeds),
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'memory_mean': np.mean(memories),
            'memory_std': np.std(memories),
        }
        
        results.append(result)
        
        print(f"  Throughput: {result['throughput_mean']:.2f} ± {result['throughput_std']:.2f} img/s")
        print(f"  Latency:    {result['latency_mean']:.2f} ± {result['latency_std']:.2f} ms/img")
        print(f"  Memory:     {result['memory_mean']:.3f} ± {result['memory_std']:.3f} GB")
    
    # GMACs 측정
    gmacs = get_macs(model, None, img_size) * 1e-9
    print(f"\n📈 GMACs: {gmacs:.3f}")
    
    # 결과 저장
    df = pd.DataFrame(results)
    df['gmacs'] = gmacs
    
    csv_path = os.path.join(output_dir, f"{model_name}_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
    
    # 요약 출력
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(df.to_string(index=False))
    
    return df

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.9, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Methods of selecting propagated tokens
    parser.add_argument('--selection', default='None', 
                        choices=['CLSAttnMean', 'CLSAttnMax', 'IMGAttnMean', 'IMGAttnMax', 
                                 'DiagAttnMean', 'DiagAttnMax', 'MixedAttnMean', 'MixedAttnMax',
                                 'CosSimMean', 'CosSimMax', 'SumAttnMax','Random', 'None'],
                        type=str)
    # Methods of propagating tokens
    parser.add_argument('--propagation', default='None',
                        choices=['None', 'Mean', 'GraphProp'],
                        type=str)
    # Types of graph
    parser.add_argument('--graph_type', default='None', choices=['None', 'Spatial', 'Semantic', 'Mixed'], type=str)
    parser.add_argument('--num_prop', type=int, default=0)
    parser.add_argument('--num_neighbours', type=int, default=8)
    parser.add_argument('--sparsity', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--token_scale', action='store_true', default=False)
    
    # speed test
    parser.add_argument('--test_speed', action='store_true')
    parser.add_argument('--only_test_speed', action='store_true')     
    
    # my idea
    parser.add_argument('--prepatch_token_drop', action='store_true',
                       help='Enable token pruning before patch embedding (wrapper).')
    parser.add_argument('--prepatch_drop_ratio', type=float, default=0.0,
                     help='Drop ratio (0~1). e.g., 0.3 drops 30% of tokens before patch embedding.')
    parser.add_argument('--prepatch_backbone', type=str, default=None,
                      help='timm backbone name for prepatch wrapper. If None, use args.model when possible.')
    parser.add_argument('--tome_r', type=int, default=0, help='ToMe reduction (tokens merged per layer; 0 to disable). Suggested: 8-16')

    
    parser.add_argument('--profile', action='store_true',
                       help='Enable detailed profiling (generates Chrome trace)')
    parser.add_argument('--profile_layers', action='store_true',
                       help='Enable layer-by-layer profiling')
    parser.add_argument('--comprehensive_benchmark', action='store_true',
                       help='Run comprehensive benchmark across multiple batch sizes')
    parser.add_argument('--trace_file', type=str, default='trace.json',
                       help='Path to save Chrome trace file')
    parser.add_argument('--benchmark_batch_sizes', type=int, nargs='+', 
                       default=[1, 32, 64, 128],
                       help='Batch sizes for comprehensive benchmark')
    
    
    parser.add_argument('--papr', action='store_true', help='Enable PaPr (Patch Pruning)')
    parser.add_argument('--papr_ratio', type=float, default=1.0, help='Keeping ratio for PaPr (e.g., 0.5 keeps 50% tokens)')
    parser.add_argument('--papr_cnn', type=str, default='mobileone_s0', help='CNN backbone for PaPr proposal')
                        
                        
    return parser



def main(args):
    utils.init_distributed_mode(args)
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    # GTP-ViT options   
    # model = create_model(
    #     args.model,
    #     pretrained=True,
    #     num_classes=args.nb_classes,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    #     img_size=args.input_size,
    #     selection=args.selection,
    #     propagation=args.propagation,
    #     num_prop=args.num_prop,
    #     num_neighbours=args.num_neighbours,
    #     sparsity=args.sparsity,
    #     alpha=args.alpha,
    #     token_scale=args.token_scale,
    #     graph_type=args.graph_type
    # )

    # my idea
    # === (NEW) Pre-patch token drop wrapper ===
    if args.prepatch_token_drop:
        from models_prepruning import build_token_drop_prepatch
        from models_prepruning1_1 import build_token_drop_after_patch
        from models_prepruning1_2 import build_token_drop_after_posembed
        from models_prepruning2_1 import build_token_drop_after_patch2_1
        from models_prepruning3_1 import build_token_drop_after_patch_saliency
        from models_prepruning4_1 import build_token_drop_after_patch4_1
        from models_prepruning4 import build_token_drop_before_patch4
        from models_prepruning5 import build_token_drop_saliency
        from models_prepruning6 import build_token_drop_by_pos_change
        from models_prepruning6_1 import build_token_drop_raw_sim
        from models_prepruning6_2 import build_token_drop_by_final_embed_norm
        from models_prepruning6_3 import build_token_drop_by_raw_norm
        from models_prepruning7 import build_token_drop_by_pos_change_only
        from models_prepruning9 import build_token_drop_by_pos_change_9 
        from models_prepruning9_1 import build_token_drop_by_pos_change_9_1
        from models_prepruning10 import build_token_drop_by_pos_change_10
        from models_prepruning11 import build_token_drop_by_pos_change_11
        from models_prepruning12 import build_token_drop_by_pos_change_12
        from models_prepruning_random import build_token_drop_random
        from models_prepruning13_1 import build_token_drop_by_pos_change_13_1
        # 백본 이름 결정: 명시되면 그걸, 아니면 현재 args.model이 timm 백본 이름일 때 사용
        
        backbone_name = args.prepatch_backbone or 'vit_base_patch16_224'
        
        print(f"[TokenDropPrePatch] wrapping timm backbone={backbone_name} with drop_ratio={args.prepatch_drop_ratio}")
        # timm 백본을 새로 만들고 래핑 → 기존 head/num_classes는 wrapper 내부에서 그대로 사용
        wrapped = build_token_drop_by_pos_change_9_1(
            backbone_name,
            drop_ratio=args.prepatch_drop_ratio,
            pretrained=True,
            num_classes=args.nb_classes,
            img_size=args.input_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None
        )
        # 분류 헤드 차원이 다르면 timm가 맞춰서 만들어주므로 그대로 사용
        # student 파이프라인과 호환 위해 model 교체
        model = wrapped
# ========================================
    else:
        model = create_model(
            args.model,
            pretrained=True,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size,
            #============ GTP-ViT options ============#
            # python main.py --data-path /home/esoc/datasets/ILSVRC2012  --model graph_propagation_vit_base_patch16_224_augreg --eval --resume /home/esoc/junsung/GTP-ViT_JS/vit_base_augreg_in21k_ft_in1k.pth --sparsity 1.0 --alpha 0.1 --num_prop 4 --selection MixedAttnMax --propagation GraphProp --batch-size 64
            # python main.py --data-path /home/esoc/datasets/ILSVRC2012  --model graph_propagation_vit_small_patch16_224_augreg --eval --resume /home/esoc/junsung/GTP-ViT_JS/vit_small_augreg_in21k_ft_in1k.pth --sparsity 1.0 --alpha 0.1 --num_prop 4 --selection MixedAttnMax --propagation GraphProp --batch-size 64
            # selection=args.selection,
            # propagation=args.propagation,
            # num_prop=args.num_prop,
            # num_neighbours=args.num_neighbours,
            # sparsity=args.sparsity,
            # alpha=args.alpha,
            # token_scale=args.token_scale,
            # graph_type=args.graph_type
        )
        # Apply ToMe patching if enabled
    if args.papr:
        from models_papr import build_papr_model
        print(f"[PaPr] Building PaPr Model with ratio={args.papr_ratio}, CNN={args.papr_cnn}")
        model = build_papr_model(
            vit_name=args.model,
            ratio=args.papr_ratio,
            cnn_model_name=args.papr_cnn,
            pretrained=True,
            num_classes=args.nb_classes,
            img_size=args.input_size
        )
    
    if args.tome_r:
        # Determine if prop_attn is needed propotional attention
        if isinstance(args.tome_r, int):
            args.tome_r = [args.tome_r] * 12  # depth = 블록 개수 (예: 12)
        is_vit = 'vit' in (args.prepatch_backbone or args.model).lower()
        base_model = model.m if args.prepatch_token_drop else model
        tome.patch.timm(base_model, prop_attn=True)  # prop_attn=True 하면 정확도 올라감 Deit, ViT 둘다 가능
        base_model._tome_info["r"] = args.tome_r.copy()
        base_model.r = args.tome_r
        print(f"[ToMe] Patched model with r={args.tome_r}, prop_attn={is_vit}")
    # ========================================
    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='token',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.endswith(".npz"):
            _load_weights(model_without_ddp, args.resume)
        else:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            elif args.resume.endswith(".safetensors"):
                checkpoint = safetensors.torch.load_file(args.resume, device="cpu")
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            
            if 'student' in checkpoint:
                new_checkpoint = dict()
                for key, value in checkpoint["student"].items():
                    key = key.replace("module.", "")
                    if key == "head.last_layer.weight":
                        new_checkpoint[key+"_v"] = value
                    else:
                        new_checkpoint[key] = value
                checkpoint = new_checkpoint
            
            try:
                model_without_ddp.load_state_dict(checkpoint['model'])
            except:
                model_without_ddp.load_state_dict(checkpoint)
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if args.model_ema:
                    utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
            lr_scheduler.step(args.start_epoch)
    
    if args.test_speed:
        print('='*70)
        print('COMPUTATIONAL COST ANALYSIS')
        print('='*70)
        #         # ✨ [추가] PaPr 상세 분석 로직
        if args.papr:
            print(f"[PaPr Analysis] Ratio: {args.papr_ratio}, CNN: {args.papr_cnn}")
            
            # 1. Baseline ViT (Pruning 없음)
            baseline_vit = timm.create_model(
                args.model, pretrained=False, num_classes=args.nb_classes, img_size=args.input_size
            ).cuda().eval()
            baseline_macs = get_macs(baseline_vit, None, args.input_size)
            
            # 2. Proposal CNN (Reparameterized)
            cnn = timm.create_model(args.papr_cnn, features_only=True, pretrained=False).cuda().eval()
            # MobileOne 등 Reparameterization 필요 시 수행
            if hasattr(cnn, 'reparameterize'): cnn.reparameterize()
            for m in cnn.modules():
                if hasattr(m, 'reparameterize'): m.reparameterize()
            
            cnn_macs = get_macs(cnn, None, args.input_size)
            
            # 3. 실제 PaPr 모델 측정
            real_macs = get_macs(model_without_ddp, None, args.input_size)
            
            # 4. 이론적 근사치 (CNN + ViT * Ratio)
            # ViT의 PatchEmbed/CLS 등은 Ratio 영향 안 받지만, 대략적인 계산
            theoretical_macs = cnn_macs + (baseline_macs * args.papr_ratio)
            
            print(f"\n[Detailed Breakdown]")
            print(f"  1. Baseline ViT:    {baseline_macs * 1e-9:.3f} G")
            print(f"  2. Proposal CNN:    {cnn_macs * 1e-9:.3f} G")
            print(f"  3. Theoretical Est: {theoretical_macs * 1e-9:.3f} G (CNN + ViT × Ratio)")
            print(f"  4. Measured (PaPr): {real_macs * 1e-9:.3f} G")
            
            diff = real_macs - theoretical_macs
            print(f"\n  Difference (Measured - Theoretical): {diff * 1e-9:.3f} G")
            print(f"  (Positive diff means overheads like Interpolation/Gather are included)")
            
            MACs = real_macs
            del baseline_vit, cnn
            
        if args.prepatch_token_drop:
            # ✨ Baseline (wrapper 없이)
            baseline = timm.create_model(
                args.prepatch_backbone or 'vit_base_patch16_224',
                pretrained=False,
                num_classes=args.nb_classes,
                img_size=args.input_size
            ).cuda()
            baseline.eval()
            
            baseline_macs = get_macs(baseline, None, args.input_size)
            wrapped_macs = get_macs(model_without_ddp, None, args.input_size)
            
            print(f'\n[Baseline Model]')
            print(f'  GMACs: {baseline_macs * 1e-9:.3f}')
            
            print(f'\n[With Wrapper]')
            print(f'  GMACs: {wrapped_macs * 1e-9:.3f}')
            print(f'  Overhead: +{(wrapped_macs - baseline_macs) * 1e-9:.3f} GMACs')
            print(f'  Overhead %: +{(wrapped_macs / baseline_macs - 1) * 100:.1f}%')
            
            MACs = wrapped_macs
        else:
            MACs = get_macs(model_without_ddp, None, args.input_size)
        
        print(f'\nFinal GMACs: {MACs * 1e-9:.3f}')
        print('='*70 + '\n')



        print('='*70)
        print('Start inference speed testing...')
        print(f'Batch size: {args.batch_size}, Image size: {args.input_size}')
        print('='*70)
        
        # Warmup
        warmup_speed, _, _ = speed_test(model, ntest=20, batchsize=args.batch_size, 
                                        img_size=args.input_size, warmup=10)
        print(f'Warmup complete: {warmup_speed:.2f} images/s\n')
        
        # Multiple runs for statistics
        speeds = []
        batch_latencies = []
        image_latencies = []
        num_runs = 5
        
        for i in range(num_runs):
            torch.cuda.empty_cache()
            speed, batch_lat, img_lat = speed_test(model, ntest=100, 
                                                    batchsize=args.batch_size, 
                                                    img_size=args.input_size,
                                                    warmup=5)
            speeds.append(speed)
            batch_latencies.append(batch_lat)
            image_latencies.append(img_lat)
            print(f'Run {i+1}: Throughput={speed:.2f} img/s, '
                f'Batch Latency={batch_lat:.2f}ms, '
                f'Per-Image Latency={img_lat:.2f}ms')
        
        # Calculate statistics
        print('\n' + '='*70)
        print('THROUGHPUT STATISTICS:')
        print(f'  Average: {np.mean(speeds):.2f} ± {np.std(speeds):.2f} images/s')
        print(f'  Min: {np.min(speeds):.2f}, Max: {np.max(speeds):.2f} images/s')
        
        print('\nLATENCY STATISTICS (Batch):')
        print(f'  Average: {np.mean(batch_latencies):.2f} ± {np.std(batch_latencies):.2f} ms')
        print(f'  Min: {np.min(batch_latencies):.2f}, Max: {np.max(batch_latencies):.2f} ms')
        
        print('\nLATENCY STATISTICS (Per-Image):')
        print(f'  Average: {np.mean(image_latencies):.2f} ± {np.std(image_latencies):.2f} ms')
        print(f'  Min: {np.min(image_latencies):.2f}, Max: {np.max(image_latencies):.2f} ms')
        print('='*70 + '\n')
        
        MACs = get_macs(model_without_ddp, None, args.input_size)
        print(f'GMACs: {MACs * 1e-9:.3f}\n')
        
        
        
        print('='*70)
        print('COMPUTATIONAL COST ANALYSIS')
# ================================= 전력 및 메모리 측정 ===========================================
        print(f'GMACs: {MACs * 1e-9:.3f}\n')
        
        # ✨ 전력 및 메모리 측정 추가
        print('='*70)
        print('POWER & MEMORY ANALYSIS')
        print('='*70)
        
        # 정확한 측정을 위해 반복 횟수를 늘림
        n_measure = 200 
        throughput, latency, peak_mem, avg_power = measure_power_memory(
            model, ntest=n_measure, batchsize=args.batch_size, 
            img_size=args.input_size, warmup=20
        )
        
        print(f"Batch Size: {args.batch_size}")
        print(f"Peak Memory Usage: {peak_mem:.2f} MB")
        if HAS_PYNVML and avg_power > 0:
            print(f"Average Power Consumption: {avg_power:.2f} W")
            # 에너지 효율성 (Images per Joule) = Throughput / Power
            if avg_power > 0:
                print(f"Energy Efficiency: {throughput / avg_power:.2f} img/J")
        else:
            print("Power measurement unavailable (requires pynvml and supported GPU)")
        print('='*70 + '\n')
        
        # =============================================================================================
        
        # ✨ Latency Breakdown 측정 추가
        # print('='*70)
        # print('LATENCY BREAKDOWN ANALYSIS')
        # print('='*70)
        
        # breakdown = measure_latency_breakdown(
        #     model, ntest=100, batchsize=args.batch_size, 
        #     img_size=args.input_size, warmup=20
        # )
        
        # print(f"\n[Pre-Transformer (Patch Embed + Token Drop + Pos Embed)]")
        # print(f"  Batch Latency:     {breakdown['pre_transformer']['mean_ms']:.3f} ± {breakdown['pre_transformer']['std_ms']:.3f} ms")
        # print(f"  Per-Image Latency: {breakdown['pre_transformer']['per_image_ms']:.3f} ms")
        # print(f"  Ratio of Total:    {breakdown['pre_transformer_ratio']:.1f}%")
        
        # print(f"\n[Transformer (Blocks + Norm + Head)]")
        # print(f"  Batch Latency:     {breakdown['transformer']['mean_ms']:.3f} ± {breakdown['transformer']['std_ms']:.3f} ms")
        # print(f"  Per-Image Latency: {breakdown['transformer']['per_image_ms']:.3f} ms")
        # print(f"  Ratio of Total:    {breakdown['transformer_ratio']:.1f}%")
        
        # print(f"\n[Total (End-to-End)]")
        # print(f"  Batch Latency:     {breakdown['total']['mean_ms']:.3f} ± {breakdown['total']['std_ms']:.3f} ms")
        # print(f"  Per-Image Latency: {breakdown['total']['per_image_ms']:.3f} ms")
        
        # # 합산 검증
        # sum_parts = breakdown['pre_transformer']['mean_ms'] + breakdown['transformer']['mean_ms']
        # print(f"\n[Validation]")
        # print(f"  Sum of parts: {sum_parts:.3f} ms")
        # print(f"  Measured total: {breakdown['total']['mean_ms']:.3f} ms")
        # print(f"  Difference: {abs(sum_parts - breakdown['total']['mean_ms']):.3f} ms (overhead/measurement noise)")
        
        # print('='*70 + '\n')
    # =============================================================================================
        
    # if args.test_speed:
    #     # test model throughput for three times to ensure accuracy
    #     print('Start inference speed testing...')
    #     inference_speed = speed_test(model, img_size=args.input_size)
    #     print('inference_speed (inaccurate):', inference_speed, 'images/s')
    #     total = 0
    #     inference_speed = speed_test(model, img_size=args.input_size)
    #     print('inference_speed:', inference_speed, 'images/s')
    #     total = total + inference_speed
    #     inference_speed = speed_test(model, img_size=args.input_size)
    #     print('inference_speed:', inference_speed, 'images/s')
    #     total = total + inference_speed
    #     #inference_speed = speed_test(model)
    #     #print('inference_speed:', inference_speed, 'images/s')
    #     #total = total + inference_speed
    #     print('Average throughput:', round(total/2, 2), 'images/s')
    #     MACs = get_macs(model_without_ddp,  None, args.input_size)
    #     print('GMACs:', MACs * 1e-9)
    if args.only_test_speed:
        return

    # ✨ 프로파일링 로직 추가 (test_speed 이후)
    if args.profile:
        print("\n" + "="*100)
        print("STARTING DETAILED PROFILING")
        print("="*100)
        
        trace_path = Path(args.output_dir) / args.trace_file if args.output_dir else args.trace_file
        profile_model(model, img_size=args.input_size, 
                     batch_size=args.batch_size, trace_file=str(trace_path))
    
    if args.profile_layers:
        profile_layers(model, img_size=args.input_size, batch_size=args.batch_size)
    
    if args.comprehensive_benchmark:
        model_name = f"{args.model}"
        if args.prepatch_token_drop:
            model_name += f"_prepruning{int(args.prepatch_drop_ratio*100)}"
        if args.tome_r > 0:
            model_name += f"_tome{args.tome_r}"
        
        output_dir = Path(args.output_dir) / "benchmarks" if args.output_dir else "benchmark_results"
        comprehensive_benchmark(
            model, model_name, 
            img_size=args.input_size,
            batch_sizes=args.benchmark_batch_sizes,
            output_dir=str(output_dir)
        )
    
    if args.only_test_speed or args.profile or args.profile_layers or args.comprehensive_benchmark:
        return
            
    if args.eval:
        MACs = get_macs(model_without_ddp, None, args.input_size)
        print('GMACs:', MACs * 1e-9)
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        
        ####### delete
        with open(args.model+"_results", "a") as fp:
            fp.write("|  Num Prop: %2d  " % args.num_prop)
            fp.write("|  GMACs: %1.3f  " % (MACs*1e-9))
            fp.write("|  Top-1 Acc: %2.2f  " % test_stats['acc1'])
            fp.write("|  Selection: %13s  " % args.selection)
            fp.write("|  Propagation: %10s  " % args.propagation)
            fp.write("|  Sparsity: %1.2f  " % args.sparsity)
            fp.write("|  Alpha: %1.2f  " % args.alpha)
            fp.write("|  Scale: %5s  " % str(args.token_scale))
            fp.write("|  Graph: %8s  |\n\n" % str(args.graph_type))
        ####### delete
        
        
        return
    
    MACs = get_macs(model)
    print('Model GMACs:', MACs * 1e-9)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        
        
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
