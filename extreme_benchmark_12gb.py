# Copyright 2025-2026 SKDOSS Pvt Ltd. Author: Sai Kamal Doss Ambalapattil (SKDOSS).
# Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
# Patent pending: U.S. Provisional Patent Application No. 63/950,423.
# SPDX-License-Identifier: Apache-2.0

"""
COMPLETE COMPREHENSIVE BENCHMARK - ALL MODELS, ALL TESTS
=========================================================

Tests EVERYTHING:
1. Images: All models at all resolutions (224-4096)
2. Videos: All models at ALL resolutions (360p, 480p, 720p, 1080p, 4K)
           Progressive durations (1s, 3s, 5s, 10s, 20s, 30s, 60s, 120s)
3. Batch: All models at various batch sizes
4. Shows exact OOM point for each model/config
5. Comprehensive rating tables

Author: Sai Kamal Doss Ambalapattil (SKDOSS)
Purpose: COMPLETE performance analysis for patent
"""

import torch
import torch.nn as nn
import time
import gc
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import models
from Vision_AHCN import VisionAHCN, HighResVisionAHCN, VideoAHCN


# =============================================================================
# BASELINE MODELS
# =============================================================================

class StandardCNN(nn.Module):
    """ResNet-18 baseline."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class VisionTransformer(nn.Module):
    """ViT baseline."""
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 embed_dim=512, depth=6, num_heads=8):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) + self.pos_embed[:, :x.size(1)+1, :]
        return self.head(self.norm(self.transformer(x)[:, 0]))


class Video3DCNN(nn.Module):
    """I3D-style 3D CNN."""
    def __init__(self, num_classes=400):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, (3,7,7), (1,2,2), (1,3,3))
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d((1,3,3), (1,2,2), (0,1,1))
        
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_c, out_c, blocks):
        layers = []
        for i in range(blocks):
            layers.extend([
                nn.Conv3d(in_c if i==0 else out_c, out_c, 3, (1,2,2) if i==0 else 1, 1),
                nn.BatchNorm3d(out_c), nn.ReLU()
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W) -> (B,C,T,H,W)
        x = self.maxpool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# =============================================================================
# UTILITIES
# =============================================================================

def clear_gpu():
    """Aggressive GPU cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_memory_mb():
    """Get GPU VRAM usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        device = torch.cuda.current_device()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        return min(peak_mb, total_mb)
    return 0


def test_single_config(model, input_shape, device='cuda', max_vram_mb=11000, iters=10):
    """Test single configuration."""
    clear_gpu()
    
    try:
        model = model.to(device).eval()
        x = torch.randn(*input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        clear_gpu()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.time()
                output = model(x)
                torch.cuda.synchronize()
                times.append((time.time() - t0) * 1000)
        
        mem_mb = get_gpu_memory_mb()
        
        # Check if approaching limit
        if mem_mb > max_vram_mb:
            del model, x, output
            clear_gpu()
            return {'success': False, 'error': 'Approaching VRAM limit', 'memory_mb': mem_mb}
        
        result = {
            'success': True,
            'memory_mb': mem_mb,
            'latency_ms': float(np.mean(times)),
            'throughput': 1000.0 / np.mean(times)
        }
        
        del model, x, output
        clear_gpu()
        return result
        
    except RuntimeError as e:
        clear_gpu()
        if "out of memory" in str(e).lower():
            return {'success': False, 'error': 'OOM', 'memory_mb': 0}
        return {'success': False, 'error': str(e)[:50], 'memory_mb': 0}
    except Exception as e:
        clear_gpu()
        return {'success': False, 'error': str(e)[:50], 'memory_mb': 0}


def format_duration(seconds):
    """Format duration as human readable."""
    if seconds >= 60:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m{secs}s" if secs > 0 else f"{mins}m"
    return f"{seconds}s"


# =============================================================================
# COMPREHENSIVE TESTS
# =============================================================================

def test_image_all_models(max_vram_mb=11000):
    """Test ALL models on image classification."""
    print("\n" + "="*100)
    print("TEST 1: IMAGE CLASSIFICATION - ALL MODELS")
    print("="*100)
    
    resolutions = [224, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096]
    
    models_to_test = {
        'CNN': lambda res: StandardCNN(num_classes=1000),
        'ViT': lambda res: VisionTransformer(img_size=res, num_classes=1000, depth=6),
        'AHCN-BlockSparse': lambda res: VisionAHCN(img_size=res, num_classes=1000, attention_type='block_sparse'),
        'AHCN-Hierarchical': lambda res: HighResVisionAHCN(img_size=res, num_classes=1000)
    }
    
    results = {}
    
    for model_name, model_fn in models_to_test.items():
        print(f"\n[{model_name}]")
        print("-" * 100)
        
        model_results = []
        max_res = None
        
        for res in resolutions:
            print(f"  {res}×{res}...", end=" ", flush=True)
            
            model = model_fn(res)
            result = test_single_config(model, (1, 3, res, res), max_vram_mb=max_vram_mb)
            
            if result['success']:
                print(f"✅ {result['memory_mb']:.0f}MB, {result['latency_ms']:.1f}ms")
                max_res = res
                model_results.append({'resolution': res, **result})
            else:
                print(f"❌ {result['error']}")
                model_results.append({'resolution': res, **result})
                break
        
        results[model_name] = {
            'max_resolution': max_res,
            'results': model_results
        }
    
    return results


def test_video_all_models_all_resolutions(max_vram_mb=11000):
    """Test ALL models on ALL video resolutions with progressive durations."""
    print("\n" + "="*100)
    print("TEST 2: VIDEO PROCESSING - ALL MODELS, ALL RESOLUTIONS")
    print("="*100)
    
    # Video scenarios: (name, width, height, fps, durations_to_test)
    video_configs = [
        ('360p', 640, 360, 30, [1, 3, 5, 10, 20, 30, 60, 120]),
        ('480p', 854, 480, 30, [1, 3, 5, 10, 20, 30, 60]),
        ('720p', 1280, 720, 30, [1, 3, 5, 10, 20, 30]),
        ('1080p', 1920, 1080, 30, [1, 3, 5, 10, 20]),
        ('4K', 3840, 2160, 30, [1, 3, 5, 10])
    ]
    
    models_to_test = {
        '3D-CNN': lambda size, frames: Video3DCNN(num_classes=400),
        'ViT-Video': lambda size, frames: VisionTransformer(img_size=size, num_classes=400, depth=6) if size <= 512 else None,
        'AHCN-BlockSparse': lambda size, frames: VideoAHCN(img_size=size, num_frames=frames, num_classes=400, attention_type='block_sparse'),
        'AHCN-Hierarchical': lambda size, frames: VideoAHCN(img_size=size, num_frames=frames, num_classes=400, attention_type='hierarchical')
    }
    
    results = {}
    
    for model_name in models_to_test.keys():
        print(f"\n{'='*100}")
        print(f"[{model_name}]")
        print('='*100)
        
        model_results = {}
        
        for res_name, W, H, fps, durations in video_configs:
            print(f"\n{res_name} ({W}×{H}) @ {fps}fps:")
            print("-" * 100)
            
            # Calculate model input size (square, rounded to patch size)
            model_size = max(W, H)
            model_size = ((model_size + 15) // 16) * 16
            
            res_results = []
            max_duration = None
            
            for duration_sec in durations:
                frames = fps * duration_sec
                dur_str = format_duration(duration_sec)
                
                print(f"  {dur_str:5s} ({frames:4d} frames)...", end=" ", flush=True)
                
                # Check if model supports this config
                model_fn = models_to_test[model_name]
                try:
                    model = model_fn(model_size, frames)
                    if model is None:
                        print(f"⚠️  Not supported (resolution too high)")
                        res_results.append({
                            'duration_sec': duration_sec,
                            'frames': frames,
                            'success': False,
                            'error': 'Not supported'
                        })
                        break
                except Exception as e:
                    print(f"⚠️  Model creation failed: {str(e)[:30]}")
                    res_results.append({
                        'duration_sec': duration_sec,
                        'frames': frames,
                        'success': False,
                        'error': 'Model creation failed'
                    })
                    break
                
                # Test configuration
                result = test_single_config(model, (1, frames, 3, model_size, model_size), max_vram_mb=max_vram_mb)
                
                if result['success']:
                    print(f"✅ {result['memory_mb']:.0f}MB, {result['latency_ms']:.1f}ms")
                    max_duration = duration_sec
                    res_results.append({
                        'duration_sec': duration_sec,
                        'frames': frames,
                        **result
                    })
                else:
                    print(f"❌ {result['error']}")
                    res_results.append({
                        'duration_sec': duration_sec,
                        'frames': frames,
                        **result
                    })
                    break
            
            model_results[res_name] = {
                'max_duration_sec': max_duration,
                'results': res_results
            }
        
        results[model_name] = model_results
    
    return results


def test_batch_all_models(max_vram_mb=11000):
    """Test ALL models on batch processing."""
    print("\n" + "="*100)
    print("TEST 3: BATCH PROCESSING - ALL MODELS")
    print("="*100)
    
    base_res = 512
    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    
    models_to_test = {
        'CNN': lambda: StandardCNN(num_classes=1000),
        'ViT': lambda: VisionTransformer(img_size=base_res, num_classes=1000, depth=6),
        'AHCN-BlockSparse': lambda: VisionAHCN(img_size=base_res, num_classes=1000, attention_type='block_sparse'),
        'AHCN-Hierarchical': lambda: HighResVisionAHCN(img_size=base_res, num_classes=1000)
    }
    
    results = {}
    
    for model_name, model_fn in models_to_test.items():
        print(f"\n[{model_name} @ {base_res}×{base_res}]")
        print("-" * 100)
        
        model_results = []
        max_batch = None
        
        for bs in batch_sizes:
            print(f"  Batch {bs:3d}...", end=" ", flush=True)
            
            model = model_fn()
            result = test_single_config(model, (bs, 3, base_res, base_res), max_vram_mb=max_vram_mb)
            
            if result['success']:
                throughput = result['throughput'] * bs
                print(f"✅ {result['memory_mb']:.0f}MB, {throughput:.1f} imgs/s")
                max_batch = bs
                model_results.append({
                    'batch_size': bs,
                    'throughput_imgs_sec': throughput,
                    **result
                })
            else:
                print(f"❌ {result['error']}")
                model_results.append({
                    'batch_size': bs,
                    **result
                })
                break
        
        results[model_name] = {
            'max_batch_size': max_batch,
            'results': model_results
        }
    
    return results


# =============================================================================
# COMPREHENSIVE SUMMARY & RATINGS
# =============================================================================

def generate_comprehensive_summary(image_results, video_results, batch_results):
    """Generate complete summary with ratings."""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*100)
    
    # Image Summary
    print("\n[IMAGE CLASSIFICATION - MAXIMUM RESOLUTION]")
    print("-" * 100)
    print(f"{'Model':<20} {'Max Res':>15} {'Memory':>12} {'Latency':>12} {'Rating':>10}")
    print("-" * 100)
    
    for model, data in image_results.items():
        if data['max_resolution']:
            max_result = next((r for r in data['results'] if r['resolution'] == data['max_resolution'] and r.get('success')), None)
            if max_result:
                rating = rate_image_performance(data['max_resolution'], max_result['latency_ms'], max_result['memory_mb'])
                print(f"{model:<20} {data['max_resolution']:>6d}×{data['max_resolution']:<6d} {max_result['memory_mb']:>9.0f}MB {max_result['latency_ms']:>9.1f}ms {rating:>9.1f}/10")
        else:
            print(f"{model:<20} {'FAILED':>15} {'---':>12} {'---':>12} {'0.0':>10}")
    
    # Video Summary
    print("\n[VIDEO PROCESSING - MAXIMUM DURATION PER RESOLUTION]")
    print("-" * 100)
    
    resolutions = ['360p', '480p', '720p', '1080p', '4K']
    
    for res in resolutions:
        print(f"\n{res}:")
        print("-" * 100)
        print(f"{'Model':<20} {'Max Duration':>15} {'Frames':>10} {'Memory':>12} {'Rating':>10}")
        print("-" * 100)
        
        for model, data in video_results.items():
            if res in data:
                res_data = data[res]
                if res_data['max_duration_sec']:
                    max_result = next((r for r in res_data['results'] if r['duration_sec'] == res_data['max_duration_sec'] and r.get('success')), None)
                    if max_result:
                        dur_str = format_duration(res_data['max_duration_sec'])
                        rating = rate_video_performance(res, res_data['max_duration_sec'], max_result.get('latency_ms', 0))
                        print(f"{model:<20} {dur_str:>15} {max_result['frames']:>10d} {max_result['memory_mb']:>9.0f}MB {rating:>9.1f}/10")
                else:
                    print(f"{model:<20} {'FAILED':>15} {'---':>10} {'---':>12} {'0.0':>10}")
    
    # Batch Summary
    print("\n[BATCH PROCESSING @ 512×512]")
    print("-" * 100)
    print(f"{'Model':<20} {'Max Batch':>12} {'Throughput':>15} {'Memory':>12} {'Rating':>10}")
    print("-" * 100)
    
    for model, data in batch_results.items():
        if data['max_batch_size']:
            max_result = next((r for r in data['results'] if r['batch_size'] == data['max_batch_size'] and r.get('success')), None)
            if max_result:
                rating = rate_batch_performance(data['max_batch_size'], max_result['throughput_imgs_sec'])
                print(f"{model:<20} {data['max_batch_size']:>12d} {max_result['throughput_imgs_sec']:>12.1f} i/s {max_result['memory_mb']:>9.0f}MB {rating:>9.1f}/10")
        else:
            print(f"{model:<20} {'FAILED':>12} {'---':>15} {'---':>12} {'0.0':>10}")
    
    # Overall Ratings
    print("\n[OVERALL PERFORMANCE RATINGS]")
    print("-" * 100)
    print(f"{'Model':<20} {'Images':>10} {'Video':>10} {'Batch':>10} {'Average':>10} {'Rank':>8}")
    print("-" * 100)
    
    overall_ratings = calculate_overall_ratings(image_results, video_results, batch_results)
    sorted_models = sorted(overall_ratings.items(), key=lambda x: x[1]['average'], reverse=True)
    
    for i, (model, ratings) in enumerate(sorted_models, 1):
        rank = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        img_str = f"{ratings['image']:.1f}" if ratings['image'] else "N/A"
        vid_str = f"{ratings['video']:.1f}" if ratings['video'] else "N/A"
        bat_str = f"{ratings['batch']:.1f}" if ratings['batch'] else "N/A"
        print(f"{model:<20} {img_str:>10} {vid_str:>10} {bat_str:>10} {ratings['average']:>9.1f}/10 {rank:>8}")


def rate_image_performance(max_res, latency_ms, memory_mb):
    """Rate image performance (1-10)."""
    res_score = min(10, (max_res / 4096) * 10)
    speed_score = min(10, (100 / max(latency_ms, 1)) * 10)
    mem_score = min(10, (5000 / max(memory_mb, 1)) * 5)
    return (res_score * 0.4 + speed_score * 0.4 + mem_score * 0.2)


def rate_video_performance(resolution, max_duration_sec, latency_ms):
    """Rate video performance (1-10)."""
    res_weights = {'360p': 2, '480p': 4, '720p': 6, '1080p': 8, '4K': 10}
    res_score = res_weights.get(resolution, 5)
    
    duration_score = min(10, (max_duration_sec / 120) * 10)
    speed_score = min(10, (1000 / max(latency_ms, 1)) * 10) if latency_ms > 0 else 5
    
    return (res_score * 0.3 + duration_score * 0.5 + speed_score * 0.2)


def rate_batch_performance(max_batch, throughput):
    """Rate batch performance (1-10)."""
    batch_score = min(10, (max_batch / 128) * 10)
    throughput_score = min(10, (throughput / 2000) * 10)
    return (batch_score * 0.5 + throughput_score * 0.5)


def calculate_overall_ratings(image_results, video_results, batch_results):
    """Calculate overall ratings for each model."""
    ratings = {}
    
    # Get all unique models
    all_models = set(list(image_results.keys()) + list(video_results.keys()) + list(batch_results.keys()))
    
    for model in all_models:
        model_ratings = {}
        
        # Image rating
        if model in image_results and image_results[model]['max_resolution']:
            max_res = image_results[model]['max_resolution']
            max_result = next((r for r in image_results[model]['results'] if r['resolution'] == max_res and r.get('success')), None)
            if max_result:
                model_ratings['image'] = rate_image_performance(max_res, max_result['latency_ms'], max_result['memory_mb'])
            else:
                model_ratings['image'] = None
        else:
            model_ratings['image'] = None
        
        # Video rating (average across all resolutions)
        if model in video_results:
            video_scores = []
            for res_name, res_data in video_results[model].items():
                if res_data['max_duration_sec']:
                    max_result = next((r for r in res_data['results'] if r['duration_sec'] == res_data['max_duration_sec'] and r.get('success')), None)
                    if max_result:
                        score = rate_video_performance(res_name, res_data['max_duration_sec'], max_result.get('latency_ms', 0))
                        video_scores.append(score)
            model_ratings['video'] = np.mean(video_scores) if video_scores else None
        else:
            model_ratings['video'] = None
        
        # Batch rating
        if model in batch_results and batch_results[model]['max_batch_size']:
            max_batch = batch_results[model]['max_batch_size']
            max_result = next((r for r in batch_results[model]['results'] if r['batch_size'] == max_batch and r.get('success')), None)
            if max_result:
                model_ratings['batch'] = rate_batch_performance(max_batch, max_result['throughput_imgs_sec'])
            else:
                model_ratings['batch'] = None
        else:
            model_ratings['batch'] = None
        
        # Average (only non-None values)
        valid_scores = [v for v in [model_ratings['image'], model_ratings['video'], model_ratings['batch']] if v is not None]
        model_ratings['average'] = np.mean(valid_scores) if valid_scores else 0.0
        
        ratings[model] = model_ratings
    
    return ratings


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_complete_benchmark(max_vram_mb=11000):
    """Run COMPLETE comprehensive benchmark."""
    
    print("\n" + "="*100)
    print("COMPLETE COMPREHENSIVE BENCHMARK - ALL MODELS, ALL TESTS")
    print("="*100)
    
    if not torch.cuda.is_available():
        print("❌ CUDA required!")
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    
    print(f"\nGPU: {gpu_name}")
    print(f"Total VRAM: {total_vram_mb:.0f}MB ({total_vram_mb/1024:.1f}GB)")
    print(f"Safe Limit: {max_vram_mb}MB ({max_vram_mb/1024:.1f}GB)")
    print(f"\n⚡ Testing ALL models on ALL applicable tasks")
    print(f"⚡ Progressive testing until OOM at each configuration")
    print(f"⚡ Complete comparison matrix\n")
    
    # Run all tests
    image_results = test_image_all_models(max_vram_mb)
    video_results = test_video_all_models_all_resolutions(max_vram_mb)
    batch_results = test_batch_all_models(max_vram_mb)
    
    # Generate summary
    generate_comprehensive_summary(image_results, video_results, batch_results)
    
    # Save results
    complete_results = {
        'gpu': gpu_name,
        'vram_mb': total_vram_mb,
        'safe_limit_mb': max_vram_mb,
        'image': image_results,
        'video': video_results,
        'batch': batch_results
    }
    
    with open('complete_benchmark_results.json', 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print("\n" + "="*100)
    print("✅ COMPLETE BENCHMARK FINISHED")
    print("="*100)
    print(f"\n📁 Results saved to: complete_benchmark_results.json")
    print(f"🏆 Check summary above for rankings and ratings\n")
    
    return complete_results


if __name__ == '__main__':
    # Run with 11GB limit (safe margin on 12GB GPU)
    results = run_complete_benchmark(max_vram_mb=11000)