#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 체크포인트 구조 분석 스크립트
"""
import torch
import json

# 체크포인트 로드
ckpt_path = "./models_patchtst/patchtst_best.ckpt"
print(f"체크포인트 로딩: {ckpt_path}")
print("=" * 80)

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print("\n📦 체크포인트 구조:")
    print(f"  Keys: {list(ckpt.keys())}")
    
    if 'meta' in ckpt:
        print("\n📋 Meta 정보:")
        for k, v in ckpt['meta'].items():
            print(f"  {k}: {v}")
    
    if 'feat_cols' in ckpt:
        print(f"\n🔧 Feature 개수: {len(ckpt['feat_cols'])}")
        print(f"  Features: {ckpt['feat_cols'][:5]}...")
    
    print("\n🧠 모델 State Dict 구조:")
    print("-" * 80)
    
    model_state = ckpt['model']
    
    # 레이어별로 정리
    layers = {}
    for key in model_state.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            layer_name = '.'.join(parts[:2])
            if layer_name not in layers:
                layers[layer_name] = []
            layers[layer_name].append(key)
        else:
            if 'other' not in layers:
                layers['other'] = []
            layers['other'].append(key)
    
    # 출력
    for layer_name in sorted(layers.keys()):
        print(f"\n[{layer_name}]")
        for key in sorted(layers[layer_name]):
            shape = model_state[key].shape
            dtype = model_state[key].dtype
            print(f"  {key:50s} {str(shape):30s} {dtype}")
    
    # Classifier/Head 구조 상세 분석
    print("\n" + "=" * 80)
    print("🎯 Classifier 구조 상세 분석:")
    print("=" * 80)
    
    classifier_keys = [k for k in model_state.keys() if 'classifier' in k or 'head' in k]
    
    if classifier_keys:
        print("\nClassifier/Head 레이어:")
        for key in sorted(classifier_keys):
            shape = model_state[key].shape
            print(f"  {key:50s} shape={shape}")
            
        # 구조 추론
        print("\n추론된 Classifier 구조:")
        
        # 레이어 번호별로 그룹화
        layer_groups = {}
        for key in classifier_keys:
            if 'classifier' in key:
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    if layer_idx not in layer_groups:
                        layer_groups[layer_idx] = {}
                    param_type = parts[2] if len(parts) >= 3 else 'unknown'
                    layer_groups[layer_idx][param_type] = model_state[key].shape
        
        for idx in sorted(layer_groups.keys()):
            params = layer_groups[idx]
            print(f"\n  Layer {idx}:")
            for param_name, shape in params.items():
                print(f"    {param_name}: {shape}")
            
            # 레이어 타입 추론
            if 'weight' in params:
                weight_shape = params['weight']
                if len(weight_shape) == 1:
                    print(f"    → LayerNorm (features={weight_shape[0]})")
                elif len(weight_shape) == 2:
                    print(f"    → Linear (in={weight_shape[1]}, out={weight_shape[0]})")
    else:
        print("  ⚠️  Classifier/Head 레이어를 찾을 수 없습니다!")
    
    # 정확한 모델 코드 생성
    print("\n" + "=" * 80)
    print("📝 정확한 모델 구조 코드:")
    print("=" * 80)
    
    if classifier_keys:
        print("\nself.classifier = nn.Sequential(")
        for idx in sorted(layer_groups.keys()):
            params = layer_groups[idx]
            if 'weight' in params:
                weight_shape = params['weight']
                if len(weight_shape) == 1:
                    print(f"    nn.LayerNorm({weight_shape[0]}),")
                elif len(weight_shape) == 2:
                    print(f"    nn.Linear({weight_shape[1]}, {weight_shape[0]}),")
            
            # ReLU나 Dropout이 있을 수도 있음 (추론)
            if idx < max(layer_groups.keys()) - 1:  # 마지막이 아니면
                if len(weight_shape) == 2 and weight_shape[0] != 2:  # 출력이 2가 아니면
                    print(f"    nn.ReLU(),")
        print(")")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"❌ 에러: {e}")
    import traceback
    traceback.print_exc()
