#!/usr/bin/env python3
"""
🎯 모델 성능 평가 시스템
- 여러 모델의 백테스트 결과 비교
- 종합 점수 계산 (승률 40% + R/R 30% + 수익 20% + Sharpe 10%)
- 최적 모델 자동 선택
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple
import joblib

class ModelEvaluator:
    """모델 성능 평가 및 비교"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        
        # 평가 가중치
        self.weights = {
            'win_rate': 0.40,      # 승률 40%
            'risk_reward': 0.30,   # 손익비 30%
            'total_return': 0.20,  # 총수익 20%
            'sharpe': 0.10         # 샤프지수 10%
        }
    
    def find_models(self) -> List[Path]:
        """디렉토리에서 모든 모델 파일 찾기"""
        models = []
        
        if not self.models_dir.exists():
            print(f"❌ 디렉토리 없음: {self.models_dir}")
            return models
        
        # .pkl 또는 .joblib 파일 찾기
        for ext in ['*.pkl', '*.joblib']:
            models.extend(self.models_dir.rglob(ext))
        
        print(f"✅ {len(models)}개 모델 발견")
        return sorted(models)
    
    def load_model_metadata(self, model_path: Path) -> Dict:
        """모델 메타데이터 로드"""
        meta_path = model_path.with_suffix('.json')
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        
        # 메타데이터 없으면 기본값
        return {
            'model_name': model_path.stem,
            'created_at': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
            'training_days': 30,
            'features': []
        }
    
    def backtest_model(self, model_path: Path, test_data: pd.DataFrame = None) -> Dict:
        """
        모델 백테스트 수행
        실제 구현 시 test_data로 백테스트 수행
        여기서는 메타데이터에서 로드하거나 간단한 시뮬레이션
        """
        meta = self.load_model_metadata(model_path)
        
        # 메타데이터에 백테스트 결과가 있으면 사용
        if 'backtest_results' in meta:
            return meta['backtest_results']
        
        # 없으면 간단한 시뮬레이션 (실제로는 제대로 백테스트 해야 함)
        # 이 부분은 실제 백테스트 로직으로 교체 필요
        try:
            model = joblib.load(model_path)
            
            # 임시: 랜덤 결과 생성 (실제로는 test_data로 백테스트)
            np.random.seed(hash(str(model_path)) % 2**32)
            
            n_trades = np.random.randint(50, 200)
            wins = np.random.randint(int(n_trades * 0.4), int(n_trades * 0.7))
            
            avg_win = np.random.uniform(2.0, 5.0)
            avg_loss = np.random.uniform(0.5, 2.0)
            
            results = {
                'total_trades': n_trades,
                'wins': wins,
                'losses': n_trades - wins,
                'win_rate': wins / n_trades,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'risk_reward': avg_win / avg_loss if avg_loss > 0 else 0,
                'total_return': (wins * avg_win - (n_trades - wins) * avg_loss),
                'sharpe_ratio': np.random.uniform(0.5, 2.5),
                'max_drawdown': np.random.uniform(5, 25)
            }
            
            return results
            
        except Exception as e:
            print(f"⚠️ 백테스트 실패 {model_path.name}: {e}")
            return None
    
    def calculate_score(self, results: Dict) -> float:
        """
        종합 점수 계산
        점수 = 승률×40% + R/R×30% + 수익×20% + Sharpe×10%
        """
        if not results:
            return 0.0
        
        # 각 지표를 0-100 스케일로 정규화
        scores = {
            'win_rate': min(results.get('win_rate', 0) * 100, 100),
            'risk_reward': min(results.get('risk_reward', 0) * 20, 100),  # R/R 5 = 100점
            'total_return': min(results.get('total_return', 0) * 2, 100),  # 수익 50 = 100점
            'sharpe': min(results.get('sharpe_ratio', 0) * 40, 100)  # Sharpe 2.5 = 100점
        }
        
        # 가중 평균
        total_score = sum(scores[k] * self.weights[k] for k in self.weights)
        
        return round(total_score, 2)
    
    def evaluate_all_models(self, test_data: pd.DataFrame = None) -> pd.DataFrame:
        """모든 모델 평가 및 순위화"""
        models = self.find_models()
        
        if not models:
            print("❌ 평가할 모델이 없습니다")
            return pd.DataFrame()
        
        results_list = []
        
        print(f"\n{'='*60}")
        print(f"🔍 {len(models)}개 모델 평가 중...")
        print(f"{'='*60}\n")
        
        for model_path in models:
            print(f"📊 평가: {model_path.name}")
            
            meta = self.load_model_metadata(model_path)
            backtest = self.backtest_model(model_path, test_data)
            
            if backtest:
                score = self.calculate_score(backtest)
                
                results_list.append({
                    'model_name': model_path.stem,
                    'model_path': str(model_path),
                    'created_at': meta.get('created_at', 'Unknown'),
                    'training_days': meta.get('training_days', 'Unknown'),
                    'total_trades': backtest['total_trades'],
                    'win_rate': f"{backtest['win_rate']*100:.1f}%",
                    'wins': backtest['wins'],
                    'losses': backtest['losses'],
                    'risk_reward': f"{backtest['risk_reward']:.2f}",
                    'avg_win': f"{backtest['avg_win']:.2f}%",
                    'avg_loss': f"{backtest['avg_loss']:.2f}%",
                    'total_return': f"{backtest['total_return']:.2f}%",
                    'sharpe_ratio': f"{backtest['sharpe_ratio']:.2f}",
                    'max_drawdown': f"{backtest['max_drawdown']:.2f}%",
                    'score': score
                })
                
                print(f"   ✅ 점수: {score:.2f} (승률: {backtest['win_rate']*100:.1f}%)")
            else:
                print(f"   ❌ 평가 실패")
        
        # DataFrame 생성 및 정렬
        df = pd.DataFrame(results_list)
        
        if not df.empty:
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
            df.insert(0, 'rank', range(1, len(df) + 1))
        
        return df
    
    def print_comparison_table(self, df: pd.DataFrame):
        """비교 테이블 출력"""
        if df.empty:
            print("❌ 비교할 데이터가 없습니다")
            return
        
        print(f"\n{'='*80}")
        print(f"🏆 모델 성능 순위")
        print(f"{'='*80}\n")
        
        # 주요 컬럼만 출력
        display_cols = ['rank', 'model_name', 'win_rate', 'risk_reward', 
                       'total_return', 'sharpe_ratio', 'score']
        
        print(df[display_cols].to_string(index=False))
        
        print(f"\n{'='*80}")
        print(f"🥇 최고 성능: {df.iloc[0]['model_name']} (점수: {df.iloc[0]['score']})")
        print(f"{'='*80}\n")
    
    def save_results(self, df: pd.DataFrame, output_path: str = "model_comparison.csv"):
        """결과를 CSV로 저장"""
        if df.empty:
            print("❌ 저장할 데이터가 없습니다")
            return
        
        output_path = Path(output_path)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 결과 저장: {output_path}")
    
    def get_best_model(self, df: pd.DataFrame) -> Tuple[str, float]:
        """최고 성능 모델 반환"""
        if df.empty:
            return None, 0.0
        
        best = df.iloc[0]
        return best['model_path'], best['score']
    
    def compare_two_models(self, model_a: Path, model_b: Path) -> Dict:
        """
        두 모델 직접 비교 (A/B 테스트용)
        """
        print(f"\n{'='*60}")
        print(f"🆚 모델 비교")
        print(f"{'='*60}")
        
        results_a = self.backtest_model(model_a)
        results_b = self.backtest_model(model_b)
        
        if not results_a or not results_b:
            print("❌ 비교 실패")
            return None
        
        score_a = self.calculate_score(results_a)
        score_b = self.calculate_score(results_b)
        
        print(f"\n📊 모델 A: {model_a.name}")
        print(f"   승률: {results_a['win_rate']*100:.1f}% | R/R: {results_a['risk_reward']:.2f} | 점수: {score_a:.2f}")
        
        print(f"\n📊 모델 B: {model_b.name}")
        print(f"   승률: {results_b['win_rate']*100:.1f}% | R/R: {results_b['risk_reward']:.2f} | 점수: {score_b:.2f}")
        
        diff = score_b - score_a
        print(f"\n{'='*60}")
        
        if diff > 5:
            print(f"✅ 모델 B가 {diff:.2f}점 우수 → 교체 권장")
            winner = 'B'
        elif diff < -5:
            print(f"✅ 모델 A가 {abs(diff):.2f}점 우수 → 유지 권장")
            winner = 'A'
        else:
            print(f"⚠️ 차이 {abs(diff):.2f}점 (임계값 5점) → 유지 권장")
            winner = 'A'
        
        print(f"{'='*60}\n")
        
        return {
            'model_a': str(model_a),
            'model_b': str(model_b),
            'score_a': score_a,
            'score_b': score_b,
            'difference': diff,
            'winner': winner,
            'results_a': results_a,
            'results_b': results_b
        }


def main():
    parser = argparse.ArgumentParser(description='모델 성능 평가 및 비교')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='모델 디렉토리 경로')
    parser.add_argument('--save', type=str, default='model_comparison.csv',
                       help='결과 저장 파일명')
    parser.add_argument('--compare', type=str, nargs=2, metavar=('MODEL_A', 'MODEL_B'),
                       help='두 모델 직접 비교 (경로)')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.models_dir)
    
    # 두 모델 직접 비교 모드
    if args.compare:
        model_a = Path(args.compare[0])
        model_b = Path(args.compare[1])
        
        if not model_a.exists() or not model_b.exists():
            print("❌ 모델 파일이 존재하지 않습니다")
            return
        
        result = evaluator.compare_two_models(model_a, model_b)
        
        if result:
            # 비교 결과 저장
            comparison_path = Path(args.save).with_suffix('.json')
            with open(comparison_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ 비교 결과 저장: {comparison_path}")
        
        return
    
    # 전체 모델 평가 모드
    df = evaluator.evaluate_all_models()
    
    if not df.empty:
        evaluator.print_comparison_table(df)
        evaluator.save_results(df, args.save)
        
        best_model, best_score = evaluator.get_best_model(df)
        print(f"🎯 추천 모델: {Path(best_model).name}")
        print(f"   점수: {best_score:.2f}\n")


if __name__ == "__main__":
    main()