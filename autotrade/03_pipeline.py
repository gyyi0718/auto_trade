#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Bybit 급등 코인 자동 학습 파이프라인 v2

완전 자동화:
1. 급등 코인 스캔
2. 데이터 다운로드
3. 모델 학습
4. 성능 평가
"""
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
import pandas as pd


class BybitPipeline:
    """Bybit 자동화 파이프라인"""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get('output_dir', './bybit_models')
        self.data_dir = config.get('data_dir', './bybit_data')
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 저장
        self.results = {
            'timestamp': self.timestamp,
            'config': config,
            'symbols': [],
            'data_file': '',
            'model_dir': '',
            'success': False
        }
    
    def log(self, message: str):
        """로그 출력"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def run_command(self, cmd: list) -> bool:
        """명령어 실행"""
        try:
            self.log(f"실행: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ 실패: {e}")
            if e.stderr:
                self.log(f"   에러: {e.stderr[:200]}")
            return False
    
    def step1_scan_coins(self) -> list:
        """1단계: 급등 코인 스캔"""
        self.log("\n" + "="*60)
        self.log("STEP 1: Bybit 급등 코인 스캔")
        self.log("="*60)
        
        try:
            from bybit_scanner import BybitCoinScanner
            
            scanner = BybitCoinScanner()
            df = scanner.scan(
                top_n=self.config['top_n'],
                min_turnover=self.config['min_turnover'],
                min_change=self.config['min_change']
            )
            
            if len(df) == 0:
                self.log("⚠️  급등 코인이 없습니다. 프로세스 중단.")
                return []
            
            scanner.display(df)
            
            # 저장
            scan_file = os.path.join(self.data_dir, f"scan_{self.timestamp}.csv")
            df.to_csv(scan_file, index=False)
            self.log(f"✅ 스캔 결과 저장: {scan_file}")
            
            symbols = df['symbol'].tolist()
            self.results['symbols'] = symbols
            
            # 상세 정보 저장
            self.results['scan_info'] = {
                'count': len(df),
                'avg_change': float(df['change_24h'].mean()),
                'avg_turnover': float(df['turnover_24h'].mean()),
                'symbols': symbols
            }
            
            return symbols
            
        except ImportError:
            self.log("❌ bybit_scanner.py를 찾을 수 없습니다.")
            return []
        except Exception as e:
            self.log(f"❌ 스캔 실패: {e}")
            return []
    
    def step2_download_data(self, symbols: list) -> str:
        """2단계: 데이터 다운로드"""
        self.log("\n" + "="*60)
        self.log("STEP 2: Bybit 데이터 다운로드")
        self.log("="*60)
        
        try:
            from bybit_downloader import BybitDataDownloader
            
            downloader = BybitDataDownloader()
            df = downloader.download_multiple(
                symbols=symbols,
                days=self.config['days'],
                interval=self.config['interval'],
                max_workers=5
            )
            
            if len(df) == 0:
                self.log("⚠️  다운로드된 데이터가 없습니다.")
                return ""
            
            # 데이터 품질 체크
            self.log("\n📊 데이터 품질 체크:")
            for sym in symbols:
                sym_df = df[df['symbol'] == sym]
                self.log(f"   {sym}: {len(sym_df):,}개 캔들")
                
                if len(sym_df) < 1000:
                    self.log(f"      ⚠️  데이터 부족 (최소 1000개 권장)")
            
            # 저장
            data_file = os.path.join(self.data_dir, f"data_{self.timestamp}.parquet")
            downloader.save_data(df, data_file)
            
            self.results['data_file'] = data_file
            self.results['data_info'] = {
                'total_candles': len(df),
                'symbols': df['symbol'].nunique(),
                'date_range': f"{df['date'].min()} ~ {df['date'].max()}"
            }
            
            return data_file
            
        except ImportError:
            self.log("❌ bybit_downloader.py를 찾을 수 없습니다.")
            return ""
        except Exception as e:
            self.log(f"❌ 다운로드 실패: {e}")
            return ""
    
    def step3_train_model(self, data_file: str) -> str:
        """3단계: 모델 학습"""
        self.log("\n" + "="*60)
        self.log("STEP 3: 모델 학습")
        self.log("="*60)
        
        model_dir = os.path.join(self.output_dir, f"model_{self.timestamp}")
        train_script = self.config.get('train_script', 'train_tcn_5minutes.py')
        
        if not os.path.exists(train_script):
            self.log(f"⚠️  학습 스크립트를 찾을 수 없습니다: {train_script}")
            self.log("   train_tcn_5minutes.py를 현재 디렉토리에 준비하세요.")
            return ""
        
        cmd = [
            'python', train_script,
            '--data', data_file,
            '--epochs', str(self.config['epochs']),
            '--seq_len', str(self.config['seq_len']),
            '--horizon', str(self.config['horizon']),
            '--batch', str(self.config['batch']),
            '--lr', str(self.config['lr']),
            '--out_dir', model_dir
        ]
        
        success = self.run_command(cmd)
        
        if success:
            self.log(f"✅ 모델 학습 완료: {model_dir}")
            self.results['model_dir'] = model_dir
            self.results['success'] = True
            return model_dir
        else:
            self.log(f"❌ 학습 실패")
            return ""
    
    def step4_evaluate_model(self, model_dir: str):
        """4단계: 모델 평가 (선택적)"""
        self.log("\n" + "="*60)
        self.log("STEP 4: 모델 평가")
        self.log("="*60)
        
        # 메타 파일 확인
        meta_file = os.path.join(model_dir, "5min_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            self.log("✅ 모델 메타 정보:")
            self.log(f"   시퀀스 길이: {meta.get('seq_len')}")
            self.log(f"   예측 호라이즌: {meta.get('horizon_candles')}")
            self.log(f"   피처 수: {len(meta.get('feat_cols', []))}")
            
            self.results['model_info'] = meta
        else:
            self.log("⚠️  메타 파일을 찾을 수 없습니다.")
    
    def save_results(self):
        """결과 저장"""
        result_file = os.path.join(self.output_dir, f"pipeline_result_{self.timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log(f"\n💾 파이프라인 결과 저장: {result_file}")
    
    def run(self):
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        
        self.log("\n" + "🚀" + "="*58 + "🚀")
        self.log("   Bybit 급등 코인 자동 학습 파이프라인 v2")
        self.log("🚀" + "="*58 + "🚀")
        
        try:
            # 1. 급등 코인 스캔
            symbols = self.step1_scan_coins()
            if not symbols:
                return
            
            # 2. 데이터 다운로드
            data_file = self.step2_download_data(symbols)
            if not data_file:
                return
            
            # 3. 모델 학습
            model_dir = self.step3_train_model(data_file)
            if not model_dir:
                return
            
            # 4. 평가
            self.step4_evaluate_model(model_dir)
            
            # 결과 저장
            self.save_results()
            
            # 완료
            elapsed = (datetime.now() - start_time).total_seconds()
            self.log("\n" + "✅" + "="*58 + "✅")
            self.log(f"   파이프라인 완료! (소요 시간: {elapsed/60:.1f}분)")
            self.log("✅" + "="*58 + "✅\n")
            
            # 요약
            self.log("📊 요약:")
            self.log(f"   급등 코인: {len(symbols)}개")
            self.log(f"   데이터: {self.results.get('data_info', {}).get('total_candles', 0):,}개 캔들")
            self.log(f"   모델: {model_dir}")
            self.log("")
            
        except KeyboardInterrupt:
            self.log("\n\n⚠️  사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.log(f"\n\n❌ 예기치 않은 오류: {e}")
            import traceback
            traceback.print_exc()


def parse_args():
    """인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Bybit 급등 코인 자동 학습 파이프라인 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python 03_pipeline.py --top 5 --days 30
  python 03_pipeline.py --top 10 --days 60 --epochs 50
  python 03_pipeline.py --top 3 --min-change 3.0 --days 14 --epochs 20
        """
    )
    
    # 스캔 설정
    scan_group = parser.add_argument_group('스캔 설정')
    scan_group.add_argument("--top", type=int, default=5,
                           help="상위 N개 급등 코인 (기본값: 5)")
    scan_group.add_argument("--min-turnover", type=float, default=5_000_000,
                           help="최소 거래대금 (기본값: 5M)")
    scan_group.add_argument("--min-change", type=float, default=5.0,
                           help="최소 변화율 (기본값: 5%%)")
    
    # 데이터 설정
    data_group = parser.add_argument_group('데이터 설정')
    data_group.add_argument("--days", type=int, default=30,
                           help="며칠치 데이터 (기본값: 30)")
    data_group.add_argument("--interval", type=str, default="5",
                           help="시간봉 (1, 5, 15, 60) (기본값: 5)")
    
    # 학습 설정
    train_group = parser.add_argument_group('학습 설정')
    train_group.add_argument("--epochs", type=int, default=30,
                            help="에포크 수 (기본값: 30)")
    train_group.add_argument("--seq-len", type=int, default=72,
                            help="시퀀스 길이 (기본값: 72)")
    train_group.add_argument("--horizon", type=int, default=72,
                            help="예측 호라이즌 (기본값: 72)")
    train_group.add_argument("--batch", type=int, default=512,
                            help="배치 크기 (기본값: 512)")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="학습률 (기본값: 0.0001)")
    
    # 출력 설정
    output_group = parser.add_argument_group('출력 설정')
    output_group.add_argument("--output-dir", type=str, default="./bybit_models",
                             help="모델 출력 디렉토리")
    output_group.add_argument("--data-dir", type=str, default="./bybit_data",
                             help="데이터 디렉토리")
    output_group.add_argument("--train-script", type=str, 
                             default="train_tcn_5minutes.py",
                             help="학습 스크립트 경로")
    
    return parser.parse_args()


def main():
    """메인 실행"""
    args = parse_args()
    
    # 설정 생성
    config = {
        'top_n': args.top,
        'min_turnover': args.min_turnover,
        'min_change': args.min_change,
        'days': args.days,
        'interval': args.interval,
        'epochs': args.epochs,
        'seq_len': args.seq_len,
        'horizon': args.horizon,
        'batch': args.batch,
        'lr': args.lr,
        'output_dir': args.output_dir,
        'data_dir': args.data_dir,
        'train_script': args.train_script
    }
    
    # 파이프라인 실행
    pipeline = BybitPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()