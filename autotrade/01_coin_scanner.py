#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit 급등 코인 스캐너
- Bybit V5 API 사용
- USDT 선물 (Linear Perpetual) 중심
"""
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict
import argparse


class BybitCoinScanner:
    """Bybit 급등 코인 스캐너"""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        
    def get_tickers(self) -> List[Dict]:
        """24시간 티커 정보 가져오기"""
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": "linear"}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('retCode') == 0:
                return data.get('result', {}).get('list', [])
            else:
                print(f"[ERROR] API 에러: {data.get('retMsg')}")
                return []
        except Exception as e:
            print(f"[ERROR] 요청 실패: {e}")
            return []
    
    def filter_usdt(self, tickers: List[Dict]) -> List[Dict]:
        """USDT 페어만 필터링"""
        return [
            t for t in tickers 
            if t.get('symbol', '').endswith('USDT')
        ]
    
    def calculate_metrics(self, tickers: List[Dict]) -> pd.DataFrame:
        """메트릭 계산"""
        data = []
        
        for t in tickers:
            try:
                symbol = t.get('symbol', '')
                price = float(t.get('lastPrice', 0))
                change_pct = float(t.get('price24hPcnt', 0)) * 100
                turnover = float(t.get('turnover24h', 0))
                high = float(t.get('highPrice24h', 0))
                low = float(t.get('lowPrice24h', 0))
                
                # 변동성
                volatility = ((high - low) / low * 100) if low > 0 else 0
                
                data.append({
                    'symbol': symbol,
                    'change_24h': change_pct,
                    'turnover_24h': turnover,
                    'volatility': volatility,
                    'price': price,
                })
            except:
                continue
        
        return pd.DataFrame(data)
    
    def apply_filters(self, df: pd.DataFrame, 
                     min_turnover: float = 5_000_000,
                     min_change: float = 5.0) -> pd.DataFrame:
        """필터링"""
        if len(df) == 0:
            return df
        
        return df[
            (df['turnover_24h'] >= min_turnover) &
            (df['change_24h'] >= min_change)
        ].copy()
    
    def rank_coins(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """랭킹"""
        if len(df) == 0:
            return df
        
        # 정규화
        df['turnover_score'] = (df['turnover_24h'] - df['turnover_24h'].min()) / \
                               (df['turnover_24h'].max() - df['turnover_24h'].min() + 1)
        
        # 점수
        df['score'] = (
            df['change_24h'] * 0.5 +
            df['turnover_score'] * 100 * 0.3 +
            df['volatility'] * 0.2
        )
        
        return df.sort_values('score', ascending=False).head(top_n)
    
    def scan(self, top_n: int = 10, 
             min_turnover: float = 5_000_000,
             min_change: float = 5.0) -> pd.DataFrame:
        """전체 스캔"""
        print(f"\n{'='*60}")
        print(f"🔍 Bybit 급등 코인 스캔 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # 1. 데이터 수집
        print("[1/4] 티커 데이터 수집 중...")
        tickers = self.get_tickers()
        print(f"   ✓ {len(tickers)}개 수집")
        
        if not tickers:
            return pd.DataFrame()
        
        # 2. USDT 필터링
        print("[2/4] USDT 페어 필터링...")
        usdt_tickers = self.filter_usdt(tickers)
        print(f"   ✓ {len(usdt_tickers)}개")
        
        # 3. 메트릭 계산
        print("[3/4] 분석 중...")
        df = self.calculate_metrics(usdt_tickers)
        df = self.apply_filters(df, min_turnover, min_change)
        print(f"   ✓ {len(df)}개 코인 통과")
        
        # 4. 랭킹
        print(f"[4/4] Top {top_n} 선택...")
        df = self.rank_coins(df, top_n)
        
        print(f"\n✅ 완료 - {len(df)}개 발견\n")
        return df
    
    def display(self, df: pd.DataFrame):
        """결과 출력"""
        if len(df) == 0:
            print("급등 코인이 없습니다.")
            return
        
        print(f"{'='*80}")
        print(f"🔥 급등 코인 TOP {len(df)}")
        print(f"{'='*80}")
        print(f"{'순위':<4} {'심볼':<15} {'24h 변화':<12} {'거래대금(USDT)':<18} {'점수':<10}")
        print(f"{'-'*80}")
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"{i:<4} {row['symbol']:<15} "
                  f"{row['change_24h']:>+10.2f}% "
                  f"${row['turnover_24h']:>15,.0f} "
                  f"{row['score']:>9.2f}")
        
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Bybit 급등 코인 스캐너")
    parser.add_argument("--top", type=int, default=10, help="상위 N개")
    parser.add_argument("--min-turnover", type=float, default=5_000_000, help="최소 거래대금")
    parser.add_argument("--min-change", type=float, default=5.0, help="최소 변화율")
    parser.add_argument("--save", type=str, help="결과 저장 경로")
    
    args = parser.parse_args()
    
    scanner = BybitCoinScanner()
    df = scanner.scan(
        top_n=args.top,
        min_turnover=args.min_turnover,
        min_change=args.min_change
    )
    
    scanner.display(df)
    
    if args.save and len(df) > 0:
        df.to_csv(args.save, index=False)
        print(f"✅ 저장: {args.save}\n")


if __name__ == "__main__":
    main()