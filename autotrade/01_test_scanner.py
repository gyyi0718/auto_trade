#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit 스캐너 사용 예시
"""

def example_basic():
    """기본 사용"""
    print("\n" + "="*60)
    print("예시 1: 기본 사용")
    print("="*60)
    
    from bybit_scanner import BybitCoinScanner
    
    scanner = BybitCoinScanner()
    df = scanner.scan(top_n=5)
    scanner.display(df)
    
    return df


def example_get_symbols():
    """심볼 리스트 추출"""
    print("\n" + "="*60)
    print("예시 2: 심볼 리스트 추출")
    print("="*60)
    
    from bybit_scanner import BybitCoinScanner
    
    scanner = BybitCoinScanner()
    df = scanner.scan(top_n=5, min_change=3.0)
    
    if len(df) > 0:
        symbols = df['symbol'].tolist()
        print(f"\n📋 심볼 리스트: {symbols}\n")
        return symbols
    else:
        print("\n⚠️  급등 코인 없음\n")
        return []


def example_filter():
    """고거래량 필터"""
    print("\n" + "="*60)
    print("예시 3: 고거래량 코인만 (20M USDT)")
    print("="*60)
    
    from bybit_scanner import BybitCoinScanner
    
    scanner = BybitCoinScanner()
    df = scanner.scan(
        top_n=10,
        min_turnover=20_000_000,  # 20M USDT
        min_change=2.0
    )
    scanner.display(df)
    
    return df


def example_save():
    """결과 저장"""
    print("\n" + "="*60)
    print("예시 4: 결과 CSV 저장")
    print("="*60)
    
    from bybit_scanner import BybitCoinScanner
    
    scanner = BybitCoinScanner()
    df = scanner.scan(top_n=5)
    
    if len(df) > 0:
        filename = "top_coins.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ 저장 완료: {filename}\n")
        
        # 확인
        print("저장된 컬럼:")
        print(df.columns.tolist())
    
    return df


def show_usage():
    """사용법 출력"""
    print("\n" + "🚀"*30)
    print("   Bybit 스캐너 사용 예시")
    print("🚀"*30)
    
    examples = """
1️⃣  기본 실행
   $ python bybit_scanner.py

2️⃣  Top 5 선택
   $ python bybit_scanner.py --top 5

3️⃣  고거래량 (20M USDT)
   $ python bybit_scanner.py --min-turnover 20000000

4️⃣  작은 변화도 포함 (2%)
   $ python bybit_scanner.py --min-change 2.0

5️⃣  저장
   $ python bybit_scanner.py --top 5 --save coins.csv

6️⃣  Python 코드
   from bybit_scanner import BybitCoinScanner
   
   scanner = BybitCoinScanner()
   df = scanner.scan(top_n=5)
   symbols = df['symbol'].tolist()

7️⃣  크론잡 (매일 오전 9시)
   0 9 * * * python bybit_scanner.py --save daily.csv
    """
    
    print(examples)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            example_basic()
        elif sys.argv[1] == "2":
            example_get_symbols()
        elif sys.argv[1] == "3":
            example_filter()
        elif sys.argv[1] == "4":
            example_save()
        else:
            show_usage()
    else:
        show_usage()
        
        # 실제 실행 여부 확인
        print("\n실제로 스캔을 실행하시겠습니까? (y/n): ", end='')
        
        try:
            answer = input().lower()
            if answer == 'y':
                example_basic()
        except:
            print("\n종료")