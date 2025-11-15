# paper_trading.py
# -*- coding: utf-8 -*-
"""
Paper Trading - 학습된 모델로 실시간 시뮬레이션
"""
import numpy as np
import pandas as pd
import time
from datetime import datetime
from stable_baselines3 import PPO
from rl_trading_env_final import CryptoTradingEnv
import requests


def fetch_latest_data(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """바이비트에서 최신 데이터 가져오기"""
    url = "https://api.bybit.com/v5/market/kline"
    
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("retCode") != 0:
        raise Exception(f"API 오류: {data.get('retMsg')}")
    
    candles = data["result"]["list"]
    
    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def paper_trading(
    model_path: str,
    symbol: str = "BTCUSDT",
    interval: str = "5",
    initial_balance: float = 10000,
    duration_minutes: int = 60,
    update_interval: int = 5
):
    """
    Paper Trading 실행
    
    Args:
        model_path: 학습된 모델 경로
        symbol: 거래 심볼
        interval: 시간 간격 (분)
        initial_balance: 초기 자본
        duration_minutes: 실행 시간 (분)
        update_interval: 업데이트 간격 (분)
    """
    print("\n" + "=" * 80)
    print(f"{'📊 Paper Trading 시작':^80}")
    print("=" * 80)
    print(f"\n설정:")
    print(f"   모델: {model_path}")
    print(f"   심볼: {symbol}")
    print(f"   간격: {interval}분")
    print(f"   초기 자본: ${initial_balance:,.0f}")
    print(f"   실행 시간: {duration_minutes}분")
    print(f"   업데이트: 매 {update_interval}분\n")
    
    # 모델 로드
    print("🔄 모델 로드 중...")
    model = PPO.load(model_path)
    print("✅ 모델 로드 완료\n")
    
    # 초기 데이터
    print("📥 초기 데이터 다운로드 중...")
    df = fetch_latest_data(symbol, interval, limit=500)
    print(f"✅ {len(df)}개 캔들 로드\n")
    
    # 환경 생성
    env = CryptoTradingEnv(
        df=df,
        window_size=30,
        initial_balance=initial_balance,
        leverage=10,
        commission=0.0006,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
        min_holding_steps=3,
        force_initial_position=True,
        debug=True
    )
    
    obs = env.reset()
    
    print("=" * 80)
    print(f"{'🚀 거래 시작!':^80}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            elapsed = (time.time() - start_time) / 60
            if elapsed >= duration_minutes:
                break
            
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 모델 예측
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            # 액션 이름
            action_name = ['LONG', 'SHORT', 'CLOSE'][action]
            
            # 환경 스텝
            obs, reward, done, info = env.step(action)
            
            # 현재 상태 출력
            print(f"[{current_time}] #{iteration}")
            print(f"  액션: {action_name}")
            print(f"  포지션: {info['position'] or 'None'}")
            print(f"  자산: ${info['equity']:,.2f}")
            print(f"  총 거래: {info['total_trades']}회")
            if info['total_trades'] > 0:
                print(f"  승률: {info['win_rate']*100:.1f}%")
                print(f"  손익: ${info['pnl']:,.2f} ({(info['equity']/initial_balance-1)*100:+.2f}%)")
            print()
            
            if done:
                print("⚠️  에피소드 종료 - 데이터 갱신 중...")
                
                # 새 데이터 다운로드
                df = fetch_latest_data(symbol, interval, limit=500)
                
                # 환경 재생성
                env = CryptoTradingEnv(
                    df=df,
                    window_size=30,
                    initial_balance=info['equity'],  # 현재 자산으로 계속
                    leverage=10,
                    commission=0.0006,
                    stop_loss_pct=0.05,
                    take_profit_pct=0.08,
                    min_holding_steps=3,
                    force_initial_position=True,
                    debug=True
                )
                
                obs = env.reset()
                print("✅ 데이터 갱신 완료\n")
            
            # 대기
            time.sleep(update_interval * 60)
            
    except KeyboardInterrupt:
        print("\n\n⏸️  사용자가 중단했습니다.")
    
    # 최종 통계
    stats = env.get_stats()
    
    print("\n" + "=" * 80)
    print(f"{'📊 최종 결과':^80}")
    print("=" * 80)
    print(f"\n실행 시간: {elapsed:.1f}분")
    print(f"총 반복: {iteration}회\n")
    
    print("성과:")
    print(f"   초기 자산: ${initial_balance:,.2f}")
    print(f"   최종 자산: ${stats['final_equity']:,.2f}")
    print(f"   총 수익률: {stats['total_return']:+.2f}%")
    print(f"   총 손익: ${stats['total_pnl']:,.2f}")
    print(f"\n거래:")
    print(f"   총 거래: {stats['total_trades']}회")
    print(f"   승률: {stats['win_rate']:.1f}%")
    print(f"   최대 낙폭: {stats['max_drawdown']:.2f}%")
    print(f"   샤프 비율: {stats['sharpe_ratio']:.2f}")
    
    if env.trade_history:
        print(f"\n거래 내역:")
        for i, trade in enumerate(env.trade_history[-10:], 1):  # 최근 10개
            print(f"   #{i}: {trade['direction'].upper():5s} | "
                  f"Entry: {trade['entry_price']:8.2f} | "
                  f"Exit: {trade['exit_price']:8.2f} | "
                  f"PNL: {trade['pnl']:+8.2f} ({trade['pnl_pct']*100:+5.2f}%) | "
                  f"{trade['reason']}")
    
    print("\n" + "=" * 80)
    print(f"{'✅ Paper Trading 완료!':^80}")
    print("=" * 80)


def backtest_recent_data(
    model_path: str,
    symbol: str = "BTCUSDT",
    interval: str = "5",
    initial_balance: float = 10000
):
    """
    최근 데이터로 백테스트 (실시간 대기 없이)
    """
    print("\n" + "=" * 80)
    print(f"{'📊 백테스트 (최근 데이터)':^80}")
    print("=" * 80)
    
    # 모델 로드
    print(f"\n🔄 모델 로드: {model_path}")
    model = PPO.load(model_path)
    
    # 최신 데이터
    print(f"📥 {symbol} {interval}분봉 최신 데이터 다운로드 중...")
    df = fetch_latest_data(symbol, interval, limit=2000)
    print(f"✅ {len(df)}개 캔들 로드")
    
    # 테스트 기간 표시
    start_date = df['timestamp'].iloc[0]
    end_date = df['timestamp'].iloc[-1]
    print(f"   기간: {start_date} ~ {end_date}\n")
    
    # 환경 생성
    env = CryptoTradingEnv(
        df=df,
        window_size=30,
        initial_balance=initial_balance,
        leverage=10,
        commission=0.0006,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
        min_holding_steps=3,
        force_initial_position=True,
        debug=False  # 백테스트는 로그 끄기
    )
    
    obs = env.reset()
    done = False
    
    action_counts = {0: 0, 1: 0, 2: 0}
    
    print("🚀 백테스트 실행 중...\n")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1
        obs, reward, done, info = env.step(action)
    
    # 결과
    stats = env.get_stats()
    
    print("=" * 80)
    print(f"{'📊 백테스트 결과':^80}")
    print("=" * 80)
    
    print(f"\n성과:")
    print(f"   초기 자산: ${initial_balance:,.2f}")
    print(f"   최종 자산: ${stats['final_equity']:,.2f}")
    print(f"   총 수익률: {stats['total_return']:+.2f}%")
    print(f"   총 손익: ${stats['total_pnl']:,.2f}")
    
    print(f"\n거래:")
    print(f"   총 거래: {stats['total_trades']}회")
    print(f"   승률: {stats['win_rate']:.1f}%")
    print(f"   최대 낙폭: {stats['max_drawdown']:.2f}%")
    print(f"   샤프 비율: {stats['sharpe_ratio']:.2f}")
    
    print(f"\n행동 분포:")
    total = sum(action_counts.values())
    print(f"   LONG:  {action_counts[0]} ({action_counts[0]/total*100:.1f}%)")
    print(f"   SHORT: {action_counts[1]} ({action_counts[1]/total*100:.1f}%)")
    print(f"   CLOSE: {action_counts[2]} ({action_counts[2]/total*100:.1f}%)")
    
    if env.trade_history:
        print(f"\n최근 거래 (최근 10개):")
        for i, trade in enumerate(env.trade_history[-10:], 1):
            print(f"   #{len(env.trade_history)-10+i}: {trade['direction'].upper():5s} | "
                  f"Entry: {trade['entry_price']:9.2f} | "
                  f"Exit: {trade['exit_price']:9.2f} | "
                  f"PNL: {trade['pnl']:+9.2f} ({trade['pnl_pct']*100:+6.2f}%) | "
                  f"보유: {trade['holding_time']:3d}스텝 | "
                  f"{trade['reason']}")
    
    print("\n" + "=" * 80)
    print(f"{'✅ 백테스트 완료!':^80}")
    print("=" * 80)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading & Backtest')
    parser.add_argument('--mode', type=str, default='backtest', 
                       choices=['paper', 'backtest'],
                       help='paper: 실시간 시뮬레이션, backtest: 최근 데이터 백테스트')
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--interval', type=str, default='5', help='시간 간격 (분)')
    parser.add_argument('--balance', type=float, default=10000, help='초기 자본')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Paper trading 실행 시간 (분)')
    parser.add_argument('--update', type=int, default=5, 
                       help='Paper trading 업데이트 간격 (분)')
    
    args = parser.parse_args()
    
    if args.mode == 'paper':
        # 실시간 Paper Trading
        paper_trading(
            model_path=args.model,
            symbol=args.symbol,
            interval=args.interval,
            initial_balance=args.balance,
            duration_minutes=args.duration,
            update_interval=args.update
        )
    else:
        # 백테스트
        backtest_recent_data(
            model_path=args.model,
            symbol=args.symbol,
            interval=args.interval,
            initial_balance=args.balance
        )
