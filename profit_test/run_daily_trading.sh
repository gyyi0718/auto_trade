#!/bin/bash
# run_daily_trading.sh
# TCN Daily 페이퍼 트레이딩 실행 스크립트

# 설정
export SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT"
export TCN_CKPT="./models_daily_v2/daily_simple_best.ckpt"
export INITIAL_CAPITAL="10000"
export LEVERAGE="10"
export CONF_THRESHOLD="0.55"
export POSITION_SIZE_PCT="0.1"
export MAX_POSITIONS="3"
export STOP_LOSS_PCT="0.02"
export TAKE_PROFIT_PCT="0.03"
export MAX_HOLD_HOURS="24"
export INTERVAL_SEC="60"
export USE_TESTNET="0"
export TRADE_LOG_FILE="trades_daily.json"

echo "=========================================="
echo "  TCN Daily 페이퍼 트레이딩 시작"
echo "=========================================="
echo "심볼: $SYMBOLS"
echo "모델: $TCN_CKPT"
echo "초기 자본: $$INITIAL_CAPITAL"
echo "레버리지: ${LEVERAGE}x"
echo "신뢰도 임계값: $CONF_THRESHOLD"
echo "=========================================="
echo ""

# 실행
python paper_trading_daily.py