#!/usr/bin/env python3
import requests
import time
from datetime import datetime

# ─── 설정 ─────────────────────────────────────────────
SYMBOL     = "1000PEPEUSDC"  # Binance 심볼
INVESTMENT = 1000.0          # 투자액 (USD)

# ─── 현재가 조회 함수 ───────────────────────────────────
def get_current_price(symbol: str) -> float:
    """
    Binance Futures API에서 현재 가격을 조회합니다.
    """
    url    = "https://fapi.binance.com/fapi/v1/ticker/price"
    params = {"symbol": symbol}
    resp   = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])

# ─── 메인 루프 ─────────────────────────────────────────
def main():
    prev_price = None
    print("▶ 매초 현재가 변화량 및 이론적 PnL/ROI 모니터링 시작\n")
    while True:
        try:
            price = get_current_price(SYMBOL)
            now   = datetime.now().strftime("%H:%M:%S")

            if prev_price is not None:
                delta      = price - prev_price
                delta_pct  = (delta / prev_price) * 100
                direction  = "Long" if delta > 0 else ("Short" if delta < 0 else "No Change")

                # 이론적 PnL 및 ROI 계산
                pnl_usd    = (delta / prev_price) * INVESTMENT
                roi_pct    = (pnl_usd / INVESTMENT) * 100

                print(
                    f"[{now}] Price={price:.7f} USDC  "
                    f"ΔPrice={delta:+.7f}  "
                    f"ΔPct={delta_pct:+.2f}%  "
                    f"{direction}  "
                    f"PnL={pnl_usd:+.2f} USDC  "
                    f"ROI={roi_pct:+.2f}%"
                )
            else:
                print(f"[{now}] Price={price:.7f} USDC  (초기 조회)")

            prev_price = price

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        time.sleep(10)

if __name__ == "__main__":
    main()
