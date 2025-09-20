# analyze_trades.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 경로
TRADES_CSV = "paper_trades_allmodes_leverage10_1000.csv"  # 실제 파일명으로 바꿔주세요

df = pd.read_csv(TRADES_CSV)

# EXIT 이벤트만 분석
exits = df[df["event"].str.startswith("EXIT")].copy()
exits["pnl_usdt"] = exits["pnl_usdt"].astype(float)
exits["roi"] = exits["roi"].astype(float)

# Sharpe, MDD 함수
def sharpe_ratio(returns, risk_free=0.0):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free) / returns.std()

def max_drawdown(cum_pnl):
    roll_max = cum_pnl.cummax()
    drawdown = (cum_pnl - roll_max)
    return drawdown.min()

# 모드별 성과 요약
rows = []
for mode, g in exits.groupby("mode"):
    g = g.sort_values("ts")
    cum_pnl = g["pnl_usdt"].cumsum()
    row = {
        "mode": mode,
        "n_trades": len(g),
        "win_rate": (g["pnl_usdt"] > 0).mean() * 100,
        "avg_pnl": g["pnl_usdt"].mean(),
        "total_pnl": g["pnl_usdt"].sum(),
        "sharpe": sharpe_ratio(g["pnl_usdt"]),
        "mdd": max_drawdown(cum_pnl),
    }
    rows.append(row)

summary = pd.DataFrame(rows)
print("=== 성과 요약 ===")
print(summary)

# 그래프 1: 누적 PnL
plt.figure(figsize=(10,6))
for mode, g in exits.groupby("mode"):
    g = g.sort_values("ts")
    plt.plot(g["ts"], g["pnl_usdt"].cumsum(), label=mode)
plt.legend()
plt.title("Cumulative PnL by Mode")
plt.xlabel("Trade index")
plt.ylabel("PnL (USDT)")
plt.show()

# 그래프 2: 누적 승률
plt.figure(figsize=(10,6))
for mode, g in exits.groupby("mode"):
    g = g.sort_values("ts")
    win = (g["pnl_usdt"] > 0).astype(int)
    plt.plot(g["ts"], win.expanding().mean()*100, label=mode)
plt.legend()
plt.title("Cumulative Win Rate by Mode")
plt.xlabel("Trade index")
plt.ylabel("Win Rate (%)")
plt.show()

# 그래프 3: Drawdown
plt.figure(figsize=(10,6))
for mode, g in exits.groupby("mode"):
    g = g.sort_values("ts")
    cum_pnl = g["pnl_usdt"].cumsum()
    dd = cum_pnl - cum_pnl.cummax()
    plt.plot(g["ts"], dd, label=mode)
plt.legend()
plt.title("Drawdown by Mode")
plt.xlabel("Trade index")
plt.ylabel("Drawdown (USDT)")
plt.show()
