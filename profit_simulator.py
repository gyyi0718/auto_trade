import pandas as pd
import matplotlib.pyplot as plt

# 설정값
initial_capital = 500  # 초기 자본
win_rate = 0.6          # 승률
num_trades_per_day = 10 # 하루 진입 횟수
profit_per_trade = 1.0  # 이긴 트레이드당 수익 ($)
loss_per_trade = 0.6    # 진 트레이드당 손실 ($)
days = 250              # 거래일 수 (1년 기준)

# 시뮬레이션 저장 리스트
balance_list = [initial_capital]
daily_profit_list = []

capital = initial_capital

for day in range(1, days + 1):
    win_trades = int(num_trades_per_day * win_rate)
    loss_trades = num_trades_per_day - win_trades

    daily_profit = (win_trades * profit_per_trade) - (loss_trades * loss_per_trade)
    capital += daily_profit

    balance_list.append(capital)
    daily_profit_list.append(daily_profit)

# 결과 요약
print("✅ 시뮬레이션 완료")
print(f"초기 자본: ${initial_capital:.2f}")
print(f"마지막 잔고: ${capital:.2f}")
print(f"총 수익: ${capital - initial_capital:.2f}")

# CSV 저장
result_df = pd.DataFrame({
    "Day": range(1, days + 1),
    "Balance": balance_list[1:],
    "DailyProfit": daily_profit_list
})
result_df.to_csv("compound_profit_log.csv", index=False)
print("📁 결과 저장됨: compound_profit_log.csv")

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(result_df["Day"], result_df["Balance"], label="Balance ($)", color="green")
plt.title("Compound Profit Simulation (1 Year)")
plt.xlabel("Day")
plt.ylabel("Balance ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compound_profit_chart.png")
print("📊 그래프 저장됨: compound_profit_chart.png")
plt.show()
