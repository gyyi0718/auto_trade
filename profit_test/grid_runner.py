import os, sys, time


def save_equity(equity: float, path="last_equity.txt"):
    with open(path, "w") as f:
        f.write(str(equity))


def load_equity(path="last_equity.txt", default=1000.0):
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except FileNotFoundError:
        return default

START_EQUITY = load_equity()
while True:
    # 직전 equity 불러오기
    START_EQUITY = load_equity()

    # trading 루프
    equity = run_trading(START_EQUITY)

    # 종료 시 청산 후 잔고 저장
    save_equity(equity)

    # 1분 후 코드 재실행
    time.sleep(60)
    os.execv(sys.executable, [sys.executable] + sys.argv)
