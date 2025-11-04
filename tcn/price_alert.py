# price_alert.py
# -*- coding: utf-8 -*-
"""
실시간 가격 모니터링 + 알람 시스템
진입/청산 가격 근접 시 사운드 알람
"""
import os, sys, time, json, threading
from datetime import datetime
from typing import Dict, List
import requests
import pandas as pd
import winsound  # Windows 전용

# 크로스 플랫폼 사운드 대안
try:
    import playsound

    USE_PLAYSOUND = True
except:
    USE_PLAYSOUND = False


# ========= 설정 =========
class Config:
    # 알람 트리거 범위
    ENTRY_THRESHOLD = 0.005  # 진입가 ±0.5% 이내
    TP_THRESHOLD = 0.003  # TP ±0.3% 이내
    SL_THRESHOLD = 0.002  # SL ±0.2% 이내 (긴급!)

    # 체크 주기
    CHECK_INTERVAL = 5  # 5초마다 체크

    # 알람 쿨다운 (같은 알람 반복 방지)
    ALERT_COOLDOWN = 60  # 60초 동안 같은 알람 금지

    # 사운드 파일 (없으면 기본 beep)
    SOUND_ENTRY = "entry_alert.wav"
    SOUND_TP = "tp_alert.wav"
    SOUND_SL = "sl_alert.wav"

    # Binance API
    BINANCE_API = "https://api.binance.com/api/v3/ticker/price"


# ========= 사운드 재생 =========
def play_sound(sound_type="beep", frequency=1000, duration=500):
    """
    사운드 재생
    sound_type: "beep", "entry", "tp", "sl"
    """
    try:
        if sound_type == "entry" and os.path.exists(Config.SOUND_ENTRY):
            if USE_PLAYSOUND:
                playsound.playsound(Config.SOUND_ENTRY)
            else:
                winsound.PlaySound(Config.SOUND_ENTRY, winsound.SND_FILENAME)
        elif sound_type == "tp" and os.path.exists(Config.SOUND_TP):
            if USE_PLAYSOUND:
                playsound.playsound(Config.SOUND_TP)
            else:
                winsound.PlaySound(Config.SOUND_TP, winsound.SND_FILENAME)
        elif sound_type == "sl" and os.path.exists(Config.SOUND_SL):
            if USE_PLAYSOUND:
                playsound.playsound(Config.SOUND_SL)
            else:
                winsound.PlaySound(Config.SOUND_SL, winsound.SND_FILENAME)
        else:
            # 기본 beep
            if sound_type == "sl":
                # SL은 3번 울림 (긴급!)
                for _ in range(3):
                    winsound.Beep(2000, 300)
                    time.sleep(0.1)
            elif sound_type == "tp":
                # TP는 2번 울림
                for _ in range(2):
                    winsound.Beep(1500, 400)
                    time.sleep(0.1)
            else:
                # 진입은 1번
                winsound.Beep(frequency, duration)
    except Exception as e:
        print(f"[ERROR] 사운드 재생 실패: {e}")


# ========= 가격 조회 =========
def get_binance_price(symbol: str) -> float:
    """Binance에서 실시간 가격 조회"""
    try:
        # 심볼 형식 변환 (BTC → BTCUSDT)
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"

        response = requests.get(f"{Config.BINANCE_API}?symbol={symbol}", timeout=5)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        print(f"[ERROR] {symbol} 가격 조회 실패: {e}")
        return None


def get_multiple_prices(symbols: List[str]) -> Dict[str, float]:
    """여러 심볼 가격 한번에 조회"""
    try:
        response = requests.get(Config.BINANCE_API, timeout=10)
        data = response.json()

        prices = {}
        for item in data:
            symbol = item["symbol"]
            # USDT 페어만
            if symbol.endswith("USDT"):
                base = symbol[:-4]  # BTCUSDT → BTC
                prices[base] = float(item["price"])

        return prices
    except Exception as e:
        print(f"[ERROR] 가격 조회 실패: {e}")
        return {}


# ========= 알람 매니저 =========
class AlertManager:
    def __init__(self):
        self.alert_history = {}  # {(symbol, type): last_alert_time}
        self.lock = threading.Lock()

    def can_alert(self, symbol: str, alert_type: str) -> bool:
        """쿨다운 체크"""
        key = (symbol, alert_type)
        with self.lock:
            last_time = self.alert_history.get(key, 0)
            now = time.time()

            if now - last_time > Config.ALERT_COOLDOWN:
                self.alert_history[key] = now
                return True
            return False

    def trigger_alert(self, symbol: str, alert_type: str, message: str):
        """알람 발동"""
        if not self.can_alert(symbol, alert_type):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        # 콘솔 출력
        if alert_type == "SL":
            print(f"\n{'=' * 70}")
            print(f"🚨🚨🚨 [STOP LOSS] {symbol} - {timestamp} 🚨🚨🚨")
            print(f"{message}")
            print(f"{'=' * 70}\n")
        elif alert_type == "TP":
            print(f"\n{'=' * 70}")
            print(f"💰 [TAKE PROFIT] {symbol} - {timestamp} 💰")
            print(f"{message}")
            print(f"{'=' * 70}\n")
        else:  # ENTRY
            print(f"\n{'=' * 70}")
            print(f"📍 [ENTRY ZONE] {symbol} - {timestamp} 📍")
            print(f"{message}")
            print(f"{'=' * 70}\n")

        # 사운드 재생 (별도 스레드)
        sound_type = alert_type.lower()
        threading.Thread(target=play_sound, args=(sound_type,), daemon=True).start()


# ========= 포지션 추적 =========
class Position:
    def __init__(self, symbol: str, side: str, entry_target: float,
                 tp_price: float, sl_price: float, confidence: float = 0):
        self.symbol = symbol
        self.side = side
        self.entry_target = entry_target
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.confidence = confidence
        self.entered = False  # 진입 완료 여부
        self.alerted_entry = False
        self.alerted_tp = False
        self.alerted_sl = False


# ========= 모니터링 시스템 =========
class PriceMonitor:
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.alert_manager = AlertManager()
        self.running = False
        self.thread = None

    def add_position(self, pos: Position):
        """포지션 추가"""
        self.positions[pos.symbol] = pos
        print(f"[ADD] {pos.symbol} {pos.side} 모니터링 시작")
        print(f"      진입: ${pos.entry_target:,.2f} | TP: ${pos.tp_price:,.2f} | SL: ${pos.sl_price:,.2f}")

    def remove_position(self, symbol: str):
        """포지션 제거"""
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"[REMOVE] {symbol} 모니터링 종료")

    def mark_entered(self, symbol: str):
        """진입 완료 표시 (수동)"""
        if symbol in self.positions:
            self.positions[symbol].entered = True
            print(f"[ENTERED] {symbol} 진입 완료로 표시")

    def check_prices(self):
        """가격 체크 및 알람"""
        if not self.positions:
            return

        # 심볼 리스트
        symbols = list(self.positions.keys())

        # 가격 조회
        prices = get_multiple_prices(symbols)

        if not prices:
            return

        # 각 포지션 체크
        for symbol, pos in list(self.positions.items()):
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # 진입 전
            if not pos.entered:
                entry_diff = abs(current_price - pos.entry_target) / pos.entry_target

                if entry_diff <= Config.ENTRY_THRESHOLD and not pos.alerted_entry:
                    msg = (f"현재가: ${current_price:,.2f}\n"
                           f"목표가: ${pos.entry_target:,.2f}\n"
                           f"차이: {entry_diff * 100:.2f}%\n"
                           f"→ 진입 구간 도달! {pos.side} 주문 실행하세요!")

                    self.alert_manager.trigger_alert(symbol, "ENTRY", msg)
                    pos.alerted_entry = True

            # 진입 후
            else:
                # TP 체크
                if pos.side == "LONG":
                    tp_diff = (pos.tp_price - current_price) / pos.tp_price
                    if tp_diff <= Config.TP_THRESHOLD and tp_diff >= 0 and not pos.alerted_tp:
                        msg = (f"현재가: ${current_price:,.2f}\n"
                               f"TP: ${pos.tp_price:,.2f}\n"
                               f"차이: {tp_diff * 100:.2f}%\n"
                               f"→ TP 근접! 수익 실현 준비!")

                        self.alert_manager.trigger_alert(symbol, "TP", msg)
                        pos.alerted_tp = True

                else:  # SHORT
                    tp_diff = (current_price - pos.tp_price) / pos.tp_price
                    if tp_diff <= Config.TP_THRESHOLD and tp_diff >= 0 and not pos.alerted_tp:
                        msg = (f"현재가: ${current_price:,.2f}\n"
                               f"TP: ${pos.tp_price:,.2f}\n"
                               f"차이: {tp_diff * 100:.2f}%\n"
                               f"→ TP 근접! 수익 실현 준비!")

                        self.alert_manager.trigger_alert(symbol, "TP", msg)
                        pos.alerted_tp = True

                # SL 체크 (더 민감하게)
                if pos.side == "LONG":
                    sl_diff = (current_price - pos.sl_price) / pos.sl_price
                    if sl_diff <= Config.SL_THRESHOLD and sl_diff >= -0.01:  # SL 근처 또는 아래
                        msg = (f"현재가: ${current_price:,.2f}\n"
                               f"SL: ${pos.sl_price:,.2f}\n"
                               f"차이: {abs(sl_diff) * 100:.2f}%\n"
                               f"⚠️  손절 가격 근접! 즉시 확인 필요!")

                        self.alert_manager.trigger_alert(symbol, "SL", msg)
                        pos.alerted_sl = True

                else:  # SHORT
                    sl_diff = (pos.sl_price - current_price) / pos.sl_price
                    if sl_diff <= Config.SL_THRESHOLD and sl_diff >= -0.01:
                        msg = (f"현재가: ${current_price:,.2f}\n"
                               f"SL: ${pos.sl_price:,.2f}\n"
                               f"차이: {abs(sl_diff) * 100:.2f}%\n"
                               f"⚠️  손절 가격 근접! 즉시 확인 필요!")

                        self.alert_manager.trigger_alert(symbol, "SL", msg)
                        pos.alerted_sl = True

    def monitor_loop(self):
        """모니터링 루프"""
        print(f"\n{'=' * 70}")
        print("🎯 가격 모니터링 시작")
        print(f"체크 주기: {Config.CHECK_INTERVAL}초")
        print(f"진입 범위: ±{Config.ENTRY_THRESHOLD * 100:.1f}%")
        print(f"TP 범위: ±{Config.TP_THRESHOLD * 100:.1f}%")
        print(f"SL 범위: ±{Config.SL_THRESHOLD * 100:.1f}%")
        print(f"{'=' * 70}\n")

        while self.running:
            try:
                if self.positions:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] 체크 중... ({len(self.positions)}개 포지션)")
                    self.check_prices()

                time.sleep(Config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] 모니터링 오류: {e}")
                time.sleep(Config.CHECK_INTERVAL)

        print("\n[STOP] 모니터링 종료")

    def start(self):
        """모니터링 시작"""
        if self.running:
            print("[WARN] 이미 실행 중")
            return

        self.running = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """모니터링 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def status(self):
        """현재 상태 출력"""
        print(f"\n{'=' * 70}")
        print(f"📊 모니터링 상태 - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 70}")

        if not self.positions:
            print("추적 중인 포지션 없음")
            return

        # 가격 조회
        symbols = list(self.positions.keys())
        prices = get_multiple_prices(symbols)

        for symbol, pos in self.positions.items():
            status_text = "🔴 진입 대기" if not pos.entered else "🟢 진입 완료"
            print(f"\n{status_text} {symbol} {pos.side}")

            if symbol in prices:
                current = prices[symbol]
                entry_diff = ((current - pos.entry_target) / pos.entry_target) * 100

                print(f"  현재가: ${current:,.2f}")
                print(f"  진입목표: ${pos.entry_target:,.2f} ({entry_diff:+.2f}%)")
                print(f"  TP: ${pos.tp_price:,.2f}")
                print(f"  SL: ${pos.sl_price:,.2f}")

                if pos.entered:
                    if pos.side == "LONG":
                        pnl_pct = ((current - pos.entry_target) / pos.entry_target) * 100
                    else:
                        pnl_pct = ((pos.entry_target - current) / pos.entry_target) * 100

                    emoji = "💰" if pnl_pct > 0 else "📉"
                    print(f"  {emoji} 손익: {pnl_pct:+.2f}%")

        print(f"\n{'=' * 70}\n")


# ========= 메인 인터페이스 =========
def load_signals_from_csv(csv_path: str, monitor: PriceMonitor):
    """CSV에서 신호 로드 및 모니터링 추가"""
    df = pd.read_csv(csv_path)

    print(f"\n[LOAD] {len(df)}개 신호 로드")

    for _, row in df.iterrows():
        pos = Position(
            symbol=row["symbol"],
            side=row["side"],
            entry_target=float(row["entry_estimate"]),
            tp_price=float(row["tp_price"]),
            sl_price=float(row["sl_price"]),
            confidence=float(row.get("confidence", 0))
        )
        monitor.add_position(pos)


def interactive_mode(monitor: PriceMonitor):
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("📱 대화형 모드")
    print("=" * 70)
    print("\n명령어:")
    print("  add <SYMBOL> <SIDE> <ENTRY> <TP> <SL>  - 포지션 추가")
    print("  remove <SYMBOL>                        - 포지션 제거")
    print("  enter <SYMBOL>                         - 진입 완료 표시")
    print("  status                                 - 현재 상태")
    print("  load <CSV_PATH>                        - CSV 로드")
    print("  test <TYPE>                            - 알람 테스트")
    print("  quit                                   - 종료\n")

    while monitor.running:
        try:
            cmd = input(">>> ").strip().split()

            if not cmd:
                continue

            action = cmd[0].lower()

            if action == "quit":
                break

            elif action == "add" and len(cmd) >= 6:
                symbol = cmd[1].upper()
                side = cmd[2].upper()
                entry = float(cmd[3])
                tp = float(cmd[4])
                sl = float(cmd[5])

                pos = Position(symbol, side, entry, tp, sl)
                monitor.add_position(pos)

            elif action == "remove" and len(cmd) >= 2:
                monitor.remove_position(cmd[1].upper())

            elif action == "enter" and len(cmd) >= 2:
                monitor.mark_entered(cmd[1].upper())

            elif action == "status":
                monitor.status()

            elif action == "load" and len(cmd) >= 2:
                load_signals_from_csv(cmd[1], monitor)

            elif action == "test" and len(cmd) >= 2:
                test_type = cmd[1].lower()
                print(f"[TEST] {test_type} 알람 테스트...")
                play_sound(test_type)

            else:
                print("[ERROR] 잘못된 명령어")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")

    monitor.stop()


# ========= 메인 =========
def main():
    import argparse

    parser = argparse.ArgumentParser(description="실시간 가격 모니터링 + 알람")
    parser.add_argument("--signals", help="신호 CSV 파일 (자동 로드)")
    parser.add_argument("--interval", type=int, default=5, help="체크 주기 (초)")
    parser.add_argument("--test", action="store_true", help="알람 테스트")
    args = parser.parse_args()

    # 설정
    Config.CHECK_INTERVAL = args.interval

    # 알람 테스트
    if args.test:
        print("알람 테스트 중...")
        print("1. 진입 알람")
        play_sound("entry")
        time.sleep(2)

        print("2. TP 알람")
        play_sound("tp")
        time.sleep(2)

        print("3. SL 알람")
        play_sound("sl")

        print("테스트 완료!")
        return

    # 모니터 생성
    monitor = PriceMonitor()
    monitor.start()

    # CSV 자동 로드
    if args.signals and os.path.exists(args.signals):
        load_signals_from_csv(args.signals, monitor)

    # 대화형 모드
    try:
        interactive_mode(monitor)
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()