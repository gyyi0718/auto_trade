#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit 계정 완전 진단 도구
모든 마진 락 원인을 찾아냅니다.
"""

import os
import sys
import json
from pybit.unified_trading import HTTP

# API 설정
API_KEY = "Dlp4eJD6YFmO99T8vC"
API_SECRET = "YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U"
TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() in ("1", "true", "y", "yes")

if not API_KEY or not API_SECRET:
    print("❌ Error: BYBIT_API_KEY and BYBIT_API_SECRET must be set")
    sys.exit(1)

client = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    testnet=TESTNET,
    timeout=30
)

CATEGORIES = ["linear", "inverse", "option", "spot"]
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT",
    "BNBUSDT", "XRPUSDT", "ADAUSDT"
]


def print_section(title):
    """섹션 헤더 출력"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def check_wallet_balance():
    """지갑 잔고 상세 확인"""
    print_section("💰 WALLET BALANCE (Unified Account)")

    try:
        resp = client.get_wallet_balance(accountType="UNIFIED")
        result = resp.get("result", {})
        accounts = result.get("list", [])

        if not accounts:
            print("❌ No account data found")
            return None

        account = accounts[0]
        total_equity = float(account.get("totalEquity", 0))
        total_wallet_balance = float(account.get("totalWalletBalance", 0))
        total_margin_balance = float(account.get("totalMarginBalance", 0))
        total_available_balance = float(account.get("totalAvailableBalance", 0))
        total_perpetual_upnl = float(account.get("totalPerpUPL", 0))
        total_initial_margin = float(account.get("totalInitialMargin", 0))
        total_maintenance_margin = float(account.get("totalMaintenanceMargin", 0))

        print(f"\n📊 Account Summary:")
        print(f"   Total Equity:             ${total_equity:,.2f}")
        print(f"   Total Wallet Balance:     ${total_wallet_balance:,.2f}")
        print(f"   Total Margin Balance:     ${total_margin_balance:,.2f}")
        print(f"   Total Available Balance:  ${total_available_balance:,.2f}")
        print(f"   Total Perpetual uPnL:     ${total_perpetual_upnl:+,.2f}")
        print(f"   Total Initial Margin:     ${total_initial_margin:,.2f}")
        print(f"   Total Maintenance Margin: ${total_maintenance_margin:,.2f}")

        # 코인별 상세
        print(f"\n💎 Assets by Coin:")
        print("-" * 100)
        coins = account.get("coin", [])

        has_locked = False
        for coin_data in coins:
            coin = coin_data.get("coin", "")
            equity = float(coin_data.get("equity", 0))

            if equity <= 0:
                continue

            wallet_balance = float(coin_data.get("walletBalance", 0))
            available = float(coin_data.get("availableToWithdraw", 0))
            upnl = float(coin_data.get("unrealisedPnl", 0))
            locked = float(coin_data.get("locked", 0))
            borrow = float(coin_data.get("borrowAmount", 0))

            print(f"\n{coin}:")
            print(f"   Equity:               ${equity:,.8f}")
            print(f"   Wallet Balance:       ${wallet_balance:,.8f}")
            print(f"   Available to Withdraw: ${available:,.8f}")
            print(f"   Unrealised PnL:       ${upnl:+,.8f}")
            print(f"   Locked:               ${locked:,.8f} {'⚠️ LOCKED!' if locked > 0 else ''}")
            print(f"   Borrow Amount:        ${borrow:,.8f} {'⚠️ BORROWED!' if borrow > 0 else ''}")

            if locked > 0:
                has_locked = True

        print("-" * 100)

        if has_locked:
            print("\n⚠️  WARNING: Some funds are LOCKED!")
            print("   This could be due to:")
            print("   - Open orders in ANY category (linear, spot, option)")
            print("   - Pending withdrawals")
            print("   - Collateral for positions")
            print("   - Auto-repay settings")

        return {
            "total_equity": total_equity,
            "total_available": total_available_balance,
            "total_initial_margin": total_initial_margin,
            "coins": coins
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_positions_all_categories():
    """모든 카테고리의 포지션 확인"""
    print_section("📍 POSITIONS (All Categories)")

    has_positions = False

    for category in CATEGORIES:
        try:
            if category in ["linear", "inverse"]:
                resp = client.get_positions(category=category, settleCoin="USDT")
            else:
                # spot과 option은 다른 방식
                continue

            positions = (resp.get("result") or {}).get("list") or []
            active_positions = [p for p in positions if float(p.get("size", 0)) > 0]

            if active_positions:
                has_positions = True
                print(f"\n🔴 {category.upper()} ({len(active_positions)} positions):")
                print("-" * 100)

                for p in active_positions:
                    symbol = p.get("symbol")
                    side = p.get("side")
                    size = float(p.get("size", 0))
                    entry = float(p.get("avgPrice", 0))
                    mark = float(p.get("markPrice", 0))
                    leverage = p.get("leverage", "0")
                    upnl = float(p.get("unrealisedPnl", 0))
                    im = float(p.get("positionIM", 0))  # Initial Margin
                    mm = float(p.get("positionMM", 0))  # Maintenance Margin

                    print(f"\n   {symbol}:")
                    print(f"      Side: {side} | Size: {size} | Leverage: {leverage}x")
                    print(f"      Entry: ${entry:.4f} | Mark: ${mark:.4f}")
                    print(f"      uPnL: ${upnl:+.2f}")
                    print(f"      Initial Margin: ${im:.2f} ⚠️ LOCKED")
                    print(f"      Maintenance Margin: ${mm:.2f}")

        except Exception as e:
            print(f"   {category}: {e}")

    if not has_positions:
        print("\n✅ No active positions in any category")

    return has_positions


def check_open_orders_all_categories():
    """모든 카테고리의 미체결 주문 확인"""
    print_section("📋 OPEN ORDERS (All Categories)")

    total_orders = 0

    for category in CATEGORIES:
        try:
            # Linear와 Inverse
            if category in ["linear", "inverse"]:
                for symbol in SYMBOLS:
                    try:
                        resp = client.get_open_orders(category=category, symbol=symbol)
                        orders = (resp.get("result") or {}).get("list") or []

                        if orders:
                            total_orders += len(orders)
                            print(f"\n🔴 {category.upper()} - {symbol} ({len(orders)} orders):")
                            print("-" * 100)

                            for o in orders:
                                print(f"   Order ID: {o.get('orderId')}")
                                print(f"      Side: {o.get('side')} | Type: {o.get('orderType')}")
                                print(f"      Price: ${o.get('price')} | Qty: {o.get('qty')}")
                                print(f"      Status: {o.get('orderStatus')}")
                                print(f"      Created: {o.get('createdTime')}")
                                print()
                    except:
                        pass

            # Spot
            elif category == "spot":
                try:
                    resp = client.get_open_orders(category=category)
                    orders = (resp.get("result") or {}).get("list") or []

                    if orders:
                        total_orders += len(orders)
                        print(f"\n🔴 SPOT ({len(orders)} orders):")
                        print("-" * 100)

                        for o in orders:
                            print(f"   {o.get('symbol')}: {o.get('side')} {o.get('qty')} @ ${o.get('price')}")
                            print(f"      Order ID: {o.get('orderId')} | Status: {o.get('orderStatus')}")
                            print()
                except Exception as e:
                    print(f"   spot: {e}")

            # Option
            elif category == "option":
                try:
                    resp = client.get_open_orders(category=category)
                    orders = (resp.get("result") or {}).get("list") or []

                    if orders:
                        total_orders += len(orders)
                        print(f"\n🔴 OPTION ({len(orders)} orders):")
                        print("-" * 100)

                        for o in orders:
                            print(f"   {o.get('symbol')}: {o.get('side')} {o.get('qty')} @ ${o.get('price')}")
                            print(f"      Order ID: {o.get('orderId')} | Status: {o.get('orderStatus')}")
                            print()
                except Exception as e:
                    print(f"   option: {e}")

        except Exception as e:
            print(f"   {category}: Error - {e}")

    if total_orders == 0:
        print("\n✅ No open orders in any category")
    else:
        print(f"\n⚠️  TOTAL: {total_orders} open orders found!")

    return total_orders


def check_account_info():
    """계정 정보 확인"""
    print_section("👤 ACCOUNT INFO")

    try:
        # Account info
        resp = client.get_account_info()
        result = resp.get("result", {})

        print(f"\nAccount Type: {result.get('unifiedMarginStatus', 'N/A')}")
        print(f"Account Status: {result.get('status', 'N/A')}")

        # VIP 등급 및 수수료
        print(f"\nFee Rate:")
        fee_rate = result.get("feeRate", {})
        if fee_rate:
            print(f"   Spot: {fee_rate}")

    except Exception as e:
        print(f"❌ Error: {e}")


def check_transaction_log():
    """최근 거래 내역 확인"""
    print_section("📜 RECENT TRANSACTION LOG (Last 50)")

    try:
        resp = client.get_transaction_log(
            accountType="UNIFIED",
            category="linear",
            limit=50
        )

        logs = (resp.get("result") or {}).get("list") or []

        if not logs:
            print("\n✅ No recent transactions")
            return

        print(f"\nFound {len(logs)} recent transactions:")
        print("-" * 100)

        for log in logs[:20]:  # 최근 20개만 표시
            trans_type = log.get("type", "")
            coin = log.get("coin", "")
            amount = log.get("amount", "0")
            balance = log.get("balance", "0")
            timestamp = log.get("transactionTime", "")

            print(f"{timestamp} | {trans_type:20} | {coin:8} | {amount:>15} | Balance: {balance}")

        if len(logs) > 20:
            print(f"\n... and {len(logs) - 20} more transactions")

    except Exception as e:
        print(f"❌ Error: {e}")


def generate_report(wallet_info, has_positions, total_orders):
    """진단 리포트 생성"""
    print_section("🔍 DIAGNOSTIC REPORT")

    if not wallet_info:
        print("\n❌ Unable to generate report - wallet info unavailable")
        return

    total_equity = wallet_info["total_equity"]
    total_available = wallet_info["total_available"]
    total_im = wallet_info["total_initial_margin"]

    locked = total_equity - total_available - total_im

    print(f"\n📊 Summary:")
    print(f"   Total Equity:        ${total_equity:,.2f}")
    print(f"   Available Balance:   ${total_available:,.2f} ({total_available / total_equity * 100:.1f}%)")
    print(f"   Initial Margin:      ${total_im:,.2f}")
    print(f"   Other Locked:        ${locked:,.2f}")

    print(f"\n🔍 Analysis:")

    # 문제 진단
    problems = []

    if total_available <= 0:
        problems.append("❌ CRITICAL: Available Balance is $0 or negative!")
    elif total_available < total_equity * 0.1:
        problems.append("⚠️  WARNING: Less than 10% of equity is available")

    if has_positions:
        problems.append(f"📍 You have active positions consuming ${total_im:,.2f} as Initial Margin")

    if total_orders > 0:
        problems.append(f"📋 You have {total_orders} open orders that may be locking margin")

    if locked > 0:
        problems.append(f"🔒 ${locked:,.2f} is locked for unknown reasons (check Bybit web interface)")

    if problems:
        print("\n❗ Issues Found:")
        for i, problem in enumerate(problems, 1):
            print(f"   {i}. {problem}")
    else:
        print("\n✅ No obvious issues detected")

    print(f"\n💡 Recommendations:")

    if total_orders > 0:
        print("   1. Cancel all open orders:")
        print("      python cancel_open_orders.py --cancel")

    if has_positions:
        print("   2. Close positions you don't need (via Bybit web or API)")

    if total_available < total_equity * 0.2:
        print("   3. Check Bybit web interface for:")
        print("      - Pending withdrawals")
        print("      - Spot orders")
        print("      - Option positions")
        print("      - Lending/borrowing")

    print("   4. Contact Bybit support if issue persists:")
    print("      - Live chat: https://www.bybit.com")
    print("      - Save this diagnostic report")

    print(f"\n📄 Full diagnostic saved to: bybit_diagnostic_report.json")


def save_full_diagnostic():
    """전체 진단 정보를 JSON으로 저장"""
    try:
        diagnostic = {
            "wallet_balance": client.get_wallet_balance(accountType="UNIFIED").get("result"),
            "account_info": client.get_account_info().get("result"),
            "positions": {},
            "open_orders": {}
        }

        # 포지션
        for cat in ["linear", "inverse"]:
            try:
                diagnostic["positions"][cat] = client.get_positions(
                    category=cat, settleCoin="USDT"
                ).get("result")
            except:
                pass

        # 주문
        for cat in CATEGORIES:
            try:
                if cat in ["linear", "inverse"]:
                    orders = {}
                    for sym in SYMBOLS:
                        try:
                            orders[sym] = client.get_open_orders(
                                category=cat, symbol=sym
                            ).get("result")
                        except:
                            pass
                    diagnostic["open_orders"][cat] = orders
                else:
                    diagnostic["open_orders"][cat] = client.get_open_orders(
                        category=cat
                    ).get("result")
            except:
                pass

        with open("bybit_diagnostic_report.json", "w") as f:
            json.dump(diagnostic, f, indent=2)

        return True
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
        return False


def main():
    """메인 실행"""
    print("\n" + "=" * 100)
    print("  🔧 Bybit Account Diagnostic Tool")
    print("=" * 100)
    print(f"  Testnet: {TESTNET}")
    print(f"  API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print("=" * 100)

    # 1. 지갑 잔고
    wallet_info = check_wallet_balance()

    # 2. 계정 정보
    check_account_info()

    # 3. 포지션
    has_positions = check_positions_all_categories()

    # 4. 미체결 주문
    total_orders = check_open_orders_all_categories()

    # 5. 거래 내역
    check_transaction_log()

    # 6. 리포트
    generate_report(wallet_info, has_positions, total_orders)

    # 7. 전체 정보 저장
    save_full_diagnostic()

    print("\n" + "=" * 100)
    print("  Diagnostic Complete!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
9