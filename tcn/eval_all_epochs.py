# eval_all_epochs.py
# -*- coding: utf-8 -*-
import os, sys, re, csv, subprocess, hashlib
from collections import deque
import torch


def get_seq_len(ckpt):
    try:
        return torch.load(ckpt, map_location="cpu", weights_only=False).get("meta", {}).get("seq_len", 60)
    except:
        return 60


# ===== 고정 경로 =====
CKPT_DIR = "models_daily_v3"
DATA = "val_daily.csv"
OUT_SUMMARY = "./eval_daily_all_epochs_result.csv"

# ===== 실행 조건 =====
SEQ_LEN = 240
HORIZON = 1
TP_BPS = 350
SL_BPS = 175  # ⭐ Stop Loss 추가! (TP의 1.33배)
START_EQUITY = 10000
ENTRY_PCT = 0.05
LEVERAGE = 1
TAKER_FEE = 0.0010
MAKER_FEE = 0.0002
SLIP_IN_BPS = 2
SLIP_OUT_BPS = 3
MAX_OPEN = 5
MODE = "maker"
FLAT_EDGE_MIN = 0

# ===== 출력 파싱 =====
PAT1 = re.compile(r"\[FULL-VAL\].*?trades=(\d+).*?hit=([\d\.]+)%.*?sum_bps=([\-+\deE\.]+).*?x([\-+\deE\.]+)")
PAT2 = re.compile(r"trades=(\d+).*?hit=([\d\.]+)%.*?sum_bps=([\-+\deE\.]+).*?equity.*?x([\-+\deE\.]+)")
PAT3 = re.compile(r"trades=(\d+).*?hit=([\d\.]+).*?bps=([\-+\deE\.]+).*?\(x([\-+\deE\.]+)\)")

TIMEOUT_SEC = 1200
LAST_N_PRINT = 200


def md5sum(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(nbytes), b""):
            h.update(b)
    return h.hexdigest()


def parse_from_text(text: str):
    """여러 패턴으로 시도"""
    for pat in [PAT1, PAT2, PAT3]:
        ms = list(pat.finditer(text))
        if ms:
            m = ms[-1]
            trades, hit, sum_bps, eqx = m.groups()
            return dict(trades=int(trades), hit=float(hit), sum_bps=float(sum_bps), equity_x=float(eqx))

    lines = text.strip().split('\n')
    print(f"[DEBUG] 패턴 매칭 실패. 마지막 10줄:")
    for line in lines[-10:]:
        print(f"  {line}")
    return None


def build_cmd(ckpt: str, ep: int) -> list:
    seq_len = get_seq_len(ckpt)
    return [
        sys.executable, "./eval_tcn_daily_full.py",
        "--ckpt", ckpt,
        "--val", DATA,
        "--seq_len", str(seq_len),
        "--horizon", str(HORIZON),
        "--tp_bps", str(TP_BPS),
        "--sl_bps", str(SL_BPS),  # ⭐ SL 추가
        "--equity0", str(START_EQUITY),
        "--equity_pct", str(ENTRY_PCT),
        "--leverage", str(LEVERAGE),
        "--taker_fee", str(TAKER_FEE),
        "--maker_fee", str(MAKER_FEE),
        "--slip_in_bps", str(SLIP_IN_BPS),
        "--slip_out_bps", str(SLIP_OUT_BPS),
        "--max_open", str(MAX_OPEN),
        "--mode", MODE,
        "--out_csv", f"./_tmp_val_ep{ep:03d}.csv",
        "--flat_edge_min", str(FLAT_EDGE_MIN),
    ]


def run_epoch(ep: int):
    os.makedirs("./_logs", exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"daily_simple_ep{ep:03d}.ckpt")
    if not os.path.exists(ckpt_path):
        return None, f"ckpt 없음: {ckpt_path}"

    ckpt_md5 = md5sum(ckpt_path)[:8]
    cmd = build_cmd(ckpt_path, ep)
    print(f"\n===== EPOCH {ep} =====  (md5={ckpt_md5})")
    print("CMD:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True
    )
    tail = deque(maxlen=LAST_N_PRINT)
    lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            tail.append(line)
            lines.append(line)
            if "FULL-VAL" in line or "trades=" in line:
                print(f"  → {line}")
        ret = proc.wait(timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        ret = -1

    with open(f"./_logs/ep{ep:03d}.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    full_text = "\n".join(lines)
    rec = parse_from_text(full_text)

    if ret != 0:
        print(f"E{ep:03d} | ❌ 비정상 종료 코드={ret}")
        if rec is None:
            return None, f"비정상 종료 코드={ret}\n" + "\n".join(list(tail))

    if rec is None:
        return None, "결과 라인 미검출\n" + "\n".join(list(tail)[-20:])

    rec.update(epoch=ep, md5=ckpt_md5)
    return rec, None


def main():
    results = []
    header = ["epoch", "ckpt_md5", "trades", "hit(%)", "sum_bps", "equity_x"]

    ckpt_files = []
    for f in os.listdir(CKPT_DIR):
        if f.startswith("daily_simple_ep") and f.endswith(".ckpt"):
            match = re.search(r"ep(\d+)", f)
            if match:
                ckpt_files.append(int(match.group(1)))

    ckpt_files = sorted(ckpt_files)
    print(f"\n🎯 Stop Loss: {SL_BPS} bps (TP: {TP_BPS} bps)")
    print(f"📊 Risk:Reward = {SL_BPS/TP_BPS:.2f}:1")
    print(f"\n발견된 체크포인트: {len(ckpt_files)}개 (ep {min(ckpt_files)} ~ {max(ckpt_files)})")

    for ep in ckpt_files:
        rec, err = run_epoch(ep)
        if rec:
            print(f"E{ep:03d} | ✅ trades={rec['trades']} hit={rec['hit']:.2f}% bps={rec['sum_bps']:.1f} x{rec['equity_x']:.3f}")
            results.append([ep, rec["md5"], rec["trades"], rec["hit"], rec["sum_bps"], rec["equity_x"]])
        else:
            print(f"E{ep:03d} | ❌ {err.split(chr(10))[0]}")
            results.append([ep, None, None, None, None, None])

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(results)
    print(f"\n✅ Saved summary to {OUT_SUMMARY}")


if __name__ == "__main__":
    main()