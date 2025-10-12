# -*- coding: utf-8 -*-
"""
bybit_vol_spike_scout_v2_patched.py
- "볼륨 스파이크 + 풀백" 조건만 스캔해서 페이퍼 트레이딩하는 단일 파일 (Drop-in)
- Testnet/메인넷 자동 선택, 심볼 선택/자동수집, 수수료 반영, CSV 로깅(트레이드/에쿼티/MTM)
- 안전장치: 스프레드/호가잔량/기대엣지/쿨다운, 트레일/BE 적용 버그 수정
"""
import os, time, math, json, requests, statistics as stats, certifi, csv
from collections import defaultdict

# ===== TLS (ssl 에러 방지)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
# ==== MODEL GUARD ====
MODEL_CKPT = os.getenv("MODEL_CKPT", r"D:\ygy_work\coin\multimodel\tcn_mt_spike.ckpt")
SEQ_LEN    = int(os.getenv("SEQ_LEN","240"))

# TP/비용은 변수 참조 대신 환경변수 기본값 직접 사용
_model_tp_default = os.getenv("TP_BPS", "12.0")
_taker_fee = float(os.getenv("TAKER_FEE", "0.0006"))          # 0.06% = 6 bps
_slip_bps  = float(os.getenv("SLIPPAGE_BPS_TAKER", "0.5"))    # 0.5 bps

MODEL_TP1_BPS  = float(os.getenv("MODEL_TP1_BPS", _model_tp_default))
MODEL_COST_BPS = float(os.getenv(
    "MODEL_COST_BPS",
    str((_taker_fee*1e4 + _slip_bps)*2)   # 진입+청산 비용(bps)
))
MODEL_VFLOOR_BPS = float(os.getenv("MODEL_VFLOOR_BPS","10"))
MODEL_TMIN       = int(os.getenv("MODEL_TMIN","1"))
MODEL_TMAX       = int(os.getenv("MODEL_TMAX","5"))
_model = None
_feat_cols = None

def _load_model():
    """tcn_mt_triplebarrier.py 로 학습한 ckpt 로드"""
    global _model, _feat_cols
    if _model is not None:
        return _model, _feat_cols
    import torch
    from ..multimodel.train_tcn_model_triplebarrier import TCN_MT
    ck = torch.load(MODEL_CKPT, map_location="cpu")
    _feat_cols = ck["feat_cols"]
    _model = TCN_MT(in_f=len(_feat_cols))
    _model.load_state_dict(ck["model"])
    _model.eval()
    return _model, _feat_cols

def _kl_to_df(symbol:str, kl:list):
    """bybit kline -> 모델 입력 DF"""
    import pandas as pd, time as _t
    if not kl:
        return None
    # kl: [{"o","h","l","c","vol","turn"}...] 최신까지 1m 가정
    ts0 = int(_t.time()) - 60*(len(kl)-1)
    return pd.DataFrame({
        "timestamp": pd.to_datetime([ts0 + 60*i for i in range(len(kl))], unit="s", utc=True),
        "symbol": symbol,
        "open":  [x["o"] for x in kl],
        "high":  [x["h"] for x in kl],
        "low":   [x["l"] for x in kl],
        "close": [x["c"] for x in kl],
        "volume":[x["vol"] for x in kl],
    })

def _model_decide(symbol:str):
    """모델 추론: (side, conf, timeout_min) | side ∈ {None,'Buy','Sell'}"""
    from ..multimodel.train_tcn_model_triplebarrier import infer_one_symbol
    m, feat_cols = _load_model()
    need = max(BASELINE_WIN+2, SEQ_LEN+5)
    kl2 = kline(symbol, need)
    df_sym = _kl_to_df(symbol, kl2)
    if df_sym is None or len(df_sym) < SEQ_LEN+1:
        return None, 0.0, None
    side_m, conf, tout = infer_one_symbol(
        m, df_sym, feat_cols, seq_len=SEQ_LEN,
        vfloor_bps=MODEL_VFLOOR_BPS, tp1_bps=MODEL_TP1_BPS,
        cost_bps=MODEL_COST_BPS, dyncut=None,
        target_bps_timeout=MODEL_TP1_BPS, tmin=MODEL_TMIN, tmax=MODEL_TMAX
    )
    return side_m, conf, tout

# ===== ENV helpers
def g(name, d):
    v = os.getenv(name)
    if v is None or str(v) == "": return d
    try:
        return type(d)(v)
    except Exception:
        return d

def glist(name, default_csv):
    raw = os.getenv(name, default_csv)
    return [s.strip() for s in raw.split(",") if s.strip()]

def bps(x): return x/10000.0

# ===== ENV
CATEGORY            = g("CATEGORY", "linear")
TESTNET             = g("BYBIT_TESTNET", 0) or g("TESTNET", 0)
INTERVAL            = g("INTERVAL","1")
BASELINE_WIN        = g("BASELINE_WIN",60)
SPIKE_RATIO         = g("SPIKE_RATIO",3.0)          # 최신 turnover / 과거 중앙값
SPIKE_Z             = g("SPIKE_Z",3.0)              # robust z(MAD)
PRICE_MOVE_BPS_MIN  = g("PRICE_MOVE_BPS_MIN",20.0)
SPREAD_BPS_MAX      = g("SPREAD_BPS_MAX",10.0)
MIN_BASE_TURN_USD   = g("MIN_BASE_TURN_USD",2_000_000.0)
MIN_OB_NOTIONAL_USD = g("MIN_OB_NOTIONAL_USD",150_000.0)  # 최상단 5레벨 bid+ask 합산 USD
REQUIRE_BOTH        = g("REQUIRE_BOTH",1)           # 1이면 (비율 AND z) 둘 다 만족
PULLBACK_FRAC       = g("PULLBACK_FRAC",0.30)       # 롱: close ≤ high - frac*(high-low), 숏 반대
EXPECTED_EDGE_MIN   = g("EXPECTED_EDGE_MIN_BPS",4.0)
COOLDOWN_SEC        = g("COOLDOWN_SEC",60)
START_EQUITY        = g("START_EQUITY",10_000.0)
LEVERAGE            = g("LEVERAGE",25.0)
ENTRY_EQUITY_PCT    = g("ENTRY_EQUITY_PCT",0.05)
TP_BPS              = g("TP_BPS",12.0)
SL_BPS              = g("SL_BPS",18.0)
TRAIL_BPS           = g("TRAIL_BPS",0.0)
TAKER_FEE           = g("TAKER_FEE",0.0006)
SLIPPAGE_BPS_TAKER  = g("SLIPPAGE_BPS_TAKER",0.5)
POLL_SEC            = g("POLL_SEC",10)
TRACE               = g("TRACE",1)
SYMBOLS             = glist("SYMBOLS", "")          # 비워두면 자동수집

# ===== CSV 경로 (volspike_*.csv 네이밍 유지)
TRADES_CSV       = os.getenv("TRADES_CSV", f"volspike_trades_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_CSV       = os.getenv("EQUITY_CSV", f"volspike_equity_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_MTM_CSV   = os.getenv("EQUITY_MTM_CSV", f"volspike_equity_mtm_{START_EQUITY}_{LEVERAGE}.csv")
TRADES_FIELDS    = ["ts","event","symbol","side","qty","entry","exit","tp","sl","ret_bps","pnl","fee","equity","reason"]
EQUITY_FIELDS    = ["ts","equity"]

# ===== API
BASE = "https://api-testnet.bybit.com/v5" if TESTNET else "https://api.bybit.com/v5"
SES  = requests.Session()
SES.verify = certifi.where()
HEADERS = {"User-Agent": "vol-spike-scout/2.0", "Accept": "application/json"}

BE_BPS              = g("BE_BPS", 0)        # MFE가 BE_BPS 이상이면 손절=BE로 올림(0이면 비활성)
MAX_HOLD_SEC        = g("MAX_HOLD_SEC", 0)  # 포지션 최대 보유 시간(0이면 비활성)
ONLY_USDT           = g("ONLY_USDT", 1)
MAX_SYMS            = g("MAX_SYMS", 0)      # 0=전체
SHUFFLE             = g("SHUFFLE", 1)
SLEEP_EACH_MS       = g("SLEEP_EACH_MS", 0) # 심볼 사이 sleep(ms)


def _get(path, params, timeout=10):
    url = BASE + path
    for i in range(3):
        try:
            r = SES.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429,500,502,503,504):
                time.sleep(0.4*(i+1)); continue
            return r.json()
        except Exception:
            time.sleep(0.4*(i+1))
    return {}

# 교체용: 심볼 로딩/표시 (bybit_vol_spike_scout_v2.py)
# ==== (파일 상단 ENV 근처에 추가) ====
SYMBOLS_ENV = [s.strip() for s in os.getenv("SYMBOLS","").split(",") if s.strip()]
ONLY_USDT   = int(os.getenv("ONLY_USDT","1"))     # 1=USDT 선물만
MAX_SYMS    = int(os.getenv("MAX_SYMS","0"))      # 0=전체
SHUFFLE     = int(os.getenv("SHUFFLE","1"))       # 1=매 루프 셔플
SLEEP_EACH_MS = int(os.getenv("SLEEP_EACH_MS","0"))  # 심볼당 대기(ms) - 레이트리밋용
# ==== get_symbols() 전체 교체 ====
def get_symbols():
    params = {"category": "linear"}
    r = SES.get(f"{BASE}/market/instruments-info", params=params, timeout=10).json()
    raw = r.get("result", {}).get("list", []) or []
    out = []
    for it in raw:
        if it.get("status") != "Trading":
            continue
        # 만기·분기물 등 제거, USDT만
        if "-" in it.get("symbol",""):
            continue
        if ONLY_USDT and not it.get("symbol","").endswith("USDT"):
            continue
        # 선물(Perpetual)만
        ctype = (it.get("contractType") or "").lower()
        if "perpetual" not in ctype:
            continue
        out.append(it["symbol"])
    # 중복 제거
    out = list(dict.fromkeys(out))
    if SHUFFLE: import random; random.shuffle(out)
    if MAX_SYMS and MAX_SYMS > 0:
        out = out[:int(MAX_SYMS)]
    print(f"[SYMS] loaded {len(out)} symbols (USDT={ONLY_USDT}, cap={'all' if MAX_SYMS in (0,'0',None) else MAX_SYMS}): {', '.join(out[:20])} ...")
    return out

def orderbook_info(symbol, levels=5):
    r = SES.get(f"{BASE}/market/orderbook",
                params={"category":"linear","symbol":symbol,"limit":levels},
                timeout=10).json()
    bids = r.get("result", {}).get("b", []) or []
    asks = r.get("result", {}).get("a", []) or []
    if not bids or not asks:
        return None  # BBA 없음
    bid = float(bids[0][0]); ask = float(asks[0][0]); mid = (bid+ask)/2.0 if (bid>0 and ask>0) else 0.0
    sp_bps = (ask - bid) / max(mid, 1e-12) * 1e4 if mid > 0 else float("inf")
    # BUGFIX: 과거엔 (bids+asks)[:levels] 로 '한쪽만' 더했던 문제 수정 → 양쪽 합산
    def _sum_notional(side):
        s = 0.0
        for px, qty, *_ in side[:levels]:
            s += float(px) * float(qty)
        return s
    ob_notional = _sum_notional(bids) + _sum_notional(asks)
    return {"bid": bid, "ask": ask, "mid": mid, "spread_bps": sp_bps, "ob_notional": ob_notional}


def kline(symbol, limit):
    r=_get("/market/kline", {"category": CATEGORY, "symbol": symbol, "interval": INTERVAL, "limit": min(int(limit),1000)})
    arr=(r.get("result") or {}).get("list") or []
    arr=sorted(arr,key=lambda z:int(z[0]))  # [start, o,h,l,c,volume,turnover]
    return [{"o":float(a[1]),"h":float(a[2]),"l":float(a[3]),"c":float(a[4]),"vol":float(a[5]),"turn":float(a[6])} for a in arr]

def orderbook_levels(symbol, levels=5):
    r=_get("/market/orderbook", {"category": CATEGORY, "symbol": symbol, "limit": max(1, int(levels))}, timeout=5)
    bids=(r.get("result") or {}).get("b") or []
    asks=(r.get("result") or {}).get("a") or []
    return bids, asks

def orderbook_notional(symbol, levels=5):
    bids, asks = orderbook_levels(symbol, levels)
    nb = sum(float(px)*float(qty) for px,qty, *_ in bids[:levels])
    na = sum(float(px)*float(qty) for px,qty, *_ in asks[:levels])
    return nb + na

def spread_bps(symbol):
    bids, asks = orderbook_levels(symbol, 1)
    if not bids or not asks: return 1e9, None
    bid=float(bids[0][0]); ask=float(asks[0][0]); mid=(bid+ask)/2.0
    return (ask-bid)/max(mid,1e-12)*1e4, (bid,ask,mid)

# ===== utils
def robust_stats(vals):
    med=stats.median(vals)
    mad=stats.median([abs(v-med) for v in vals]) or 1e-9
    z = 0.6745*(vals[-1]-med)/mad
    return med, z

def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)
    if not os.path.exists(EQUITY_MTM_CSV):
        with open(EQUITY_MTM_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)

def _w_trades(row: dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])

def _w_equity(eq: float, mtm=False):
    _ensure_csv()
    path = EQUITY_MTM_CSV if mtm else EQUITY_CSV
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{float(eq):.6f}"])

def log(msg):
    if TRACE: print(msg, flush=True)

# ===== State
EQUITY = float(START_EQUITY)
OPEN   = {}
COOLDOWN_UNTIL = defaultdict(float)

def equity_mtm():
    upnl = 0.0
    for sym, pos in OPEN.items():
        sp,ba = spread_bps(sym)
        if not ba: continue
        bid,ask,mid = ba
        mark = mid
        dirn = +1 if pos["side"]=="long" else -1
        upnl += (mark - pos["entry"]) * pos["qty"] * dirn
    return EQUITY + upnl

def log_trade(ev, **kw):
    base = f"[TRADE] {ev}"
    extra = " ".join([f"{k}={kw[k]}" for k in kw])
    print(f"{base} {extra}", flush=True)

# ==== 새 ENV (파일 상단 ENV 블록에 추가)
BE_BPS             = g("BE_BPS", 6.0)        # +BE_BPS 도달 시 손절을 BE로 이동
MAX_HOLD_SEC       = g("MAX_HOLD_SEC", 180)  # 타임아웃 청산(초)

# ==== 보조 함수 (파일 어딘가 utils 섹션에 추가)
def base_vol_bps(base):
    """기준 구간의 중앙 절대 로그수익(BPS)"""
    rets=[]
    for i in range(1,len(base)):
        c0=base[i-1]["c"]; c1=base[i]["c"]
        rets.append(abs(math.log(max(c1,1e-12)/max(c0,1e-12))))
    if not rets: return 10.0
    return 1e4*stats.median(rets)

def try_signal(symbol):
    now = time.time()
    if COOLDOWN_UNTIL[symbol] > now:
        log(f"[SKIP] {symbol} cooldown {int(COOLDOWN_UNTIL[symbol]-now)}s"); return

    kl = kline(symbol, BASELINE_WIN + 2)
    if len(kl) < BASELINE_WIN + 2:
        log(f"[SKIP] {symbol} kline {len(kl)}<{BASELINE_WIN+2}"); return

    prev, last = kl[-2], kl[-1]
    base = kl[-(BASELINE_WIN+1):-1]
    base_turn = [x["turn"] for x in base]

    med, z = robust_stats(base_turn + [last["turn"]])
    if med < MIN_BASE_TURN_USD:
        return

    ratio   = (last["turn"]/med) if med>0 else 0.0
    ret_bps = (last["c"]/prev["c"] - 1.0) * 1e4

    ob = orderbook_info(symbol, 5)
    if ob is None:
        log(f"[SKIP] {symbol} no BBA"); return
    sp_bps = ob["spread_bps"]; bid, ask, mid = ob["bid"], ob["ask"], ob["mid"]

    if sp_bps > SPREAD_BPS_MAX:
        log(f"[SKIP] {symbol} spread {sp_bps:.1f}bps > {SPREAD_BPS_MAX}"); return
    if ob["ob_notional"] < MIN_OB_NOTIONAL_USD:
        log(f"[SKIP] {symbol} ob_notional {ob['ob_notional']:,.0f}<{MIN_OB_NOTIONAL_USD:,}"); return

    cond_ratio = (ratio >= float(SPIKE_RATIO))
    cond_z     = (z     >= float(SPIKE_Z))
    cond_pm    = (abs(ret_bps) >= float(PRICE_MOVE_BPS_MIN))
    ok_spike   = (cond_ratio and cond_z) if REQUIRE_BOTH else (cond_ratio or cond_z)
    if not ok_spike or not cond_pm:
        log(f"[SKIP] {symbol} spike ratio={ratio:.2f} z={z:.2f} pm={ret_bps:.1f}bps"); return

    rng = max(1e-9, last["h"] - last["l"])
    near_top = (last["h"] - last["c"]) / rng
    near_bot = (last["c"] - last["l"]) / rng
    side_rule = None
    if ret_bps > 0 and near_top >= PULLBACK_FRAC: side_rule = "long"
    elif ret_bps < 0 and near_bot >= PULLBACK_FRAC: side_rule = "short"
    else:
        log(f"[SKIP] {symbol} no pullback (ret={ret_bps:.1f}bps, ntop={near_top:.2f}, nbot={near_bot:.2f})"); return

    fee_bps  = TAKER_FEE*1e4*2
    slip_bps = SLIPPAGE_BPS_TAKER*2
    exp_edge = abs(ret_bps) - sp_bps - fee_bps - slip_bps
    if exp_edge < EXPECTED_EDGE_MIN:
        log(f"[SKIP] {symbol} exp_edge {exp_edge:.1f}bps < need {EXPECTED_EDGE_MIN}bps"); return

    # ===== 모델 가드 =====
    try:
        side_m, conf, tout = _model_decide(symbol)
        if side_m is None:
            log(f"[SKIP][MODEL] {symbol} conf={conf:.2f}"); return
        agree = ((side_m=="Buy" and side_rule=="long") or (side_m=="Sell" and side_rule=="short"))
        if not agree:
            log(f"[SKIP][DISAGREE] {symbol} rule={side_rule} model={side_m} conf={conf:.2f}"); return
    except Exception as e:
        if TRACE: print(f"[MODEL_GUARD_ERR] {e}")
        return

    # ===== 진입 =====
    px = ask if side_rule=="long" else bid
    notional = EQUITY * ENTRY_EQUITY_PCT * LEVERAGE
    qty = max(notional / max(px,1e-12), 0.0)

    tp = px * (1 + bps(TP_BPS) * (1 if side_rule=="long" else -1))
    sl = px * (1 - bps(SL_BPS) * (1 if side_rule=="long" else -1))

    # per-position 타임아웃(모델 예측 분 → 초). tout 없으면 글로벌 MAX_HOLD_SEC 사용
    tmax_s = int(tout*60) if (isinstance(tout,(int,float)) and tout>0) else int(MAX_HOLD_SEC or 0)

    OPEN[symbol] = {
        "side":side_rule,"entry":px,"qty":qty,"tp":tp,"sl":sl,
        "mfe":0.0,"t0":now,"tmax_s":tmax_s
    }
    COOLDOWN_UNTIL[symbol] = now + COOLDOWN_SEC

    log_trade("OPEN", symbol=symbol, side=side_rule, qty=f"{qty:.6f}", entry=f"{px:.6f}",
              tp=f"{tp:.6f}", sl=f"{sl:.6f}", equity=f"{EQUITY:.2f}", notional=f"{notional:.2f}",
              spread_bps=f"{sp_bps:.1f}", ratio=f"{ratio:.2f}", z=f"{z:.2f}",
              pm_bps=f"{ret_bps:.1f}", exp_edge_bps=f"{exp_edge:.1f}", model=f"{side_m}", conf=f"{conf:.2f}", tout_min=f"{tout}")
    if SLEEP_EACH_MS: time.sleep(SLEEP_EACH_MS/1000.0)

def manage_positions():
    global EQUITY
    done = []
    now = time.time()
    for sym, pos in OPEN.items():
        ob = orderbook_info(sym, 1)
        if ob is None: continue
        bid, ask = ob["bid"], ob["ask"]
        px  = bid if pos["side"]=="long" else ask
        ret = (px/pos["entry"]-1.0) * (+1 if pos["side"]=="long" else -1)
        ret_bps = ret * 1e4
        pos["mfe"] = max(pos["mfe"], ret_bps)

        # BE 승격
        if BE_BPS and pos["mfe"] >= BE_BPS:
            be_px = pos["entry"]
            if pos["side"]=="long": pos["sl"] = max(pos["sl"], be_px)
            else:                   pos["sl"] = min(pos["sl"], be_px)

        # TP
        if (pos["side"]=="long" and px>=pos["tp"]) or (pos["side"]=="short" and px<=pos["tp"]):
            pnl = ret * pos["qty"] * pos["entry"]; EQUITY += pnl
            log_trade("CLOSE_TP", symbol=sym, px=f"{px:.6f}", ret_bps=f"{ret_bps:.1f}", pnl=f"{pnl:.2f}", equity=f"{EQUITY:.2f}")
            done.append(sym); continue

        # SL
        if (pos["side"]=="long" and px<=pos["sl"]) or (pos["side"]=="short" and px>=pos["sl"]):
            pnl = ret * pos["qty"] * pos["entry"]; EQUITY += pnl
            log_trade("CLOSE_SL", symbol=sym, px=f"{px:.6f}", ret_bps=f"{ret_bps:.1f}", pnl=f"{pnl:.2f}", equity=f"{EQUITY:.2f}")
            done.append(sym); continue

        # per-position 타임아웃 우선 -> 없으면 글로벌
        tmax_s = int(pos.get("tmax_s", 0))
        if tmax_s and (now - pos.get("t0", now)) >= tmax_s:
            pnl = ret * pos["qty"] * pos["entry"]; EQUITY += pnl
            log_trade("CLOSE_TIME", symbol=sym, px=f"{px:.6f}", ret_bps=f"{ret_bps:.1f}", pnl=f"{pnl:.2f}",
                      equity=f"{EQUITY:.2f}", hold_s=int(now-pos.get("t0",now)))
            done.append(sym); continue

        # 글로벌 백업
        if MAX_HOLD_SEC and not tmax_s and (now - pos.get("t0", now)) >= MAX_HOLD_SEC:
            pnl = ret * pos["qty"] * pos["entry"]; EQUITY += pnl
            log_trade("CLOSE_TIME", symbol=sym, px=f"{px:.6f}", ret_bps=f"{ret_bps:.1f}", pnl=f"{pnl:.2f}",
                      equity=f"{EQUITY:.2f}", hold_s=int(now-pos.get("t0",now)))
            done.append(sym); continue

    for s in done: OPEN.pop(s, None)

# ===== MAIN
def main():
    global EQUITY
    syms = SYMBOLS if SYMBOLS else get_symbols()
    print(f"[START] VOL-SPIKE scout v2 (TESTNET={int(bool(TESTNET))}) | syms={len(syms)} | AND={REQUIRE_BOTH} "
          f"| pullback≥{int(PULLBACK_FRAC*100)}% | ob≥${MIN_OB_NOTIONAL_USD:,.0f} | edge≥{EXPECTED_EDGE_MIN}bps "
          f"| TP={TP_BPS} SL={SL_BPS} | spread≤{SPREAD_BPS_MAX}bps | lev={LEVERAGE}x | entry%={ENTRY_EQUITY_PCT:.2f}")
    _w_equity(EQUITY, mtm=False)
    _w_equity(equity_mtm(), mtm=True)

    last_mtm = 0.0
    while True:
        try:
            # 스캔
            for s in syms:
                try:
                    try_signal(s)
                except Exception as e:
                    if TRACE: print(f"[WARN][{s}] {e}")
                    time.sleep(0.05)

            # 포지션 관리
            manage_positions()

            # MTM 로깅(1루프 1회)
            now_eq_mtm = equity_mtm()
            if abs(now_eq_mtm - last_mtm) > 1e-9:
                _w_equity(now_eq_mtm, mtm=True)
                last_mtm = now_eq_mtm

            # 에쿼티 스냅샷은 주기적으로
            _w_equity(EQUITY, mtm=False)

            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("[STOP] keyboard interrupt")
            break
        except Exception as e:
            print("[ERR]", e)
            time.sleep(2)

if __name__=="__main__":
    main()
