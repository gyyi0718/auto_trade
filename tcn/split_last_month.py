# split_last_month.py
# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def load_df(path:str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    ts_col = "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if ts_col is None:
        raise ValueError("date 또는 timestamp 컬럼이 필요")
    if not is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    else:
        if is_datetime64tz_dtype(df[ts_col]): df[ts_col] = df[ts_col].dt.tz_convert("UTC")
        else: df[ts_col] = df[ts_col].dt.tz_localize("UTC")
    need = {"symbol","open","high","low","close","volume", ts_col}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"누락 컬럼: {miss}")
    df = df.dropna(subset=[ts_col,"symbol"]).sort_values(["symbol", ts_col]).reset_index(drop=True)
    df = df.rename(columns={ts_col:"date"})
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq_len", type=int, required=True, help="warmup 확보 길이")
    ap.add_argument("--days", type=int, default=None, help="최근 N일을 검증으로")
    ap.add_argument("--cut", type=str, default=None, help="검증 시작 UTC 날짜 예: 2025-09-01")
    ap.add_argument("--span-days", type=int, default=None, help="검증 길이 일수(예: 30). 지정 시 cut~cut+span 구간만 검증")
    ap.add_argument("--no-warmup", type=int, default=0, help="1이면 warmup 미포함 출력")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True, help="warmup 포함/제외 선택 저장")
    args = ap.parse_args()

    df = load_df(args.data)

    if args.cut:
        cut_ts = pd.to_datetime(args.cut, utc=True)
    else:
        if args.days is None:
            raise ValueError("--cut 또는 --days 중 하나는 필요")
        cut_ts = df["date"].max() - pd.Timedelta(days=args.days)

    # 검증 종료시점
    end_ts = None
    if args.span_days is not None:
        end_ts = cut_ts + pd.Timedelta(days=args.span_days)

    dfs_tr, dfs_va = [], []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date")  # 인덱스 라벨 유지
        idx_cut = g.index[g["date"] >= cut_ts]
        if len(idx_cut)==0:
            dfs_tr.append(g)  # 전구간 train
            continue

        first_val_idx = int(idx_cut.min())  # cut 이상의 첫 인덱스
        warmup_start_idx = max(int(g.index.min()), first_val_idx - args.seq_len)

        # 검증 종료 인덱스 결정
        if end_ts is not None:
            idx_end = g.index[g["date"] < end_ts]
            if len(idx_end)==0:
                # cut 이후 데이터가 없으면 전부 train
                dfs_tr.append(g)
                continue
            last_val_idx = int(idx_end.max())
        else:
            last_val_idx = int(g.index.max())

        # 분리 규칙:
        # no_warmup=0:  train < warmup_start   | val = warmup_start ~ last_val
        # no_warmup=1:  train < first_val      | val = first_val ~ last_val
        if args.no_warmup:
            tr = g.loc[g.index < first_val_idx]
            va = g.loc[first_val_idx:last_val_idx]
        else:
            tr = g.loc[g.index < warmup_start_idx]
            va = g.loc[warmup_start_idx:last_val_idx]

        if len(tr): dfs_tr.append(tr)
        if len(va): dfs_va.append(va)

    train = pd.concat(dfs_tr, ignore_index=True) if dfs_tr else df.iloc[0:0].copy()
    val   = pd.concat(dfs_va,   ignore_index=True) if dfs_va else df.iloc[0:0].copy()

    (train.to_parquet(args.out_train) if args.out_train.lower().endswith(".parquet") else train.to_csv(args.out_train, index=False))
    (val.to_parquet(args.out_val)     if args.out_val.lower().endswith(".parquet")   else val.to_csv(args.out_val, index=False))
    span_msg = f"~{(cut_ts + pd.Timedelta(days=args.span_days)).strftime('%Y-%m-%d')}" if args.span_days else "~END"
    print(f"[SPLIT] cut={cut_ts.strftime('%Y-%m-%d')}{span_msg}  train={len(train):,}  val={len(val):,}  no_warmup={args.no_warmup}")

if __name__ == "__main__":
    main()
    #python split_last_month.py  --data bybit_futures_daily.csv  --seq_len 240 --days 365 --out-train train_daily.csv --out-val val_daily.csv