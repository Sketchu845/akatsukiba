"""
Шаг 2 (v2): датасет с фичами M15 + контекст старшего таймфрейма H1.
- индикаторы считаем вручную (формулы повторим 1-в-1 в MQL5)
- H1-фичи берутся ТОЛЬКО с завершённых H1-баров (без подглядывания в будущее);
  в MQL5 это будет iClose(..., PERIOD_H1, 1) и т.д.
- разметка triple barrier: TP=1.5*ATR, SL=1.0*ATR, горизонт 16 баров

Запуск:  py prepare_dataset.py
Вход:    XAUUSDc_M15.csv
Выход:   dataset_labeled.csv
"""

import numpy as np
import pandas as pd

# ==== НАСТРОЙКИ ====
CSV_FILE   = "XAUUSDc_M15.csv"
OUT_FILE   = "dataset_labeled.csv"
ATR_PERIOD = 14
RSI_PERIOD = 14
EMA_PERIOD = 50
HORIZON    = 16    # баров вперёд (16 * M15 = 4 часа)
TP_ATR     = 1.5
SL_ATR     = 1.0
WARMUP     = 250   # больше, чем в v1: H1-индикаторам нужен свой прогрев
# ===================


def wilder(series, period):
    return series.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close, period):
    d = close.diff()
    g = wilder(d.clip(lower=0), period)
    l = wilder(-d.clip(upper=0), period)
    return 100 - 100 / (1 + g / l)


def atr(high, low, close, period):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return wilder(tr, period)


df = pd.read_csv(CSV_FILE, parse_dates=["time"])
print(f"Загружено {len(df)} свечей: {df.time.iloc[0]} — {df.time.iloc[-1]}")

# ---------- Индикаторы M15 ----------
df["atr"] = atr(df.high, df.low, df.close, ATR_PERIOD)
df["rsi"] = rsi(df.close, RSI_PERIOD)
df["ema"] = df.close.ewm(span=EMA_PERIOD, adjust=False).mean()

# ---------- Фичи M15 ----------
feat = pd.DataFrame(index=df.index)

for k in (1, 2, 3, 5, 8):
    feat[f"ret_{k}"] = (df.close - df.close.shift(k)) / df.atr

rng = (df.high - df.low).replace(0, np.nan)
feat["body"]       = (df.close - df.open) / df.atr
feat["upper_wick"] = (df.high - df[["open", "close"]].max(axis=1)) / df.atr
feat["lower_wick"] = (df[["open", "close"]].min(axis=1) - df.low) / df.atr
feat["close_pos"]  = (df.close - df.low) / rng

feat["dist_ema"] = (df.close - df.ema) / df.atr
feat["rsi"]      = df.rsi / 100.0
feat["atr_rel"]  = df.atr / df.atr.rolling(100).mean()

feat["dist_high20"] = (df.high.rolling(20).max() - df.close) / df.atr
feat["dist_low20"]  = (df.close - df.low.rolling(20).min()) / df.atr

hour = df.time.dt.hour + df.time.dt.minute / 60.0
feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
feat["dow"]      = df.time.dt.dayofweek / 4.0

# ---------- H1-контекст (только завершённые H1-бары) ----------
h1 = (df.set_index("time")
        .resample("1h")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
        .dropna())

h1["atr"] = atr(h1.high, h1.low, h1.close, ATR_PERIOD)
h1["ema"] = h1.close.ewm(span=EMA_PERIOD, adjust=False).mean()

h1f = pd.DataFrame(index=h1.index)
h1f["h1_dist_ema"]  = (h1.close - h1.ema) / h1.atr            # тренд H1
h1f["h1_ret_3"]     = (h1.close - h1.close.shift(3)) / h1.atr # импульс 3 часа
h1f["h1_rsi"]       = rsi(h1.close, RSI_PERIOD) / 100.0
r24_hi = h1.high.rolling(24).max()
r24_lo = h1.low.rolling(24).min()
h1f["h1_range_pos"] = (h1.close - r24_lo) / (r24_hi - r24_lo) # где мы в суточном диапазоне

# H1-бар с открытием T завершён в момент T+1h — только тогда его можно использовать
h1f = h1f.copy()
h1f["available_at"] = h1f.index + pd.Timedelta(hours=1)

m15_close_time = df.time + pd.Timedelta(minutes=15)
merged = pd.merge_asof(
    pd.DataFrame({"close_time": m15_close_time}),
    h1f.sort_values("available_at"),
    left_on="close_time", right_on="available_at",
    direction="backward",
)
for c in ("h1_dist_ema", "h1_ret_3", "h1_rsi", "h1_range_pos"):
    feat[c] = merged[c].to_numpy()

FEATURES = list(feat.columns)
print(f"Фичей: {len(FEATURES)}: {FEATURES}")

# ---------- Разметка triple barrier ----------
high, low, close, atr_v = (df[c].to_numpy() for c in ("high", "low", "close", "atr"))
n = len(df)
label_buy = np.zeros(n, dtype=np.int8)
label_sell = np.zeros(n, dtype=np.int8)

for i in range(n - HORIZON):
    a = atr_v[i]
    if not np.isfinite(a) or a <= 0:
        continue
    entry = close[i]
    tp_b, sl_b = entry + TP_ATR * a, entry - SL_ATR * a
    tp_s, sl_s = entry - TP_ATR * a, entry + SL_ATR * a
    done_b = done_s = False
    for j in range(i + 1, i + 1 + HORIZON):
        if not done_b:
            hit_tp, hit_sl = high[j] >= tp_b, low[j] <= sl_b
            if hit_tp or hit_sl:
                label_buy[i] = 1 if (hit_tp and not hit_sl) else 0
                done_b = True
        if not done_s:
            hit_tp, hit_sl = low[j] <= tp_s, high[j] >= sl_s
            if hit_tp or hit_sl:
                label_sell[i] = 1 if (hit_tp and not hit_sl) else 0
                done_s = True
        if done_b and done_s:
            break

df["label_buy"] = label_buy
df["label_sell"] = label_sell

# ---------- Сборка и чистка ----------
out = pd.concat([df[["time"]], feat, df[["label_buy", "label_sell"]]], axis=1)
out = out.iloc[WARMUP : n - HORIZON]
out = out.replace([np.inf, -np.inf], np.nan).dropna()

out.to_csv(OUT_FILE, index=False)
print(f"\nСохранено {len(out)} строк в {OUT_FILE}")
print(f"Баланс классов buy : {out.label_buy.mean():.3f}")
print(f"Баланс классов sell: {out.label_sell.mean():.3f}")
