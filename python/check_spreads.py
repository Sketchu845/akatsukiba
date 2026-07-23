"""
Спред в единицах ATR и порог безубыточности по инструментам.
Считается по свежему куску истории (последние 5000 баров), иначе старые
эпохи с другой волатильностью искажают медиану.

Запуск:  py check_spreads.py     (терминал MT5 запущен)
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

SYMBOLS = ["XAUUSDc", "XAGUSDc", "EURUSDc", "GBPUSDc", "BTCUSDc", "USDJPYc", "AUDJPYc"]
TF = mt5.TIMEFRAME_H1     # поменяй на TIMEFRAME_M15 для сравнения
N = 5000                  # свежий кусок для честной медианы
TP_ATR, SL_ATR = 1.5, 1.0

mt5.initialize()
print(f"{'символ':>9} | {'спред, п':>8} | {'ATR':>10} | {'спред/ATR':>9} | {'безубыток':>9}")
print("-" * 60)
for s in SYMBOLS:
    info = mt5.symbol_info(s)
    if info is None:
        print(f"{s:>9} | не найден"); continue
    if not info.visible:
        mt5.symbol_select(s, True)
    r = mt5.copy_rates_from_pos(s, TF, 0, N)
    if r is None or len(r) < 1000:
        print(f"{s:>9} | мало данных"); continue
    d = pd.DataFrame(r)
    pc = d.close.shift(1)
    tr = pd.concat([d.high-d.low, (d.high-pc).abs(), (d.low-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    cost = float(np.nanmedian(d.spread.to_numpy()*info.point / atr.to_numpy()))
    be = (SL_ATR + cost) / (TP_ATR + SL_ATR)
    print(f"{s:>9} | {np.median(d.spread):>8.0f} | {np.nanmedian(atr):>10.5f} | "
          f"{cost:>9.4f} | {be:>9.3f}")
mt5.shutdown()

print("""
спред/ATR < 0.05  — работать можно
          0.05..0.15 — тяжело, нужен заметный edge
          > 0.20     — на этом таймфрейме бессмысленно (спред съедает всё)

ВАЖНО: сравнивать инструменты и таймфреймы можно только при одинаковом N,
взятом за один и тот же свежий период — иначе медианы несопоставимы.
""")
