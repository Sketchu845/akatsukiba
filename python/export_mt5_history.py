"""
Выгрузка истории свечей из MT5 в CSV.
Требования: Windows, запущенный терминал MT5, pip install MetaTrader5 pandas
"""

import MetaTrader5 as mt5
import pandas as pd

# ==== НАСТРОЙКИ ====
SYMBOL     = "XAUUSDc"          # символ ровно как у брокера (проверь суффикс!)
TIMEFRAME  = mt5.TIMEFRAME_M15  # M5 / M15 / H1 и т.д.
OUT_FILE   = f"{SYMBOL}_M15.csv"
# ===================

if not mt5.initialize():
    raise SystemExit(f"MT5 не запустился: {mt5.last_error()}")

info = mt5.symbol_info(SYMBOL)
if info is None:
    mt5.shutdown()
    raise SystemExit(f"Символ {SYMBOL} не найден. Доступные похожие: "
                     f"{[s.name for s in mt5.symbols_get('*XAU*')]}")

if not info.visible:
    mt5.symbol_select(SYMBOL, True)  # добавить в Market Watch, иначе данных не будет

N_BARS = 120_000  # сколько последних баров хотим (M15: ~120k = ~5 лет)

# Спрашиваем у терминала его лимит баров и не просим больше него
term = mt5.terminal_info()
if term is not None:
    print(f"Лимит терминала (maxbars): {term.maxbars}")
    N_BARS = min(N_BARS, max(term.maxbars - 1, 1000))

# Пробуем от большего к меньшему — берём максимум, что терминал готов отдать
rates = None
for n in [N_BARS, 60_000, 30_000, 10_000, 5_000]:
    if n > N_BARS:
        continue
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
    if rates is not None and len(rates) > 0:
        print(f"Успешно запрошено {n} баров, получено {len(rates)}")
        break
    print(f"Запрос {n} баров не прошёл: {mt5.last_error()}")

err = mt5.last_error()
mt5.shutdown()

if rates is None or len(rates) == 0:
    raise SystemExit(
        f"Данные не пришли. Последняя ошибка MT5: {err}\n"
        "Поставь в терминале 'Макс. баров в окне' = Unlimited и перезапусти его."
    )

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]

df.to_csv(OUT_FILE, index=False)
print(f"Сохранено {len(df)} свечей: {df['time'].iloc[0]} — {df['time'].iloc[-1]}")
print(f"Файл: {OUT_FILE}")
