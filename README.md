# Akatsukiba — ML в MetaTrader 5 через ONNX

*Исследование: можно ли предсказать рынок машинным обучением на данных из
терминала — и как проверить это, не обманув себя.*

[English below](#english)

---

## О чём это

Полный пайплайн **Python (LightGBM) → ONNX → индикатор MQL5** с побитово
проверенным совпадением расчётов между Python и MQL5, и — главное — методика
честной валидации.

Шесть гипотез были проверены корректно (walk-forward с purge-зазором,
отложенный резерв, симуляция переобучения, учёт спреда). Все шесть дали
отрицательный результат. Он опубликован как есть.

**Это не торговый продукт и не обещание прибыли.** Это исследование и
инструментарий.

---

## Результаты исследования

Симуляция ежемесячного переобучения (XAUUSD M15, 10 месяцев):

```
ИТОГО: 800 сигналов, winrate 0.414
       порог безубыточности 0.400 (без спреда)
BUY : n=517  winrate=0.414 ±0.042
SELL: n=283  winrate=0.413 ±0.057
```

| # | Гипотеза | Результат |
|---|---|---|
| 1 | Контекст старшего ТФ (H1) | AUC 0.529 → 0.532, в пределах шума |
| 2 | Еженедельное переобучение | Хуже: 0.414 → 0.403 |
| 3 | Направленная разметка вместо triple barrier | AUC 0.512, хуже константного прогноза |
| 4 | Преимущество покупок над продажами | Опровергнуто: 0.414 ±0.042 vs 0.413 ±0.057 |
| 5 | Другие инструменты (6 символов, 2 ТФ) | Лучший AUC (EURUSD 0.60) → 0.377 на резерве |
| 6 | Межрыночные признаки доллара | Прирост AUC −0.003 (переобучение) |

### Два вывода, которые стоят внимания

**Высокий AUC ≠ прибыльность.** Инструменты с лучшим AUC показали худший
реальный winrate на отложенном резерве. AUC меряет ранжирование, а прибыль
решается порогом безубыточности, который задаёт спред.

**Важность признака ≠ его полезность.** Долларовые признаки заняли верх
рейтинга важности и при этом ухудшили модель — она на них переобучалась.

### Роль спреда

| Инструмент | ТФ | спред / ATR | безубыток |
|---|---|---|---|
| XAUUSD | M15 | 0.023 | 0.409 |
| XAUUSD | H1 | 0.011 | 0.405 |
| EURUSD | M15 | 0.150 | 0.460 |
| EURUSD | H1 | 0.072 | 0.429 |

Инструмент может оказаться неторгуемым на данном таймфрейме ещё до обучения
модели. Считать издержки нужно первым делом, а не последним.

---

## Что в этом репозитории

```
python/
  export_mt5_history.py   выгрузка истории свечей из MT5 в CSV
  prepare_dataset.py      21 признак (ATR-нормированные) + разметка triple barrier
  train_model.py          walk-forward валидация с purge-зазором
  check_spreads.py        спред/ATR и порог безубыточности по инструментам
mql5/
  DirectionProbability.mq5   индикатор: те же признаки вручную + ONNX + отрисовка
```

Этого достаточно, чтобы воспроизвести ключевую часть исследования: построить
признаки, разметить данные и честно измерить AUC на своих данных.

Полный набор — экспорт в ONNX с калибровкой, сверка паритета Python↔MQL5,
мультиинструментальный скан с резервом, симуляция переобучения, оценка
кандидата с учётом спреда, методичка — доступен как расширенный шаблон:
**[ссылка на страницу товара]**

---

## Типичные ошибки, которые здесь решены

| Ошибка | Что происходит | Решение |
|---|---|---|
| Случайное разбиение train/test | Модель учится на будущем | Только walk-forward |
| Нет зазора между train и test | Метки утекают в тест | Purge размером с горизонт |
| Признаки в абсолютных ценах | Модель запоминает уровни | Нормировка на ATR |
| Сырой выход принят за вероятность | «70%» ничего не значит | Изотоническая калибровка |
| Разные формулы в Python и MQL5 | Модель получает мусор вживую | Скрипт сверки паритета |
| Подгонка порога по лучшему | «Edge» в чистом шуме | Свип + проверка на резерве |

---

## Установка

```bash
pip install MetaTrader5 pandas numpy lightgbm scikit-learn
```

Требуется Windows и запущенный терминал MT5. В настройках терминала:
Сервис → Настройки → Графики → «Макс. баров в окне» = **Unlimited**, затем
перезапуск (иначе API отдаёт мало данных или падает с `Invalid params`).

```bash
cd python
py export_mt5_history.py
py prepare_dataset.py
py train_model.py
```

Шаг 3 — точка принятия решения: AUC около 0.50 означает, что сигнала нет и
дальше идти незачем.

---

## Статья

Подробный разбор исследования: **[ссылка на mql5.com]**

---

## Лицензия

MIT. Не является финансовой рекомендацией; торговля сопряжена с риском потерь.

---
---

<a name="english"></a>

# Akatsukiba — ML in MetaTrader 5 via ONNX

*A study: can you predict the market with machine learning on terminal-available
data — and how to verify it without fooling yourself.*

## What this is

A complete **Python (LightGBM) → ONNX → MQL5 indicator** pipeline with
bit-for-bit verified calculation parity between Python and MQL5, and — more
importantly — an honest validation methodology.

Six hypotheses were tested correctly (walk-forward with a purge gap, reserved
holdout, retraining simulation, spread accounting). All six produced a negative
result. It is published as-is.

**This is not a trading product and not a promise of profit.** It is research and
tooling.

## Study results

Monthly-retraining simulation (XAUUSD M15, 10 months):

```
TOTAL: 800 signals, win rate 0.414
       breakeven threshold 0.400 (excluding spread)
BUY : n=517  win rate=0.414 ±0.042
SELL: n=283  win rate=0.413 ±0.057
```

| # | Hypothesis | Result |
|---|---|---|
| 1 | Higher-timeframe context (H1) | AUC 0.529 → 0.532, within noise |
| 2 | Weekly retraining | Worse: 0.414 → 0.403 |
| 3 | Directional labels instead of triple barrier | AUC 0.512, worse than a constant forecast |
| 4 | Buy side outperforms sell side | Refuted: 0.414 ±0.042 vs 0.413 ±0.057 |
| 5 | Other instruments (6 symbols, 2 TFs) | Best AUC (EURUSD 0.60) → 0.377 on holdout |
| 6 | Cross-market dollar features | AUC gain −0.003 (overfitting) |

### Two takeaways worth your attention

**High AUC ≠ profitability.** The instruments with the best AUC produced the
worst real win rate on the reserved holdout. AUC measures ranking; profit is
decided by the breakeven threshold that spread sets.

**Feature importance ≠ feature usefulness.** Dollar features topped the
importance ranking while degrading the model — it was overfitting them.

## What's in this repository

```
python/
  export_mt5_history.py   export candle history from MT5 to CSV
  prepare_dataset.py      21 ATR-normalized features + triple-barrier labels
  train_model.py          walk-forward validation with a purge gap
  check_spreads.py        spread/ATR and breakeven threshold per instrument
mql5/
  DirectionProbability.mq5   indicator: same features by hand + ONNX + drawing
```

Enough to reproduce the core of the study: build features, label data, and
honestly measure AUC on your own data.

The full set — ONNX export with calibration, Python↔MQL5 parity check,
multi-instrument scan with holdout, retraining simulation, candidate evaluation
with spread, and the methodology guide — is available as an extended template:
**[product page link]**

## Setup

```bash
pip install MetaTrader5 pandas numpy lightgbm scikit-learn
```

Requires Windows and a running MT5 terminal. In the terminal: Tools → Options →
Charts → "Max bars in chart" = **Unlimited**, then restart.

```bash
cd python
py export_mt5_history.py
py prepare_dataset.py
py train_model.py
```

Step 3 is the decision point: AUC near 0.50 means there is no signal and no
reason to continue.

## License

MIT. Not financial advice; trading carries substantial risk of loss.
