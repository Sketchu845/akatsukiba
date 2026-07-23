"""
Шаг 3: обучение LightGBM и честная walk-forward проверка.
Никакого случайного разбиения — только "учимся на прошлом, проверяем на будущем",
с зазором (purge) в HORIZON баров между train и test против утечки меток.

Запуск:  py train_model.py
Вход:    dataset_labeled.csv
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# ==== НАСТРОЙКИ ====
DATA_FILE = "dataset_labeled.csv"
N_FOLDS   = 5
HORIZON   = 16     # тот же, что в prepare_dataset.py — размер purge-зазора
PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42,
)
# ===================

df = pd.read_csv(DATA_FILE, parse_dates=["time"])
FEATURES = [c for c in df.columns if c not in ("time", "label_buy", "label_sell")]
X = df[FEATURES].to_numpy(dtype=np.float32)
print(f"Строк: {len(df)}, фичей: {len(FEATURES)}")

def bucket_report(y_true, y_prob):
    """Фактический winrate по корзинам предсказанной вероятности."""
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 1.01]
    lines = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() > 0:
            lines.append(f"    p in [{lo:.1f},{hi:.1f}): n={m.sum():6d}  "
                         f"факт. winrate={y_true[m].mean():.3f}")
    return "\n".join(lines)

n = len(df)
fold_size = n // (N_FOLDS + 1)   # первая часть — стартовый train

for target in ("label_buy", "label_sell"):
    y = df[target].to_numpy()
    print(f"\n================ {target} ================")
    aucs = []
    all_true, all_prob = [], []

    for f in range(N_FOLDS):
        train_end = fold_size * (f + 1)
        test_start = train_end + HORIZON          # purge-зазор
        test_end = min(train_end + fold_size, n)
        if test_start >= test_end:
            break

        model = LGBMClassifier(**PARAMS)
        model.fit(X[:train_end], y[:train_end])
        prob = model.predict_proba(X[test_start:test_end])[:, 1]
        yt = y[test_start:test_end]

        auc = roc_auc_score(yt, prob)
        aucs.append(auc)
        all_true.append(yt)
        all_prob.append(prob)
        period = (df.time.iloc[test_start].date(), df.time.iloc[test_end - 1].date())
        print(f"  fold {f+1}: test {period[0]} — {period[1]}  AUC={auc:.4f}")

    all_true = np.concatenate(all_true)
    all_prob = np.concatenate(all_prob)
    print(f"  Средний AUC: {np.mean(aucs):.4f}")
    print(f"  Winrate по корзинам предсказаний (все фолды вместе):")
    print(bucket_report(all_true, all_prob))
    base = all_true.mean()
    print(f"    базовый winrate (просто везде входить): {base:.3f}")

# Важность фичей на полном датасете — чисто для ориентировки
model = LGBMClassifier(**PARAMS).fit(X, df.label_buy)
imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
print("\nТоп-10 фичей (label_buy):")
for name, v in imp[:10]:
    print(f"  {name:12s} {v}")
