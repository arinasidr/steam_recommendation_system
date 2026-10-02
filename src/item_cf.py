import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from common import load_data, evaluate

X, test_sets, maps = load_data()
results = []


def log_result(name, metrics):
    print(f"{name}:", metrics)
    results.append({"variant": name, **metrics})


def item_similarity(M):
    """Косинусное сходство игр: игры = столбцы матрицы (пользователь x игра)."""
    Mn = normalize(M, axis=0)                      # нормируем столбцы
    sim = (Mn.T @ Mn).toarray().astype("float32")  # сходство игр, 2992 x 2992
    np.fill_diagonal(sim, 0)                       # игра не похожа сама на себя
    return sim


def keep_top_k(sim, k):
    """Для каждой игры оставляем только k самых похожих игр, остальное обнуляем."""
    out = np.zeros_like(sim)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    out[rows, idx] = sim[rows, idx]
    return out


def get_scores(M, sim):
    """score игры = сумма сходств с играми, в которые пользователь уже играл."""
    return np.asarray(M @ sim, dtype="float32")


# сходство по оценке с учётом часов 
sim = item_similarity(X)
scores = get_scores(X, sim)
log_result("item-based CF (с учётом часов)", evaluate(scores, X, test_sets))

# улучшение 1: только факт взаимодействия, без часов
X_binary = X.copy().astype("float32")
X_binary.data[:] = 1.0

sim_binary = item_similarity(X_binary)
scores_binary = get_scores(X_binary, sim_binary)
log_result("item-based CF без учёта часов", evaluate(scores_binary, X, test_sets))

# улучшение 2: только k ближайших игр 
k_ndcg = {}
for k in [10, 20, 50, 100, 200]:
    scores_k = get_scores(X_binary, keep_top_k(sim_binary, k).T)
    metrics_k = evaluate(scores_k, X, test_sets)
    log_result(f"item-based CF без часов, {k} соседей", metrics_k)
    k_ndcg[k] = metrics_k["ndcg@k"]

best_k = max(k_ndcg, key=k_ndcg.get)
print("лучшее число соседей по NDCG:", best_k)

# улучшение 3: штраф за популярность 
# делим score на (число игроков игры) ** alpha, чтобы хиты не забивали выдачу
pop = np.asarray(X_binary.sum(axis=0)).ravel() + 1
base_scores = get_scores(X_binary, keep_top_k(sim_binary, best_k).T)

for alpha in [0.25, 0.5, 0.75, 1.0]:
    scores_a = (base_scores / (pop ** alpha)).astype("float32")
    log_result(
        f"item-based CF, {best_k} соседей, штраф за популярность alpha={alpha}",
        evaluate(scores_a, X, test_sets),
    )


os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/item_cf.csv", index=False)
print("результаты сохранены в results/item_cf.csv")