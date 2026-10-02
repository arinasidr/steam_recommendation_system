import os
import numpy as np
import pandas as pd
from common import load_data, evaluate

X, test_sets, maps = load_data()
n_users, n_games = X.shape
results = []


def log_result(name, metrics):
    print(f"{name}:", metrics)
    results.append({"variant": name, **metrics})


def to_scores(item_scores):
    """Одинаковая оценка игр для всех пользователей: (n_users x n_games)."""
    return np.tile(np.asarray(item_scores, dtype="float32"), (n_users, 1))


players = np.asarray((X > 0).sum(axis=0)).ravel()   # сколько игроков у игры
total = np.asarray(X.sum(axis=0)).ravel()           # суммарная вовлечённость (с часами)
mean = total / np.maximum(players, 1)               # средняя вовлечённость на игрока

# по числу игроков 
log_result("популярное: число игроков",
           evaluate(to_scores(players), X, test_sets))

# улучшение 1: по суммарному времени в игре
log_result("популярное: суммарное время",
           evaluate(to_scores(total), X, test_sets))

# улучшение 2: по среднему времени с порогом по числу игроков 
for min_players in [10, 50, 100]:
    s = mean.copy()
    s[players < min_players] = 0                    # редкие игры выкидываем
    log_result(f"популярное: среднее время, минимум {min_players} игроков",
               evaluate(to_scores(s), X, test_sets))

# улучшение 3: компромисс, и популярность, и вовлечённость
log_result("популярное: log(игроки) * среднее время",
           evaluate(to_scores(np.log1p(players) * mean), X, test_sets))

# топ-10
games = pd.read_parquet("data/processed/games.parquet")
top = np.argsort(-players)[:10]
print("\nтоп-10 по числу игроков:")
print(games.iloc[top][["app_id", "title"]].assign(players=players[top]).to_string(index=False))


os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/popular.csv", index=False)
print("результаты сохранены в results/popular.csv")