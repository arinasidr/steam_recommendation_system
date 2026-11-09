import numpy as np

from common import load_data, evaluate
from config import SEED
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import normalize

from config import OUT, SEED, K
from two_tower import train_two_tower

def make_cold_split(X, test_sets, fraction=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    n_cold = int(np.ceil(X.shape[1] * fraction))

    cold_items = np.sort(
        rng.choice(X.shape[1], size=n_cold, replace=False)
    )

    # Удаляем взаимодействия с холодными играми у всех пользователей.
    X_warm = X.copy().tocsr()
    X_warm.data[np.isin(X_warm.indices, cold_items)] = 0
    X_warm.eliminate_zeros()

    cold_set = set(cold_items.tolist())
    has_history = np.diff(X_warm.indptr) > 0

    # Оставляем пользователей с холодными играми в test
    # и хотя бы одной игрой в оставшейся истории.
    cold_test = {}

    for user, relevant in test_sets.items():
        cold_relevant = relevant & cold_set

        if cold_relevant and has_history[user]:
            cold_test[user] = cold_relevant

    return X_warm, cold_test, cold_items

def evaluate_cold(scores, X_warm, cold_test, cold_items, k=K):
    users = np.array(sorted(cold_test))

    # Берём пользователей для оценки и исключаем известные им игры.
    selected = np.array(scores[users], dtype=np.float32, copy=True)
    rows, cols = X_warm[users].nonzero()
    selected[rows, cols] = -np.inf

    # Стабильная сортировка делает порядок при равных оценках повторяемым.
    top = np.argsort(
        -selected,
        axis=1,
        kind="stable",
    )[:, :k]

    recalls = [
        len(set(top[row]) & cold_test[user]) / len(cold_test[user])
        for row, user in enumerate(users)
    ]

    return {
        f"cold_recall@{k}": float(np.mean(recalls)),
        f"cold_share@{k}": float(np.isin(top, cold_items).mean()),
    }

if __name__ == "__main__":
    X, test_sets, mappings = load_data()
    X_warm, cold_test, cold_items = make_cold_split(X, test_sets)

    print("Холодных игр:", len(cold_items))
    print("Взаимодействий train до:", X.nnz)
    print("Взаимодействий train после:", X_warm.nnz)
    print("Пользователей для оценки:", len(cold_test))
    print(
        "Тестовых взаимодействий с холодными играми:",
        sum(len(items) for items in cold_test.values()),
    )

    assert X_warm[:, cold_items].nnz == 0
    assert len(cold_test) > 0
    print("Разбиение проверено: холодных игр в train нет.")
    result_dir = Path("results/cold_start")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем выбранные игры для воспроизводимости.
    np.savez(
        result_dir / "split.npz",
        cold_items=cold_items,
        cold_app_ids=np.asarray(mappings["app_ids"])[cold_items],
        seed=SEED,
    )

    content = np.load(f"{OUT}/tt_combined.npy")
    assert content.shape[0] == X_warm.shape[1]

    X_binary = (X_warm > 0).astype(np.float32)
    results = []

    def record(name, scores):
        metrics = evaluate_cold(
            scores, X_warm, cold_test, cold_items
        )
        # Общая оценка по исходному test для всех пользователей,
        # у которых после удаления холодных игр осталась история.
        overall_test = {
            user: relevant
            for user, relevant in test_sets.items()
            if X_warm.indptr[user + 1] > X_warm.indptr[user]
        }

        overall = evaluate(scores, X_warm, overall_test)

        metrics[f"overall_recall@{K}"] = float(overall["recall@k"])
        metrics[f"overall_ndcg@{K}"] = float(overall["ndcg@k"])
        results.append({"variant": name, **metrics})

        print(name, metrics, flush=True)
        pd.DataFrame(results).to_csv(
            result_dir / "metrics.csv",
            index=False,
        )

    # 1. Content-based: профиль по метаданным известных игр.
    profiles = normalize(X_binary @ content, axis=1)
    record(
        "content-based: теги + описания",
        profiles @ content.T,
    )

    # 2. Item-based CF: сходство только по оставшимся взаимодействиям.
    normalized_items = normalize(X_binary, axis=0)
    similarity = (
        normalized_items.T @ normalized_items
    ).toarray().astype(np.float32)

    np.fill_diagonal(similarity, 0)

    record(
        "item-based CF: без часов",
        np.asarray(X_binary @ similarity, dtype=np.float32),
    )

    # 3. Two-Tower: холодные игры исключены из обучения.
    user_emb, item_emb = train_two_tower(
        X_warm,
        content=content,
        use_id=True,
        dim=64,
        temp=0.05,
        epochs=15,
        cold_items=cold_items,
        verbose=True,
    )

    record("two-tower: ID + контент", user_emb @ item_emb.T)

    print("\nХолодный старт:")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print(f"Результаты: {result_dir / 'metrics.csv'}")