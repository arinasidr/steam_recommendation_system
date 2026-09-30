import json
import numpy as np, pandas as pd
from scipy.sparse import load_npz
from config import OUT, K

def load_data():
    """Возвращает X_train (users x games, sparse), test_sets {user: {games}}, mappings."""
    X = load_npz(f"{OUT}/train.npz").tocsr()
    test = pd.read_parquet(f"{OUT}/test.parquet")
    test_sets = test.groupby("u")["i"].apply(set).to_dict()
    maps = json.load(open(f"{OUT}/mappings.json"))
    return X, test_sets, maps

def evaluate(scores, X_train, test_sets, k=K):
    """scores: np.ndarray (n_users x n_games), чем больше, тем лучше.
    Игры из train маскируются внутри функции."""
    scores = np.array(scores, dtype=np.float32)          # копия, оригинал не портим
    r, c = X_train.nonzero()
    scores[r, c] = -np.inf

    users = np.array(sorted(test_sets))
    S = scores[users]
    top = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.arange(len(users))[:, None]
    top = top[rows, np.argsort(-S[rows, top], axis=1)]   # сортируем топ-k по убыванию

    disc = 1 / np.log2(np.arange(2, k + 2))
    P = R = N = 0.0
    for row, u in enumerate(users):
        rel = test_sets[u]
        hits = np.array([g in rel for g in top[row]], dtype=float)
        P += hits.sum() / k
        R += hits.sum() / len(rel)
        N += (hits * disc).sum() / disc[:min(len(rel), k)].sum()
    n = len(users)
    return {"precision@k": P / n, "recall@k": R / n, "ndcg@k": N / n,
            "coverage": len(np.unique(top)) / scores.shape[1]}