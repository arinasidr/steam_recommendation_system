import os
import time
import numpy as np
import pandas as pd
import faiss
from common import load_data, evaluate
from config import OUT, SEED, K

N_QUERIES = 1000
N_SYNTH = 200_000   # искусственный большой каталог, чтобы увидеть выигрыш индекса


def recommend_one(index, U, X, u):
    """Топ-K для пользователя u без игр из его истории.
    Кандидатов берём K + размер истории: у некоторых пользователей сотни игр (максимум 777)."""
    played = X.indices[X.indptr[u]:X.indptr[u + 1]]
    k = min(K + len(played), index.ntotal)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = max(64, k)   # efSearch должен быть не меньше k, иначе вернутся -1
    _, cand = index.search(U[u:u + 1], k)
    played = set(played.tolist())
    return [g for g in cand[0] if g >= 0 and g not in played][:K]


def make_hnsw(vectors):
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    index.add(vectors)
    return index, time.perf_counter() - t0


def avg_ms(fn, items):
    t0 = time.perf_counter()
    for x in items:
        fn(x)
    return (time.perf_counter() - t0) / len(items) * 1000


if __name__ == "__main__":
    X, test_sets, maps = load_data()
    U = np.ascontiguousarray(np.load(f"{OUT}/tt_user_emb.npy"), dtype="float32")
    I = np.ascontiguousarray(np.load(f"{OUT}/tt_item_emb.npy"), dtype="float32")
    d = I.shape[1]
    rng = np.random.default_rng(SEED)
    users = rng.choice(X.shape[0], N_QUERIES, replace=False)

    # 1. наш каталог, 2992 игры
    exact = faiss.IndexFlatIP(d)
    exact.add(I)
    hnsw, build_sec = make_hnsw(I)

    def brute(u):
        s = I @ U[u]
        np.argpartition(-s, K)[:K]

    rec_exact = {u: recommend_one(exact, U, X, u) for u in users}
    rec_hnsw = {u: recommend_one(hnsw, U, X, u) for u in users}
    recall = np.mean([len(set(rec_exact[u]) & set(rec_hnsw[u])) / K for u in users])

    rows = [
        {"catalog": len(I), "variant": "numpy, полный перебор", "ms_per_query": avg_ms(brute, users)},
        {"catalog": len(I), "variant": "FAISS IndexFlatIP (точный)",
         "ms_per_query": avg_ms(lambda u: recommend_one(exact, U, X, u), users)},
        {"catalog": len(I), "variant": "FAISS HNSW", "build_sec": build_sec, "recall_vs_exact": recall,
         "ms_per_query": avg_ms(lambda u: recommend_one(hnsw, U, X, u), users)},
    ]

    # 2. большой искусственный каталог: так выглядел бы магазин с сотнями тысяч товаров
    big = rng.normal(size=(N_SYNTH, d)).astype("float32")
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    q = U[users[:200]]
    big_exact = faiss.IndexFlatIP(d)
    big_exact.add(big)
    big_hnsw, big_build = make_hnsw(big)
    _, top_exact = big_exact.search(q, K)

    rows += [
        {"catalog": N_SYNTH, "variant": "numpy, полный перебор",
         "ms_per_query": avg_ms(lambda v: np.argpartition(-(big @ v), K)[:K], q)},
        {"catalog": N_SYNTH, "variant": "FAISS IndexFlatIP (точный)",
         "ms_per_query": avg_ms(lambda v: big_exact.search(v[None], K), q)},
    ]
    # компромисс скорость / точность: чем больше efSearch, тем точнее и медленнее
    for ef in [16, 64, 256, 1024]:
        big_hnsw.hnsw.efSearch = ef
        _, top_hnsw = big_hnsw.search(q, K)
        rec = np.mean([len(set(a) & set(b)) / K for a, b in zip(top_exact, top_hnsw)])
        rows.append({"catalog": N_SYNTH, "variant": f"FAISS HNSW, efSearch={ef}",
                     "build_sec": big_build, "recall_vs_exact": rec,
                     "ms_per_query": avg_ms(lambda v: big_hnsw.search(v[None], K), q)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # 3. качество выдачи через HNSW на обычных метриках (должно совпасть с точным поиском)
    scores = np.full(X.shape, -np.inf, dtype=np.float32)
    for u in range(X.shape[0]):
        rec = recommend_one(hnsw, U, X, u)
        scores[u, rec] = np.arange(len(rec), 0, -1)
    print("метрики выдачи через HNSW:", evaluate(scores, X, test_sets))

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/tt_serving.csv", index=False)
