import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import OUT, SEED


class TwoTower(nn.Module):
    def __init__(self, n_items, content_dim, dim=64, use_id=True):
        super().__init__()
        self.use_id = use_id
        self.use_content = content_dim > 0
        assert use_id or self.use_content, "нужен хотя бы ID или контент"

        # башня игры: свой обучаемый вектор у каждой игры + вектор из тегов/описания
        if use_id:
            self.item_id = nn.Embedding(n_items, dim)
            nn.init.normal_(self.item_id.weight, std=0.1)
        if self.use_content:
            self.item_mlp = nn.Sequential(
                nn.Linear(content_dim, 256), nn.ReLU(), nn.Linear(256, dim))

        # башня пользователя: доучивает среднее векторов его игр
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def item_vectors(self, content, cold_items=None):
        """Башня игры: (n_items, dim), каждая строка нормирована.
        cold_items: у этих игр ID-часть обнуляем (она не обучена), остаётся только контент."""
        v = 0
        if self.use_id:
            w = self.item_id.weight
            if cold_items is not None:
                w = w.clone()
                w[cold_items] = 0
            v = v + w
        if self.use_content:
            v = v + self.item_mlp(content)
        return F.normalize(v, dim=-1)

    def user_vectors(self, hist, item_vecs):
        """Башня пользователя: (B, dim), строки нормированы.
        hist: (B, n_items), строки поделены на число игр -> hist @ item_vecs = среднее векторов его игр."""
        h = hist @ item_vecs
        return F.normalize(h + self.user_mlp(h), dim=-1)


def train_two_tower(X, content=None, use_id=True, dim=64, temp=0.05, epochs=15,
                    batch_size=512, lr=1e-3, cold_items=None, verbose=True):
    """Возвращает (user_emb (n_users, dim), item_emb (n_items, dim)) как numpy."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    Xb = (X > 0).astype(np.float32)          # история без часов, просто играл / не играл
    n_users, n_items = Xb.shape
    H_all = torch.tensor(Xb.toarray())       # 10000 x 2992, около 120 МБ
    C = torch.tensor(content, dtype=torch.float32) if content is not None \
        else torch.zeros(n_items, 0)

    # холодных игр во время обучения «ещё нет» в каталоге
    not_in_catalog = torch.zeros(n_items, dtype=torch.bool)
    if cold_items is not None:
        not_in_catalog[torch.as_tensor(cold_items)] = True

    model = TwoTower(n_items, C.shape[1], dim, use_id)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    u_idx, i_idx = Xb.nonzero()              # обучающие пары (пользователь, игра)

    for epoch in range(epochs):
        t0 = time.time()
        perm = rng.permutation(len(u_idx))
        total = 0.0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            u = torch.as_tensor(u_idx[b], dtype=torch.long)
            i = torch.as_tensor(i_idx[b], dtype=torch.long)
            rows = torch.arange(len(b))

            hist = H_all[u].clone()
            hist[rows, i] = 0                # целевую игру убираем из истории, иначе утечка
            other_pos = hist > 0             # остальные игры пользователя не считаем негативами
            hist = hist / hist.sum(1, keepdim=True).clamp(min=1)

            I = model.item_vectors(C)
            U = model.user_vectors(hist, I)
            # полный softmax по всему каталогу: игр всего 3000, можно себе позволить
            logits = (U @ I.T) / temp
            logits = logits.masked_fill(other_pos | not_in_catalog, -1e9)
            loss = F.cross_entropy(logits, i)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if verbose:
            print(f"эпоха {epoch + 1}/{epochs}: loss {total / len(perm):.4f} ({time.time() - t0:.1f} c)")

    model.eval()
    with torch.no_grad():
        I = model.item_vectors(C, cold_items)
        hist = H_all / H_all.sum(1, keepdim=True).clamp(min=1)   # для выдачи берём всю историю
        U = model.user_vectors(hist, I)
    return U.numpy(), I.numpy()


if __name__ == "__main__":
    from common import load_data, evaluate
    X, test_sets, maps = load_data()
    content = np.load(f"{OUT}/item_content.npy")
    print("контент игр:", content.shape)
    results = []

    def run(name, verbose=False, **kw):
        t0 = time.time()
        U, I = train_two_tower(X, verbose=verbose, **kw)
        m = evaluate(U @ I.T, X, test_sets)
        m["train_sec"] = round(time.time() - t0, 1)
        print(f"{name}:", {k: round(float(v), 4) for k, v in m.items()})
        results.append({"variant": name, **m})
        return U, I

    # 1. что даёт каждая башня по отдельности
    run("two-tower, только ID", content=None, use_id=True)
    run("two-tower, только контент", content=content, use_id=False)
    U, I = run("two-tower, ID + контент", verbose=True, content=content, use_id=True)

    # 2. гиперпараметры для варианта ID + контент
    for dim in [32, 128]:
        run(f"two-tower, ID + контент, dim={dim}", content=content, dim=dim)
    for temp in [0.02, 0.1, 0.2]:
        run(f"two-tower, ID + контент, temp={temp}", content=content, temp=temp)
    run("two-tower, ID + контент, 30 эпох", content=content, epochs=30)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/two_tower.csv", index=False)
    # векторы варианта ID + контент нужны для tt_serving.py
    np.save(f"{OUT}/tt_user_emb.npy", U)
    np.save(f"{OUT}/tt_item_emb.npy", I)
    print("результаты сохранены в results/two_tower.csv")
