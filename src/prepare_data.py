import json, os
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, save_npz
from config import *

# 1. читаем большой файл кусками, оставляем только положительные отзывы
chunks = []
for ch in pd.read_csv(f"{RAW}/recommendations.csv",
                      usecols=["app_id", "user_id", "is_recommended", "hours"],
                      dtype={"app_id": "int32", "user_id": "int32", "hours": "float32"},
                      chunksize=2_000_000):
    chunks.append(ch[ch.is_recommended][["app_id", "user_id", "hours"]])
df = pd.concat(chunks, ignore_index=True)
print("положительных отзывов:", len(df))

# 2. подвыборка
top_games = df.app_id.value_counts().head(N_GAMES).index
df = df[df.app_id.isin(top_games)]
cnt = df.user_id.value_counts()
df = df[df.user_id.isin(cnt[cnt >= MIN_USER_GAMES].index)]
users = df.user_id.drop_duplicates().sample(N_USERS, random_state=SEED)
df = df[df.user_id.isin(users)].drop_duplicates(["user_id", "app_id"]).copy()

# 3. индексы: id -> номер строки/столбца матрицы
user_ids = np.sort(df.user_id.unique())
app_ids = np.sort(df.app_id.unique())
u_idx = {int(u): i for i, u in enumerate(user_ids)}
i_idx = {int(a): i for i, a in enumerate(app_ids)}
df["u"] = df.user_id.map(u_idx)
df["i"] = df.app_id.map(i_idx)

# 4. train/test: у каждого пользователя прячем 20% игр
df = df.sample(frac=1, random_state=SEED)
df["pos"] = df.groupby("u").cumcount()
n = df.groupby("u")["u"].transform("size")
df["is_test"] = df["pos"] >= np.floor(n * (1 - TEST_FRAC) + 1e-9)
train, test = df[~df.is_test], df[df.is_test]

# 5. сохраняем
os.makedirs(OUT, exist_ok=True)
X = csr_matrix((1 + np.log1p(train.hours.values), (train.u.values, train.i.values)),
               shape=(len(user_ids), len(app_ids)))
save_npz(f"{OUT}/train.npz", X)
test[["u", "i"]].to_parquet(f"{OUT}/test.parquet")
json.dump({"user_ids": [int(x) for x in user_ids], "app_ids": [int(x) for x in app_ids]},
          open(f"{OUT}/mappings.json", "w"))
print("users:", len(user_ids), "games:", len(app_ids),
      "train:", len(train), "test:", len(test),
      "sparsity: %.4f%%" % (100 * len(df) / (len(user_ids) * len(app_ids))))