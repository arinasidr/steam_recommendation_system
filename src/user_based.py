from common import load_data, evaluate
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np

X, test_sets, maps = load_data()

#поиск соседей
N_NEIGHBORS = 50

knn = NearestNeighbors(
    n_neighbors=N_NEIGHBORS + 1,
    metric="cosine",
    algorithm="brute"
)

knn.fit(X)

distances, neighbors = knn.kneighbors(X)

#для каждого пользователя самый похожий пользователь это он сам
#поэтому вычеркиваем первого
neighbors = neighbors[:, 1:]
distances = distances[:, 1:]

similarities = 1 - distances

#получаем оценки игр
n_users = X.shape[0]

#rows - для какого пользователя
#cols - какой у него сосед
#data - насколько этот сосед похож
rows = np.repeat(
    np.arange(n_users),
    N_NEIGHBORS
)

cols = neighbors.ravel()
data = similarities.ravel()

user_similarity = csr_matrix(
    (data, (rows, cols)),
    shape=(n_users, n_users)
)

# считаем score каждой игры:
# чем больше похожих пользователей играли в игру
# и чем сильнее они похожи на текущего пользователя,
# тем выше итоговый score этой игры
scores = (user_similarity @ X).toarray().astype("float32")
print(scores.shape)

metrics = evaluate(
    scores,
    X,
    test_sets
)

print("user-based CF:", metrics)

#получили метрики, они нас устраивают, но все равно
#попробуем поэксперементировать внутри этого подхода

# улучшение 1: используем только факт взаимодействия с игрой,
# без учета количества часов

X_binary = X.copy().astype("float32")
X_binary.data[:] = 1.0

knn_binary = NearestNeighbors(
    n_neighbors=N_NEIGHBORS + 1,
    metric="cosine",
    algorithm="brute"
)

# ищем похожих пользователей уже по бинарной истории игр
knn_binary.fit(X_binary)

distances_binary, neighbors_binary = knn_binary.kneighbors(X_binary)

# убираем самого пользователя
neighbors_binary = neighbors_binary[:, 1:]
distances_binary = distances_binary[:, 1:]

similarities_binary = 1 - distances_binary

rows_binary = np.repeat(
    np.arange(n_users),
    N_NEIGHBORS
)

cols_binary = neighbors_binary.ravel()
data_binary = similarities_binary.ravel()

user_similarity_binary = csr_matrix(
    (data_binary, (rows_binary, cols_binary)),
    shape=(n_users, n_users)
)

# score игры зависит от того, сколько похожих пользователей
# взаимодействовали с ней, с учетом степени их похожести
scores_binary = (
    user_similarity_binary @ X_binary
).toarray().astype("float32")

metrics_binary = evaluate(
    scores_binary,
    X,
    test_sets
)

print("user-based CF без учета часов:", metrics_binary)


# улучшение 2: проверяем разное количество соседей
def evaluate_user_based_with_k(X_binary, n_neighbors):
    knn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="cosine",
        algorithm="brute"
    )

    knn.fit(X_binary)

    distances, neighbors = knn.kneighbors(X_binary)

    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    similarities = 1 - distances

    n_users = X_binary.shape[0]

    rows = np.repeat(
        np.arange(n_users),
        n_neighbors
    )

    cols = neighbors.ravel()
    data = similarities.ravel()

    user_similarity = csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_users)
    )

    scores = (
        user_similarity @ X_binary
    ).toarray().astype("float32")

    return evaluate(
        scores,
        X,
        test_sets
    )


# пробуем разное количество соседей
for n_neighbors in [20, 50, 100, 150, 200]:
    metrics_k = evaluate_user_based_with_k(
        X_binary,
        n_neighbors
    )

    print(
        f"user-based CF, {n_neighbors} соседей:",
        metrics_k
    )