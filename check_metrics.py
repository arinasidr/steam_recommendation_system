import sys
sys.path.append("src")
import numpy as np
from common import load_data, evaluate

X, test_sets, maps = load_data()
print("матрица:", X.shape)

# случайные оценки: метрики должны быть около нуля
rnd = np.random.rand(*X.shape)
print("random ", evaluate(rnd, X, test_sets))

# популярное: должно быть заметно лучше случайного
pop = np.asarray((X > 0).sum(axis=0)).ravel()
print("popular", evaluate(np.tile(pop, (X.shape[0], 1)), X, test_sets))