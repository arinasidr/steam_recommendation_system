# steam_recommendation_system

## Запуск
1. python -m venv .venv и активировать его
2. pip install -r requirements.txt
3. Скачать датасет (Game Recommendations on Steam, Kaggle) в data/raw/
4. python src/prepare_data.py
5. python check_metrics.py

## Как подключить свой подход
Подход получает X_train (scipy sparse, users x games) и возвращает
np.ndarray той же формы (оценки, больше = лучше).
Оценка: evaluate(scores, X_train, test_sets) из src/common.py.
Игры из train маскируются внутри evaluate, делать это самому не нужно.


## Структура проекта

| Файл | Что делает |
|---|---|
| `src/config.py` | Все параметры в одном месте: seed, размер подвыборки, доля теста, K |
| `src/prepare_data.py` | Читает сырой датасет, делает подвыборку, делит на train/test, сохраняет результат в `data/processed/` |
| `src/common.py` | `load_data()` загружает подготовленные данные; `evaluate()` считает Precision@K, Recall@K, NDCG, coverage |
| `check_metrics.py` | Проверка, что всё работает: случайные оценки против «популярного» |
| `requirements.txt` | Список библиотек для `pip install -r` |
| `data/raw/` | Сюда кладётся скачанный датасет (в git не попадает) |
| `data/processed/` | Создаётся `prepare_data.py` (в git не попадает) |

Подходы (content-based, item-based CF, user-based CF, популярное) добавляются в `src/` отдельными файлами, например `src/item_cf.py`.