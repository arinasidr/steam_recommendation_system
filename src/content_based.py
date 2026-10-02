from common import load_data, evaluate
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import os

X, test_sets, maps = load_data()
results = []

def log_result(name, metrics):
    print(f"{name}:", metrics)
    results.append({"variant": name, **metrics})

games = pd.read_parquet("data/processed/games.parquet")

#реализуем TF-IDF: преобразуем теги в числа, чтобы сравнивать игры математически
#сначала переводим наши теги из списков в строки
tags_string = games["tags"].apply(
    lambda tags: " ".join(tag.replace(" ", "_") for tag in tags)
)

#создаем TF-IDF преобразователь
vectorizer = TfidfVectorizer(
    analyzer=lambda text: text.split()
)

# обучаем TF-IDF на тегах и сразу преобразуем их в числа
#на выходе имеем уже матрицу (игры х теги)
game_features = vectorizer.fit_transform(tags_string)

print(game_features.shape)

#получаем профили интересов каждого пользователя
user_profiles = X @ game_features
print(user_profiles.shape)

#получаем финальные scores для всех игр и преобразуем в обычный массив
scores = (user_profiles @ game_features.T).toarray().astype("float32")
print(scores.shape)

# оцениваем, насколько каждая игра подходит каждому пользователю
metrics = evaluate(scores, X, test_sets)
log_result("content-based (теги, с учётом часов)", metrics)

#получили метрики, сейчас они не очень, попробуем поэксперементировать внутри этого подхода
#улучшение 1: строим профиль только по факту взаимодействия с игрой, не учитывая количество часов
X_binary = X.copy().astype("float32")
X_binary.data[:] = 1.0

# строим новые профили пользователей
user_profiles_binary = X_binary @ game_features

# рассчитываем оценки релевантности всех игр для каждого пользователя
scores_binary = (
    user_profiles_binary @ game_features.T
).toarray().astype("float32")

# оцениваем улучшенный вариант
metrics_binary = evaluate(
    scores_binary,
    X,
    test_sets
)

log_result("content-based (теги) без учёта часов", metrics_binary)

#улучшение 2: добавляем к тегам описания игр
# пустые описания заменяем пустой строкой
descriptions = games["description"].fillna("")

# отдельный TF-IDF для текстовых описаний
description_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    min_df=2
)

description_features = description_vectorizer.fit_transform(descriptions)

combined_features = sparse.hstack(
    [game_features, description_features],
    format="csr"
)

# строим профиль пользователя уже по тегам + описаниям
user_profiles_combined = X @ combined_features

# рассчитываем оценки релевантности всех игр
scores_combined = (
    user_profiles_combined @ combined_features.T
).toarray().astype("float32")

# оцениваем новый вариант
metrics_combined = evaluate(
    scores_combined,
    X,
    test_sets
)

log_result("content-based, теги + описания", metrics_combined)

os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/content_based.csv", index=False)
print("результаты сохранены в results/content_based.csv")