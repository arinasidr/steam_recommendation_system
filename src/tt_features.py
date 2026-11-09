import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import OUT


def build_tag_features(games):
    # Сохраняем многословные теги как один признак:
    # "Open World" -> "Open_World".
    texts = games["tags"].apply(
        lambda tags: " ".join(tag.replace(" ", "_") for tag in tags)
    )

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        dtype=np.float32,
    )

    features = vectorizer.fit_transform(texts)

    # Two-Tower получит обычную матрицу float32.
    return features.toarray(), vectorizer

def build_description_features(games):
    from sentence_transformers import SentenceTransformer

    descriptions = games["description"].fillna("").str.strip()
    has_description = descriptions.ne("").to_numpy()

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    features = np.zeros(
        (len(games), model.get_embedding_dimension()),
        dtype=np.float32,
    )

    if has_description.any():
        features[has_description] = model.encode(
            descriptions[has_description].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    return features

if __name__ == "__main__":
    games = pd.read_parquet(f"{OUT}/games.parquet")

    with open(f"{OUT}/mappings.json", encoding="utf-8") as file:
        mappings = json.load(file)

    # Порядок строк совпадает со столбцами матрицы взаимодействий.
    games = games.set_index("app_id").loc[mappings["app_ids"]]

    features, vectorizer = build_tag_features(games)

    np.save(f"{OUT}/tt_tags.npy", features)

    print("Размер матрицы:", features.shape)
    print("Количество тегов:", len(vectorizer.get_feature_names_out()))
    print("Игры без признаков:", int((features.sum(axis=1) == 0).sum()))
    print(f"Сохранено: {OUT}/tt_tags.npy")

    description_features = build_description_features(games)
    np.save(f"{OUT}/tt_descriptions.npy", description_features)

    print("Размер признаков описаний:", description_features.shape)
    print(
        "Игры без описаний:",
        int((np.linalg.norm(description_features, axis=1) == 0).sum()),
    )
    print(f"Сохранено: {OUT}/tt_descriptions.npy")

    combined_features = np.concatenate(
        [features, description_features],
        axis=1,
    )

    combined_features = normalize(
        combined_features,
        norm="l2",
        axis=1,
    ).astype(np.float32)

    np.save(f"{OUT}/tt_combined.npy", combined_features)
    np.save(f"{OUT}/item_content.npy", combined_features)

    print("Размер объединённых признаков:", combined_features.shape)
    print(f"Сохранено: {OUT}/tt_combined.npy")
    print(f"Для модели Паши сохранено: {OUT}/item_content.npy")