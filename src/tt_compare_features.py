import os
import time

import numpy as np
import pandas as pd

from common import load_data, evaluate
from config import OUT
from two_tower import train_two_tower


def main():
    X, test_sets, mappings = load_data()

    variants = {
        "two-tower: теги": "tt_tags.npy",
        "two-tower: описания": "tt_descriptions.npy",
        "two-tower: теги + описания": "tt_combined.npy",
    }

    results = []

    for name, filename in variants.items():
        content = np.load(f"{OUT}/{filename}")

        print(f"\n{name}, признаки: {content.shape}", flush=True)
        start = time.perf_counter()

        user_emb, item_emb = train_two_tower(
            X,
            content=content,
            use_id=True,
            dim=64,
            temp=0.05,
            epochs=15,
            verbose=True,
        )

        train_sec = time.perf_counter() - start
        metrics = evaluate(user_emb @ item_emb.T, X, test_sets)

        results.append({
            "variant": name,
            **metrics,
            "train_sec": round(train_sec, 1),
        })

        print(name, metrics, flush=True)

        # Сохраняем после каждого варианта, чтобы не потерять результат.
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(results).to_csv(
            "results/tt_features.csv",
            index=False,
        )

    print("\nСравнение признаков:")
    print(pd.DataFrame(results).round(4).to_string(index=False))


if __name__ == "__main__":
    main()