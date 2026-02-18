"""
Random Forest Model for Singapore TOTO Prediction

For each of the 49 numbers, builds a binary classifier predicting whether
that number will appear in the next draw based on engineered features:
- Draws since last appearance
- Rolling 10-draw frequency
- Rolling 30-draw frequency
- Burst/dormancy state (hot/cold/normal)
- Day of week (Monday=0, Thursday=3)
- Previous draw's numbers, sum, and odd/even split

Uses walk-forward validation to evaluate and train.
"""

import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]
ALL_NUMBERS = list(range(1, 50))


def _extract_draw_numbers(row):
    """Extract the 6 main numbers from a draw row."""
    return [int(row[c]) for c in NUM_COLS]


def _build_feature_matrix(df):
    """
    Build feature matrix for all draws. Each row corresponds to a draw,
    and for each of the 49 numbers we compute features.

    Returns:
        features: dict mapping number -> np.array of shape (n_draws, n_features)
        targets: dict mapping number -> np.array of shape (n_draws,) binary
    """
    df = df.sort_values("date").reset_index(drop=True)
    n_draws = len(df)

    # Pre-compute: for each draw, which numbers appeared
    draw_sets = []
    for _, row in df.iterrows():
        draw_sets.append(set(_extract_draw_numbers(row)))

    # Day of week encoding
    day_map = {"Monday": 0, "Thursday": 1, "Tuesday": 2, "Wednesday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    days = []
    for _, row in df.iterrows():
        dow = row.get("day_of_week", "Monday")
        days.append(day_map.get(str(dow), 0))

    features = {n: [] for n in ALL_NUMBERS}
    targets = {n: [] for n in ALL_NUMBERS}

    for i in range(1, n_draws):
        # Previous draw info
        prev_nums = sorted(list(draw_sets[i - 1]))
        prev_sum = sum(prev_nums)
        prev_odd_count = sum(1 for x in prev_nums if x % 2 == 1)
        prev_high_count = sum(1 for x in prev_nums if x >= 25)

        # Current draw info (target)
        current_nums = draw_sets[i]

        for n in ALL_NUMBERS:
            # Feature 1: draws since last appearance
            draws_since = 0
            for j in range(i - 1, -1, -1):
                if n in draw_sets[j]:
                    break
                draws_since += 1
            else:
                draws_since = i  # never appeared before

            # Feature 2: rolling 10-draw frequency
            lookback_10 = max(0, i - 10)
            freq_10 = sum(1 for j in range(lookback_10, i) if n in draw_sets[j]) / max(i - lookback_10, 1)

            # Feature 3: rolling 30-draw frequency
            lookback_30 = max(0, i - 30)
            freq_30 = sum(1 for j in range(lookback_30, i) if n in draw_sets[j]) / max(i - lookback_30, 1)

            # Feature 4: burst/dormancy state
            # Hot = appeared 3+ times in last 10 draws, Cold = 0 times, Normal = 1-2
            appearances_10 = sum(1 for j in range(lookback_10, i) if n in draw_sets[j])
            if appearances_10 >= 3:
                burst_state = 2  # hot
            elif appearances_10 == 0:
                burst_state = 0  # cold/dormant
            else:
                burst_state = 1  # normal

            # Feature 5: day of week
            day = days[i]

            # Feature 6: was number in previous draw?
            in_prev = 1 if n in draw_sets[i - 1] else 0

            # Feature 7: previous draw sum
            # Feature 8: previous draw odd count
            # Feature 9: previous draw high count
            # Feature 10: number's distance from mean of previous draw
            prev_mean = np.mean(prev_nums) if prev_nums else 25.0
            dist_from_prev_mean = abs(n - prev_mean)

            # Feature 11: number is odd
            is_odd = 1 if n % 2 == 1 else 0

            # Feature 12: number decade group (0-4)
            decade = (n - 1) // 10

            feature_vec = [
                draws_since,
                freq_10,
                freq_30,
                burst_state,
                day,
                in_prev,
                prev_sum,
                prev_odd_count,
                prev_high_count,
                dist_from_prev_mean,
                is_odd,
                decade,
            ]

            features[n].append(feature_vec)
            targets[n].append(1 if n in current_nums else 0)

    # Convert to numpy arrays
    for n in ALL_NUMBERS:
        features[n] = np.array(features[n], dtype=float)
        targets[n] = np.array(targets[n], dtype=int)

    return features, targets


def _walk_forward_validate(features, targets, n_splits=5, test_size_ratio=0.1):
    """
    Walk-forward validation: train on expanding window, test on next chunk.
    Returns mean AUC across numbers and splits.
    """
    n_samples = len(list(targets.values())[0])
    test_size = max(int(n_samples * test_size_ratio), 10)

    auc_scores = []
    for split_idx in range(n_splits):
        test_end = n_samples - split_idx * test_size
        test_start = test_end - test_size
        if test_start <= 50:
            break
        train_end = test_start

        split_aucs = []
        for n in ALL_NUMBERS:
            X_train = features[n][:train_end]
            y_train = targets[n][:train_end]
            X_test = features[n][test_start:test_end]
            y_test = targets[n][test_start:test_end]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            clf = RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
            )
            clf.fit(X_train, y_train)

            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                split_aucs.append(auc)
            except (ValueError, IndexError):
                continue

        if split_aucs:
            auc_scores.append(np.mean(split_aucs))

    return np.mean(auc_scores) if auc_scores else 0.5


def predict(df):
    """
    Train Random Forest classifiers for each number and predict probabilities
    for the next draw.

    Parameters
    ----------
    df : pd.DataFrame
        Historical TOTO data.

    Returns
    -------
    dict with:
        'rankings': list of (number, score) sorted by score descending
        'top_numbers': list of top 6 numbers
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL")
    print("=" * 60)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Total draws: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Build features
    print("  [RandomForest] Building feature matrix for all 49 numbers...")
    features, targets = _build_feature_matrix(df)

    n_samples = len(list(targets.values())[0])
    print(f"  [RandomForest] Feature matrix: {n_samples} samples x 12 features per number")

    # Walk-forward validation
    print("  [RandomForest] Running walk-forward validation...")
    mean_auc = _walk_forward_validate(features, targets, n_splits=3)
    print(f"  [RandomForest] Walk-forward mean AUC: {mean_auc:.4f}")

    # Train final models on all data and get predictions for next draw
    print("  [RandomForest] Training final models on full data...")
    probabilities = {}

    for n in ALL_NUMBERS:
        X = features[n]
        y = targets[n]

        if len(X) == 0 or len(np.unique(y)) < 2:
            probabilities[n] = 0.5
            continue

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X, y)

        # For prediction, use the last row's features (predicting next draw)
        last_features = X[-1:].copy()
        try:
            prob = clf.predict_proba(last_features)[0, 1]
        except (ValueError, IndexError):
            prob = 0.5
        probabilities[n] = prob

    # Sort by probability descending
    rankings = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_numbers = sorted([num for num, _ in rankings[:6]])

    # Feature importance (average across all number models)
    feature_names = [
        "draws_since_last", "freq_10", "freq_30", "burst_state", "day_of_week",
        "in_prev_draw", "prev_sum", "prev_odd_count", "prev_high_count",
        "dist_from_prev_mean", "is_odd", "decade_group",
    ]

    print(f"\n  Top 6 numbers: {top_numbers}")
    print(f"  Walk-forward AUC: {mean_auc:.4f}")
    print(f"  Top 10 rankings:")
    for i, (num, prob) in enumerate(rankings[:10]):
        print(f"    {i+1:2d}. Number {num:2d} -> probability {prob:.4f}")
    print("=" * 60)

    return {
        "rankings": rankings,
        "top_numbers": top_numbers,
        "model_name": "RandomForest",
        "validation_auc": mean_auc,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.scraper import load_data

    df = load_data()
    result = predict(df)
    print(f"\nFinal top 6: {result['top_numbers']}")
