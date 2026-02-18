"""
Weighted Scoring Model for Singapore TOTO Prediction

Scores all 49 numbers using a composite of multiple statistical factors:
- Overall frequency (10yr): 10%
- Recent 3-month frequency: 15%
- Recent 6-month frequency: 10%
- Recency / overdue factor: 20%
- Pair correlation: 15%
- Odd/even balance: 8%
- High/low balance: 8%
- Sum range fitness: 7%
- Group spread fitness: 7%
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter


# Weight configuration
WEIGHTS = {
    "overall_freq": 0.10,
    "recent_3m_freq": 0.15,
    "recent_6m_freq": 0.10,
    "recency_overdue": 0.20,
    "pair_correlation": 0.15,
    "odd_even_balance": 0.08,
    "high_low_balance": 0.08,
    "sum_range": 0.07,
    "group_spread": 0.07,
}

ALL_NUMBERS = list(range(1, 50))
NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]


def _extract_draw_numbers(row):
    """Extract the 6 main numbers from a draw row."""
    return [int(row[c]) for c in NUM_COLS]


def _normalize_scores(scores_dict):
    """Min-max normalize a dict of {number: score} to [0, 1]."""
    vals = np.array(list(scores_dict.values()), dtype=float)
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-12:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - mn) / (mx - mn) for k, v in scores_dict.items()}


def _overall_frequency(df):
    """Frequency of each number across the entire dataset."""
    print("  [WeightedScoring] Computing overall frequency...")
    counts = Counter()
    for _, row in df.iterrows():
        for n in _extract_draw_numbers(row):
            counts[n] += 1
    total_draws = len(df)
    return {n: counts.get(n, 0) / max(total_draws, 1) for n in ALL_NUMBERS}


def _recent_frequency(df, months):
    """Frequency of each number in the most recent N months."""
    print(f"  [WeightedScoring] Computing recent {months}-month frequency...")
    df_sorted = df.sort_values("date")
    if len(df_sorted) == 0:
        return {n: 0.0 for n in ALL_NUMBERS}

    cutoff = df_sorted["date"].max() - pd.Timedelta(days=months * 30)
    recent = df_sorted[df_sorted["date"] >= cutoff]

    counts = Counter()
    for _, row in recent.iterrows():
        for n in _extract_draw_numbers(row):
            counts[n] += 1
    total = len(recent)
    return {n: counts.get(n, 0) / max(total, 1) for n in ALL_NUMBERS}


def _recency_overdue(df):
    """
    Score numbers based on how overdue they are.
    Numbers that haven't appeared recently get higher scores (mean reversion).
    """
    print("  [WeightedScoring] Computing recency/overdue factor...")
    df_sorted = df.sort_values("date").reset_index(drop=True)
    total_draws = len(df_sorted)
    last_seen = {}

    for idx, row in df_sorted.iterrows():
        for n in _extract_draw_numbers(row):
            last_seen[n] = idx

    # Expected gap: each number should appear roughly every 49/6 ~ 8.17 draws
    expected_gap = 49.0 / 6.0
    scores = {}
    for n in ALL_NUMBERS:
        if n in last_seen:
            draws_since = total_draws - 1 - last_seen[n]
            # Overdue ratio: how many expected cycles overdue
            overdue_ratio = draws_since / expected_gap
            # Use sigmoid-like mapping: more overdue = higher score, but capped
            scores[n] = 1.0 - np.exp(-0.3 * overdue_ratio)
        else:
            # Never appeared -- maximally overdue
            scores[n] = 1.0

    return scores


def _pair_correlation(df, top_n_pairs=200):
    """
    For each number, sum correlation scores with other numbers based on
    how often they co-occur versus expected co-occurrence.
    """
    print("  [WeightedScoring] Computing pair correlation scores...")
    pair_counts = Counter()
    single_counts = Counter()
    total_draws = len(df)

    for _, row in df.iterrows():
        nums = _extract_draw_numbers(row)
        for n in nums:
            single_counts[n] += 1
        for pair in combinations(sorted(nums), 2):
            pair_counts[pair] += 1

    # Expected co-occurrence under independence: P(a)*P(b)*total
    # Lift = observed / expected
    pair_lift = {}
    for (a, b), count in pair_counts.items():
        expected = (single_counts[a] / total_draws) * (single_counts[b] / total_draws) * total_draws
        if expected > 0:
            pair_lift[(a, b)] = count / expected
        else:
            pair_lift[(a, b)] = 1.0

    # For each number, sum the lift of its top pairs
    scores = {n: 0.0 for n in ALL_NUMBERS}
    for (a, b), lift in pair_lift.items():
        scores[a] += lift
        scores[b] += lift

    return scores


def _odd_even_balance_score(number, historical_odd_ratios):
    """
    Score a number based on whether adding it would help achieve balanced odd/even.
    Historical draws tend toward 3 odd + 3 even.
    """
    is_odd = number % 2 == 1
    # If historically draws lean toward balanced, we want numbers that help balance
    mean_odd_ratio = np.mean(historical_odd_ratios) if historical_odd_ratios else 0.5
    # If we have too many odds historically, favor evens and vice versa
    if is_odd:
        return 1.0 - abs(mean_odd_ratio - 0.5)
    else:
        return 1.0 - abs((1 - mean_odd_ratio) - 0.5)


def _compute_odd_even_scores(df):
    """Compute odd/even balance scores for all numbers."""
    print("  [WeightedScoring] Computing odd/even balance scores...")
    # Look at recent draws to see odd/even distribution
    recent = df.sort_values("date").tail(50)
    odd_ratios = []
    for _, row in recent.iterrows():
        nums = _extract_draw_numbers(row)
        odd_count = sum(1 for n in nums if n % 2 == 1)
        odd_ratios.append(odd_count / 6.0)

    return {n: _odd_even_balance_score(n, odd_ratios) for n in ALL_NUMBERS}


def _compute_high_low_scores(df):
    """
    Score numbers based on high/low balance.
    High = 25-49, Low = 1-24. Balanced draws tend to have ~3 of each.
    """
    print("  [WeightedScoring] Computing high/low balance scores...")
    recent = df.sort_values("date").tail(50)
    high_ratios = []
    for _, row in recent.iterrows():
        nums = _extract_draw_numbers(row)
        high_count = sum(1 for n in nums if n >= 25)
        high_ratios.append(high_count / 6.0)

    mean_high = np.mean(high_ratios) if high_ratios else 0.5
    scores = {}
    for n in ALL_NUMBERS:
        is_high = n >= 25
        if is_high:
            # If too many highs recently, lower score for highs
            scores[n] = 1.0 - abs(mean_high - 0.5)
        else:
            scores[n] = 1.0 - abs((1 - mean_high) - 0.5)
    return scores


def _compute_sum_range_scores(df):
    """
    Score numbers based on how well they fit into the typical sum range.
    Typical 6-number sums for TOTO (1-49) center around 120-180.
    """
    print("  [WeightedScoring] Computing sum range fitness scores...")
    sums = []
    for _, row in df.iterrows():
        nums = _extract_draw_numbers(row)
        sums.append(sum(nums))

    mean_sum = np.mean(sums)
    std_sum = np.std(sums) if len(sums) > 1 else 30.0

    # Ideal contribution per number: mean_sum / 6
    ideal_per_number = mean_sum / 6.0

    scores = {}
    for n in ALL_NUMBERS:
        # How close is this number to the ideal contribution?
        deviation = abs(n - ideal_per_number)
        # Gaussian-like scoring
        scores[n] = np.exp(-0.5 * (deviation / (std_sum / 6.0)) ** 2)
    return scores


def _compute_group_spread_scores(df):
    """
    Score numbers based on group spread (decades: 1-9, 10-19, 20-29, 30-39, 40-49).
    Balanced draws tend to cover multiple decades.
    """
    print("  [WeightedScoring] Computing group spread scores...")
    # Count how often each decade group is represented in recent draws
    recent = df.sort_values("date").tail(100)
    decade_freq = Counter()
    total = 0
    for _, row in recent.iterrows():
        nums = _extract_draw_numbers(row)
        decades_in_draw = set()
        for n in nums:
            decade = (n - 1) // 10  # 0-4 for decades 1-9, 10-19, ..., 40-49
            decades_in_draw.add(decade)
        for d in decades_in_draw:
            decade_freq[d] += 1
        total += 1

    # Score each number: numbers in underrepresented decades get bonus
    scores = {}
    for n in ALL_NUMBERS:
        decade = (n - 1) // 10
        # Inverse frequency: less frequent decades get higher score
        freq = decade_freq.get(decade, 0) / max(total, 1)
        # We want balanced coverage, so underrepresented decades score higher
        scores[n] = 1.0 - freq + 0.5  # Shift so all are positive
    return scores


def predict(df):
    """
    Main prediction function. Scores all 49 numbers and returns rankings.

    Parameters
    ----------
    df : pd.DataFrame
        Historical TOTO data with columns: draw_number, date, day_of_week,
        num1-num6, additional_number, group1_prize, group1_winners, is_synthetic.

    Returns
    -------
    dict with:
        'rankings': list of (number, score) sorted by score descending
        'top_numbers': list of top 6 numbers
    """
    print("\n" + "=" * 60)
    print("WEIGHTED SCORING MODEL")
    print("=" * 60)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Total draws in dataset: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Compute all component scores
    raw_scores = {}

    raw_scores["overall_freq"] = _overall_frequency(df)
    raw_scores["recent_3m_freq"] = _recent_frequency(df, months=3)
    raw_scores["recent_6m_freq"] = _recent_frequency(df, months=6)
    raw_scores["recency_overdue"] = _recency_overdue(df)
    raw_scores["pair_correlation"] = _pair_correlation(df)
    raw_scores["odd_even_balance"] = _compute_odd_even_scores(df)
    raw_scores["high_low_balance"] = _compute_high_low_scores(df)
    raw_scores["sum_range"] = _compute_sum_range_scores(df)
    raw_scores["group_spread"] = _compute_group_spread_scores(df)

    # Normalize each component to [0, 1]
    normalized = {}
    for key, scores in raw_scores.items():
        normalized[key] = _normalize_scores(scores)

    # Map component names to weight keys
    weight_map = {
        "overall_freq": "overall_freq",
        "recent_3m_freq": "recent_3m_freq",
        "recent_6m_freq": "recent_6m_freq",
        "recency_overdue": "recency_overdue",
        "pair_correlation": "pair_correlation",
        "odd_even_balance": "odd_even_balance",
        "high_low_balance": "high_low_balance",
        "sum_range": "sum_range",
        "group_spread": "group_spread",
    }

    # Compute final weighted score for each number
    print("  [WeightedScoring] Computing final weighted scores...")
    final_scores = {}
    for n in ALL_NUMBERS:
        score = 0.0
        for component, weight_key in weight_map.items():
            w = WEIGHTS.get(weight_key, 0.0)
            s = normalized[component].get(n, 0.0)
            score += w * s
        final_scores[n] = score

    # Sort by score descending
    rankings = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_numbers = sorted([num for num, _ in rankings[:6]])

    print(f"\n  Top 6 numbers: {top_numbers}")
    print(f"  Top 10 rankings:")
    for i, (num, score) in enumerate(rankings[:10]):
        print(f"    {i+1:2d}. Number {num:2d} -> score {score:.4f}")
    print("=" * 60)

    return {
        "rankings": rankings,
        "top_numbers": top_numbers,
        "model_name": "WeightedScoring",
        "component_scores": {key: dict(scores) for key, scores in normalized.items()},
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.scraper import load_data

    df = load_data()
    result = predict(df)
    print(f"\nFinal top 6: {result['top_numbers']}")
