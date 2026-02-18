"""
Monte Carlo Simulation Model for Singapore TOTO Prediction

Uses historical frequency as sampling weights to simulate 1,000,000 draws.
Counts each number's appearance frequency across all simulated draws.
Also counts pair co-occurrences to identify strong number pairs.
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]
ALL_NUMBERS = list(range(1, 50))

# Simulation parameters
N_SIMULATIONS = 1_000_000
NUMBERS_PER_DRAW = 6
TOTAL_NUMBERS = 49


def _extract_draw_numbers(row):
    """Extract the 6 main numbers from a draw row."""
    return [int(row[c]) for c in NUM_COLS]


def _compute_historical_weights(df, recency_decay=0.998):
    """
    Compute weighted frequency for each number.
    More recent draws get exponentially higher weight.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data sorted by date ascending.
    recency_decay : float
        Decay factor per draw (closer to 1 = slower decay).

    Returns
    -------
    np.array of shape (49,) with sampling weights for numbers 1-49.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n_draws = len(df)

    weights = np.zeros(TOTAL_NUMBERS, dtype=np.float64)

    for idx, (_, row) in enumerate(df.iterrows()):
        # Exponential recency weight: most recent draw gets weight ~1
        draw_weight = recency_decay ** (n_draws - 1 - idx)
        nums = _extract_draw_numbers(row)
        for n in nums:
            weights[n - 1] += draw_weight

    # Ensure no zero weights (Laplace smoothing)
    weights += 0.01

    # Normalize to probability distribution
    weights /= weights.sum()

    return weights


def _run_simulation(weights, n_sims=N_SIMULATIONS, seed=42):
    """
    Run Monte Carlo simulation: draw 6 numbers from 1-49 using weighted
    sampling (without replacement) for each simulation.

    Parameters
    ----------
    weights : np.array of shape (49,)
        Sampling probability for each number (1-49).
    n_sims : int
        Number of simulated draws.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    single_counts : np.array of shape (49,) with appearance counts
    pair_counts : Counter with (a, b) tuple -> count
    """
    rng = np.random.RandomState(seed)

    single_counts = np.zeros(TOTAL_NUMBERS, dtype=np.int64)
    pair_counts = Counter()

    # Process in batches for efficiency
    batch_size = 50_000
    n_batches = (n_sims + batch_size - 1) // batch_size

    numbers = np.arange(1, 50)

    for batch_idx in range(n_batches):
        current_batch = min(batch_size, n_sims - batch_idx * batch_size)

        if batch_idx % 4 == 0:
            progress = (batch_idx * batch_size) / n_sims * 100
            print(f"    Simulation progress: {progress:.0f}% ({batch_idx * batch_size:,}/{n_sims:,})")

        # For each draw in batch, sample 6 numbers without replacement
        for _ in range(current_batch):
            # Weighted sampling without replacement
            drawn = rng.choice(numbers, size=NUMBERS_PER_DRAW, replace=False, p=weights)
            drawn_sorted = tuple(sorted(drawn))

            # Count singles
            for n in drawn_sorted:
                single_counts[n - 1] += 1

            # Count pairs (only track for top pair analysis)
            for pair in combinations(drawn_sorted, 2):
                pair_counts[pair] += 1

    print(f"    Simulation progress: 100% ({n_sims:,}/{n_sims:,})")

    return single_counts, pair_counts


def _analyze_results(single_counts, pair_counts, n_sims):
    """
    Analyze Monte Carlo simulation results.

    Returns
    -------
    number_scores : dict of {number: appearance_rate}
    top_pairs : list of ((a, b), rate) sorted by rate descending
    """
    # Number appearance rates
    number_scores = {}
    for n in ALL_NUMBERS:
        number_scores[n] = single_counts[n - 1] / n_sims

    # Top pairs by co-occurrence rate
    pair_rates = []
    for (a, b), count in pair_counts.most_common(100):
        pair_rates.append(((a, b), count / n_sims))

    return number_scores, pair_rates


def predict(df):
    """
    Run Monte Carlo simulation to predict next draw probabilities.

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
    print("MONTE CARLO SIMULATION MODEL")
    print("=" * 60)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Total draws in dataset: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Simulations: {N_SIMULATIONS:,}")

    # Compute historical weights
    print("  [MonteCarlo] Computing historical frequency weights...")
    weights = _compute_historical_weights(df)

    # Display weight distribution info
    most_weighted_idx = np.argmax(weights)
    least_weighted_idx = np.argmin(weights)
    print(f"  [MonteCarlo] Highest weight: number {most_weighted_idx + 1} ({weights[most_weighted_idx]:.4f})")
    print(f"  [MonteCarlo] Lowest weight: number {least_weighted_idx + 1} ({weights[least_weighted_idx]:.4f})")
    print(f"  [MonteCarlo] Weight range ratio: {weights.max() / weights.min():.2f}x")

    # Run simulation
    print(f"  [MonteCarlo] Running {N_SIMULATIONS:,} simulations...")
    single_counts, pair_counts = _run_simulation(weights, N_SIMULATIONS)

    # Analyze results
    print("  [MonteCarlo] Analyzing simulation results...")
    number_scores, top_pairs = _analyze_results(single_counts, pair_counts, N_SIMULATIONS)

    # Build rankings
    rankings = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
    top_numbers = sorted([num for num, _ in rankings[:6]])

    # Expected appearance rate under uniform: 6/49 ~ 0.1224
    expected_uniform = NUMBERS_PER_DRAW / TOTAL_NUMBERS

    print(f"\n  Expected uniform rate: {expected_uniform:.4f}")
    print(f"  Top 6 numbers: {top_numbers}")
    print(f"  Top 10 rankings:")
    for i, (num, rate) in enumerate(rankings[:10]):
        deviation = ((rate - expected_uniform) / expected_uniform) * 100
        print(f"    {i+1:2d}. Number {num:2d} -> rate {rate:.4f} ({deviation:+.1f}% vs uniform)")

    print(f"\n  Top 10 pairs:")
    for i, ((a, b), rate) in enumerate(top_pairs[:10]):
        print(f"    {i+1:2d}. ({a:2d}, {b:2d}) -> co-occurrence rate {rate:.4f}")
    print("=" * 60)

    return {
        "rankings": rankings,
        "top_numbers": top_numbers,
        "model_name": "MonteCarlo",
        "n_simulations": N_SIMULATIONS,
        "top_pairs": top_pairs[:20],
        "weights": weights.tolist(),
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.scraper import load_data

    df = load_data()
    result = predict(df)
    print(f"\nFinal top 6: {result['top_numbers']}")
