"""
Markov Chain Model for Singapore TOTO Prediction

Builds a 49x49 transition matrix from historical draws. Each entry T[i][j]
represents the probability that number j appears in the next draw given
that number i appeared in the current draw.

Given the most recent draw's 6 numbers, computes transition probabilities
for all 49 numbers by averaging across the 6 source numbers.

Also implements a second-order Markov chain that considers pairs of
consecutive draws for richer transition modeling.
"""

import numpy as np
import pandas as pd
from collections import Counter

NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]
ALL_NUMBERS = list(range(1, 50))
TOTAL_NUMBERS = 49


def _extract_draw_numbers(row):
    """Extract the 6 main numbers from a draw row."""
    return [int(row[c]) for c in NUM_COLS]


def _build_first_order_transition_matrix(df):
    """
    Build a 49x49 first-order transition matrix.

    T[i][j] = P(number j+1 appears in draw t+1 | number i+1 appeared in draw t)

    This is computed by counting co-occurrences across consecutive draws
    and normalizing by row.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n_draws = len(df)

    # Transition counts
    transition_counts = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS), dtype=np.float64)
    row_totals = np.zeros(TOTAL_NUMBERS, dtype=np.float64)

    for i in range(n_draws - 1):
        current_nums = _extract_draw_numbers(df.iloc[i])
        next_nums = _extract_draw_numbers(df.iloc[i + 1])

        for cn in current_nums:
            for nn in next_nums:
                transition_counts[cn - 1][nn - 1] += 1
            row_totals[cn - 1] += len(next_nums)

    # Normalize to probabilities (with Laplace smoothing)
    transition_matrix = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS), dtype=np.float64)
    for i in range(TOTAL_NUMBERS):
        total = row_totals[i]
        if total > 0:
            # Laplace smoothing: add small constant to avoid zeros
            smoothed = transition_counts[i] + 0.1
            transition_matrix[i] = smoothed / smoothed.sum()
        else:
            # Uniform if no data
            transition_matrix[i] = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS

    return transition_matrix


def _build_second_order_transitions(df):
    """
    Build second-order transition probabilities.
    Considers pairs of numbers from draw t-1 and t to predict draw t+1.

    For efficiency, we aggregate: for each number n appearing at draw t,
    and each number m appearing at draw t-1, count what appeared at t+1.
    Then return an aggregated score per number.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n_draws = len(df)

    # For each (prev_num, curr_num) pair, count transitions to next numbers
    # This would be a 49x49x49 tensor, so we use a sparse approach
    second_order_counts = {}  # (prev, curr) -> Counter of next numbers

    for i in range(1, n_draws - 1):
        prev_nums = _extract_draw_numbers(df.iloc[i - 1])
        curr_nums = _extract_draw_numbers(df.iloc[i])
        next_nums = _extract_draw_numbers(df.iloc[i + 1])

        for pn in prev_nums:
            for cn in curr_nums:
                key = (pn, cn)
                if key not in second_order_counts:
                    second_order_counts[key] = Counter()
                for nn in next_nums:
                    second_order_counts[key][nn] += 1

    return second_order_counts


def _compute_transition_scores(transition_matrix, current_nums):
    """
    Given the transition matrix and the current draw's numbers,
    compute transition probability for each of the 49 numbers.

    For each target number j, the score is the average transition
    probability from each of the 6 current numbers to j.
    """
    scores = np.zeros(TOTAL_NUMBERS, dtype=np.float64)

    for cn in current_nums:
        scores += transition_matrix[cn - 1]

    # Average across source numbers
    scores /= len(current_nums)

    return scores


def _compute_second_order_scores(second_order_counts, prev_nums, curr_nums):
    """
    Compute second-order transition scores.
    """
    scores = np.zeros(TOTAL_NUMBERS, dtype=np.float64)
    total_pairs = 0

    for pn in prev_nums:
        for cn in curr_nums:
            key = (pn, cn)
            if key in second_order_counts:
                counts = second_order_counts[key]
                total = sum(counts.values())
                if total > 0:
                    for num, count in counts.items():
                        scores[num - 1] += count / total
                    total_pairs += 1

    if total_pairs > 0:
        scores /= total_pairs
    else:
        scores = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS

    return scores


def _compute_stationary_distribution(transition_matrix, n_iter=100):
    """
    Compute the stationary distribution of the Markov chain
    by power iteration.
    """
    n = transition_matrix.shape[0]
    pi = np.ones(n) / n  # start uniform

    for _ in range(n_iter):
        pi_new = pi @ transition_matrix
        # Normalize
        pi_new /= pi_new.sum()
        # Check convergence
        if np.allclose(pi, pi_new, atol=1e-10):
            break
        pi = pi_new

    return pi


def predict(df):
    """
    Build Markov chain transition matrix and predict next draw.

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
    print("MARKOV CHAIN MODEL")
    print("=" * 60)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Total draws: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Get last two draws for prediction
    last_draw = _extract_draw_numbers(df.iloc[-1])
    second_last_draw = _extract_draw_numbers(df.iloc[-2]) if len(df) >= 2 else last_draw
    print(f"  Most recent draw: {sorted(last_draw)}")
    print(f"  Second most recent: {sorted(second_last_draw)}")

    # Build first-order transition matrix
    print("  [Markov] Building first-order transition matrix (49x49)...")
    transition_matrix = _build_first_order_transition_matrix(df)

    # Compute first-order transition scores
    print("  [Markov] Computing first-order transition scores...")
    first_order_scores = _compute_transition_scores(transition_matrix, last_draw)

    # Build second-order transitions
    print("  [Markov] Building second-order transition model...")
    second_order_counts = _build_second_order_transitions(df)
    print(f"  [Markov] Second-order pairs tracked: {len(second_order_counts):,}")

    # Compute second-order scores
    print("  [Markov] Computing second-order transition scores...")
    second_order_scores = _compute_second_order_scores(
        second_order_counts, second_last_draw, last_draw
    )

    # Compute stationary distribution
    print("  [Markov] Computing stationary distribution...")
    stationary = _compute_stationary_distribution(transition_matrix)

    # Combine scores: 50% first-order, 30% second-order, 20% stationary
    print("  [Markov] Combining scores (50% 1st-order, 30% 2nd-order, 20% stationary)...")
    combined_scores = (
        0.50 * first_order_scores +
        0.30 * second_order_scores +
        0.20 * stationary
    )

    # Build rankings
    rankings_dict = {n: combined_scores[n - 1] for n in ALL_NUMBERS}
    rankings = sorted(rankings_dict.items(), key=lambda x: x[1], reverse=True)
    top_numbers = sorted([num for num, _ in rankings[:6]])

    # Transition matrix statistics
    max_transition = np.unravel_index(np.argmax(transition_matrix), transition_matrix.shape)
    print(f"\n  Strongest transition: {max_transition[0]+1} -> {max_transition[1]+1} "
          f"(prob={transition_matrix[max_transition]:.4f})")

    # Self-transition analysis (same number appearing in consecutive draws)
    self_trans = [transition_matrix[i][i] for i in range(TOTAL_NUMBERS)]
    avg_self_trans = np.mean(self_trans)
    print(f"  Average self-transition probability: {avg_self_trans:.4f}")

    print(f"\n  Top 6 numbers: {top_numbers}")
    print(f"  Top 10 rankings:")
    for i, (num, score) in enumerate(rankings[:10]):
        fo = first_order_scores[num - 1]
        so = second_order_scores[num - 1]
        st = stationary[num - 1]
        print(f"    {i+1:2d}. Number {num:2d} -> score {score:.4f} "
              f"(1st: {fo:.4f}, 2nd: {so:.4f}, stat: {st:.4f})")
    print("=" * 60)

    return {
        "rankings": rankings,
        "top_numbers": top_numbers,
        "model_name": "MarkovChain",
        "transition_matrix_shape": transition_matrix.shape,
        "stationary_distribution": stationary.tolist(),
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.scraper import load_data

    df = load_data()
    result = predict(df)
    print(f"\nFinal top 6: {result['top_numbers']}")
