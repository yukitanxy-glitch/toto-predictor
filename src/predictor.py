"""
Board Generation & Multi-Draw Strategy for Singapore TOTO

Generates 3 prediction boards using ensemble model output and hard filters.
Also provides multi-draw rolling strategy.
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.filters import run_all_filters, get_board_stats


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_next_draw_info():
    """Determine the next TOTO draw date and number."""
    today = datetime.now().date()
    # Find next Monday or Thursday
    days_ahead_mon = (0 - today.weekday()) % 7
    days_ahead_thu = (3 - today.weekday()) % 7
    if days_ahead_mon == 0 and datetime.now().hour >= 19:
        days_ahead_mon = 7
    if days_ahead_thu == 0 and datetime.now().hour >= 19:
        days_ahead_thu = 7
    if days_ahead_mon == 0:
        days_ahead_mon = 7
    if days_ahead_thu == 0:
        days_ahead_thu = 7

    next_mon = today + timedelta(days=days_ahead_mon)
    next_thu = today + timedelta(days=days_ahead_thu)
    next_draw_date = min(next_mon, next_thu)
    day_name = "Monday" if next_draw_date.weekday() == 0 else "Thursday"
    return {
        "date": next_draw_date.strftime("%Y-%m-%d"),
        "day": day_name,
    }


def _select_with_filters(candidates, df, count=6, cluster_centroids=None, max_attempts=500):
    """
    Select `count` numbers from candidates that pass all hard filters.
    Candidates is a list of (number, score) sorted by score descending.
    """
    best_board = None
    best_passed = 0

    for attempt in range(max_attempts):
        if attempt == 0:
            # First attempt: top N numbers
            board = [c[0] for c in candidates[:count]]
        else:
            # Subsequent attempts: swap lowest-scored number(s)
            base = [c[0] for c in candidates[:count]]
            n_swaps = min(attempt, count - 1)
            n_swaps = min(n_swaps, 3)  # Max 3 swaps
            pool = [c[0] for c in candidates[count:count + 20]]
            if not pool:
                pool = list(range(1, 50))
            for _ in range(n_swaps):
                idx_to_swap = random.randint(0, len(base) - 1)
                replacement = random.choice(pool)
                while replacement in base:
                    replacement = random.choice(pool)
                base[idx_to_swap] = replacement
            board = base

        board = sorted(set(board))
        if len(board) != count:
            continue

        result = run_all_filters(board, df, cluster_centroids)
        if result["all_passed"]:
            return board, result

        if result["passed_count"] > best_passed:
            best_passed = result["passed_count"]
            best_board = board

    # Return best attempt
    if best_board is None:
        best_board = [c[0] for c in candidates[:count]]
    result = run_all_filters(best_board, df, cluster_centroids)
    return best_board, result


# ── Additional Number Prediction ─────────────────────────────────────────

def predict_additional_number(df):
    """Predict the additional number using frequency and recency analysis."""
    from collections import Counter

    # Frequency of additional numbers
    add_counts = Counter(df["additional_number"].values)

    # Recent frequency (last 3 months)
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    recent = df[df["date"] >= cutoff]
    recent_counts = Counter(recent["additional_number"].values)

    # Recency — last appearance
    last_seen = {}
    for idx, row in df.sort_values("date").iterrows():
        last_seen[int(row["additional_number"])] = row["date"]

    max_date = df["date"].max()
    scores = {}
    for num in range(1, 50):
        freq_score = add_counts.get(num, 0) / max(len(df), 1)
        recent_score = recent_counts.get(num, 0) / max(len(recent), 1)
        gap = (max_date - last_seen.get(num, df["date"].min())).days
        recency_score = min(gap / 100, 1.0)  # Higher gap = higher recency score

        scores[num] = 0.3 * freq_score + 0.3 * recent_score + 0.4 * recency_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "rankings": ranked,
        "top3": [r[0] for r in ranked[:3]],
        "recommended": ranked[0][0],
    }


# ── Board Generation ─────────────────────────────────────────────────────

def generate_board1_ensemble(ensemble_rankings, df, cluster_centroids=None):
    """
    Board 1 — Ensemble High Probability
    Top 6 from ensemble ranking, validated against all hard filters.
    """
    board, filter_result = _select_with_filters(
        ensemble_rankings, df, count=6, cluster_centroids=cluster_centroids
    )
    stats = get_board_stats(board)
    return {
        "name": "Board 1 — Ensemble High Probability",
        "strategy": "Top numbers from ensemble model consensus",
        "numbers": sorted(board),
        "filter_result": filter_result,
        "stats": stats,
    }


def generate_board2_diversified(ensemble_rankings, df, board1_numbers, cluster_centroids=None):
    """
    Board 2 — Diversified / Contrarian
    3 hot numbers + 3 overdue cold numbers, no overlap with Board 1.
    """
    from collections import Counter

    # Hot numbers: most frequent in last 3 months
    cutoff_3m = df["date"].max() - pd.Timedelta(days=90)
    recent_3m = df[df["date"] >= cutoff_3m]
    hot_counter = Counter()
    for _, row in recent_3m.iterrows():
        for i in range(1, 7):
            hot_counter[int(row[f"num{i}"])] += 1
    hot_ranked = [n for n, _ in hot_counter.most_common(49)]

    # Cold / overdue: longest gap since last appearance
    last_seen = {}
    for _, row in df.sort_values("date").iterrows():
        for i in range(1, 7):
            last_seen[int(row[f"num{i}"])] = row["date"]

    max_date = df["date"].max()
    gaps = {n: (max_date - last_seen.get(n, df["date"].min())).days for n in range(1, 50)}
    cold_ranked = sorted(gaps.keys(), key=lambda n: gaps[n], reverse=True)

    # Select 3 hot + 3 cold, no overlap with Board 1
    b1_set = set(board1_numbers)
    hot_picks = [n for n in hot_ranked if n not in b1_set][:3]
    cold_picks = [n for n in cold_ranked if n not in b1_set and n not in hot_picks][:3]
    candidates = hot_picks + cold_picks

    if len(candidates) < 6:
        # Fill from ensemble
        for num, _ in ensemble_rankings:
            if num not in b1_set and num not in candidates:
                candidates.append(num)
            if len(candidates) >= 6:
                break

    # Create candidate ranking
    candidate_rankings = [(n, 1.0 - i * 0.05) for i, n in enumerate(candidates)]
    # Add more candidates for swapping
    for num, score in ensemble_rankings:
        if num not in b1_set and num not in [c[0] for c in candidate_rankings]:
            candidate_rankings.append((num, score * 0.5))

    board, filter_result = _select_with_filters(
        candidate_rankings, df, count=6, cluster_centroids=cluster_centroids
    )
    stats = get_board_stats(board)
    return {
        "name": "Board 2 — Diversified/Contrarian",
        "strategy": "3 hot (recent 3-month) + 3 overdue cold numbers",
        "numbers": sorted(board),
        "hot_picks": hot_picks,
        "cold_picks": cold_picks,
        "filter_result": filter_result,
        "stats": stats,
    }


def generate_board3_anti_sharing(ensemble_rankings, df, board1_numbers, board2_numbers,
                                  cluster_centroids=None):
    """
    Board 3 — Maximum Prize Value (Anti-Sharing)
    Avoids commonly picked popular numbers to maximize prize if won.
    """
    # Popularity penalty scores
    popularity_penalty = {}
    birthday_range = set(range(1, 32))
    sg_lucky = {7, 8, 13, 14, 28, 38}

    for num in range(1, 50):
        penalty = 0
        if num in birthday_range:
            penalty += 0.3
        if num in sg_lucky:
            penalty += 0.5
        # Numbers divisible by 7 are often popular
        if num % 7 == 0:
            penalty += 0.1
        popularity_penalty[num] = penalty

    # Select from ensemble top 20, minimize popularity penalty
    b1_set = set(board1_numbers)
    b2_set = set(board2_numbers)
    top20 = [(num, score) for num, score in ensemble_rankings[:25]
             if num not in b1_set and num not in b2_set]

    # Sort by (ensemble_score - popularity_penalty)
    adjusted = [(num, score - popularity_penalty.get(num, 0)) for num, score in top20]
    adjusted.sort(key=lambda x: x[1], reverse=True)

    # Add more candidates
    for num, score in ensemble_rankings[25:]:
        adjusted.append((num, score - popularity_penalty.get(num, 0)))

    board, filter_result = _select_with_filters(
        adjusted, df, count=6, cluster_centroids=cluster_centroids
    )
    stats = get_board_stats(board)
    return {
        "name": "Board 3 — Maximum Prize Value",
        "strategy": "Anti-sharing: avoids birthday range and SG lucky numbers",
        "numbers": sorted(board),
        "filter_result": filter_result,
        "stats": stats,
    }


def generate_all_boards(ensemble_rankings, df, cluster_centroids=None):
    """Generate all 3 boards + additional number prediction."""
    add_pred = predict_additional_number(df)
    next_draw = _get_next_draw_info()

    board1 = generate_board1_ensemble(ensemble_rankings, df, cluster_centroids)
    board1["additional_number"] = add_pred["recommended"]

    board2 = generate_board2_diversified(
        ensemble_rankings, df, board1["numbers"], cluster_centroids
    )
    board2["additional_number"] = add_pred["top3"][1] if len(add_pred["top3"]) > 1 else add_pred["recommended"]

    board3 = generate_board3_anti_sharing(
        ensemble_rankings, df, board1["numbers"], board2["numbers"], cluster_centroids
    )
    board3["additional_number"] = add_pred["top3"][2] if len(add_pred["top3"]) > 2 else add_pred["recommended"]

    return {
        "next_draw": next_draw,
        "boards": [board1, board2, board3],
        "additional_number_analysis": add_pred,
        "ensemble_rankings": ensemble_rankings,
    }


# ── Multi-Draw Strategy ──────────────────────────────────────────────────

def generate_multi_draw_strategy(ensemble_rankings, df, cluster_centroids=None):
    """
    Generate a 4-draw rolling strategy covering 2 weeks.
    Maximizes unique number coverage across draws.
    """
    all_boards = generate_all_boards(ensemble_rankings, df, cluster_centroids)
    b1 = all_boards["boards"][0]
    b2 = all_boards["boards"][1]
    b3 = all_boards["boards"][2]

    # Draw 1: Board 1 + Board 3
    draw1 = {"boards": [b1, b3], "label": "Draw 1"}

    # Draw 2: Board 2 + Board 1 variant
    # Board 1 variant: shift 1-2 numbers
    b1v_candidates = [(n, s) for n, s in ensemble_rankings if n not in b1["numbers"]]
    b1_variant_nums = list(b1["numbers"])
    if b1v_candidates:
        # Replace lowest-ranked number in B1 with next best from ensemble
        b1_variant_nums[-1] = b1v_candidates[0][0]
        if len(b1v_candidates) > 1:
            b1_variant_nums[-2] = b1v_candidates[1][0]
    b1_variant_nums = sorted(b1_variant_nums)
    b1_variant = {
        "name": "Board 1 Variant",
        "numbers": b1_variant_nums,
        "stats": get_board_stats(b1_variant_nums),
        "filter_result": run_all_filters(b1_variant_nums, df, cluster_centroids),
    }
    draw2 = {"boards": [b2, b1_variant], "label": "Draw 2"}

    # Draw 3: Board 1 + Board 2 variant
    b2v_candidates = [(n, s) for n, s in ensemble_rankings if n not in b2["numbers"]]
    b2_variant_nums = list(b2["numbers"])
    if b2v_candidates:
        b2_variant_nums[-1] = b2v_candidates[0][0]
    b2_variant_nums = sorted(b2_variant_nums)
    b2_variant = {
        "name": "Board 2 Variant",
        "numbers": b2_variant_nums,
        "stats": get_board_stats(b2_variant_nums),
        "filter_result": run_all_filters(b2_variant_nums, df, cluster_centroids),
    }
    draw3 = {"boards": [b1, b2_variant], "label": "Draw 3"}

    # Draw 4: Board 2 variant + Board 3
    draw4 = {"boards": [b2_variant, b3], "label": "Draw 4"}

    draws = [draw1, draw2, draw3, draw4]

    # Calculate unique number coverage
    all_numbers = set()
    for draw in draws:
        for b in draw["boards"]:
            all_numbers.update(b["numbers"])
    unique_coverage = len(all_numbers)

    # Expected cumulative hit rate estimation
    # For a truly random lottery: P(match k of 6) = C(6,k)*C(43,6-k)/C(49,6)
    from math import comb
    total_combos = comb(49, 6)

    def expected_matches_per_board():
        """Expected number of matches for a single board = 6*6/49 ≈ 0.735"""
        return 6 * 6 / 49

    boards_per_strategy = sum(len(d["boards"]) for d in draws)  # ~8 boards over 4 draws
    expected_per_draw = expected_matches_per_board()

    return {
        "draws": draws,
        "unique_numbers_covered": unique_coverage,
        "total_boards_played": boards_per_strategy,
        "expected_match_per_board": round(expected_per_draw, 3),
        "projections": {
            "4_draws": {
                "boards_played": boards_per_strategy,
                "unique_numbers": unique_coverage,
            },
            "10_draws": {
                "boards_played": boards_per_strategy * 2.5,
                "note": "Rotate strategy every 4 draws with updated data",
            },
            "50_draws": {
                "boards_played": boards_per_strategy * 12.5,
                "note": "~6 months of draws, re-train models each cycle",
            },
        },
        "comparison": {
            "fixed_boards": "Playing same 3 boards every draw — lower coverage",
            "rotating_strategy": "This strategy — maximizes unique number exposure",
            "random_boards": "Random selection — no statistical advantage",
        },
    }
