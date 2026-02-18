"""
Combination Filtering Engine for Singapore TOTO

Hard elimination rules — any generated board failing these checks
must be rejected and regenerated.
"""
import numpy as np
import pandas as pd


def get_historical_sum_zone(df, coverage=0.70):
    """Calculate the sum range covering `coverage` % of historical wins."""
    sums = []
    for _, row in df.iterrows():
        s = sum(row[f"num{i}"] for i in range(1, 7))
        sums.append(s)
    sums = sorted(sums)
    n = len(sums)
    trim = int(n * (1 - coverage) / 2)
    return sums[trim], sums[n - trim - 1]


def get_historical_sum_zone_tight(df, coverage=0.50):
    """Tighter sum range covering 50% of wins."""
    return get_historical_sum_zone(df, coverage)


def filter_sum_rule(board, df):
    """Board sum must be within the historical 70% optimal zone."""
    lo, hi = get_historical_sum_zone(df)
    s = sum(board)
    passed = lo <= s <= hi
    return {
        "name": "Sum Rule",
        "passed": passed,
        "detail": f"Sum={s}, zone=[{lo},{hi}]",
    }


def filter_odd_even(board):
    """Must be 3/3, 4/2, or 2/4 odd/even split."""
    odd = sum(1 for n in board if n % 2 == 1)
    even = 6 - odd
    passed = (odd, even) in ((3, 3), (4, 2), (2, 4))
    return {
        "name": "Odd/Even Rule",
        "passed": passed,
        "detail": f"Odd={odd}, Even={even}",
    }


def filter_high_low(board):
    """Must be 3/3, 4/2, or 2/4 high/low split. Low=1-24, High=25-49."""
    low = sum(1 for n in board if n <= 24)
    high = 6 - low
    passed = (low, high) in ((3, 3), (4, 2), (2, 4))
    return {
        "name": "High/Low Rule",
        "passed": passed,
        "detail": f"Low={low}, High={high}",
    }


def filter_group_spread(board):
    """Must include numbers from at least 3 different decade groups."""
    groups = set()
    for n in board:
        if n <= 9:
            groups.add("1-9")
        elif n <= 19:
            groups.add("10-19")
        elif n <= 29:
            groups.add("20-29")
        elif n <= 39:
            groups.add("30-39")
        else:
            groups.add("40-49")
    passed = len(groups) >= 3
    return {
        "name": "Group Spread Rule",
        "passed": passed,
        "detail": f"Groups={len(groups)}: {sorted(groups)}",
    }


def filter_consecutive(board):
    """Maximum 2 consecutive numbers allowed. Reject 3+ consecutive."""
    sorted_board = sorted(board)
    max_consec = 1
    current_consec = 1
    for i in range(1, len(sorted_board)):
        if sorted_board[i] == sorted_board[i - 1] + 1:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 1
    passed = max_consec <= 2
    return {
        "name": "Consecutive Rule",
        "passed": passed,
        "detail": f"Max consecutive={max_consec}",
    }


def filter_no_recent_duplicate(board, df, lookback=50):
    """Board must not exactly match any winning combo from past `lookback` draws."""
    recent = df.sort_values("date").iloc[-lookback:]
    board_set = set(board)
    for _, row in recent.iterrows():
        winning = set(row[f"num{i}"] for i in range(1, 7))
        if board_set == winning:
            return {
                "name": "No Recent Duplicate",
                "passed": False,
                "detail": f"Matches draw {row['draw_number']}",
            }
    return {
        "name": "No Recent Duplicate",
        "passed": True,
        "detail": "No duplicate found",
    }


def filter_cluster_fit(board, cluster_centroids=None):
    """
    Board must fall within or near a top cluster profile.
    If cluster data not available, passes by default.
    """
    if cluster_centroids is None or len(cluster_centroids) == 0:
        return {
            "name": "Cluster Fit",
            "passed": True,
            "detail": "No cluster data; auto-pass",
        }

    board_vec = np.array(sorted(board), dtype=float)
    min_dist = float("inf")
    for centroid in cluster_centroids:
        dist = np.linalg.norm(board_vec - np.array(centroid))
        min_dist = min(min_dist, dist)

    # Threshold: boards more than 30 units from nearest centroid are outliers
    threshold = 30.0
    passed = min_dist <= threshold
    return {
        "name": "Cluster Fit",
        "passed": passed,
        "detail": f"Min distance to centroid={min_dist:.1f}, threshold={threshold}",
    }


def run_all_filters(board, df, cluster_centroids=None):
    """
    Run all 7 hard filters on a board.
    Returns: dict with 'passed_count', 'total', 'results' list, 'all_passed' bool.
    """
    results = [
        filter_sum_rule(board, df),
        filter_odd_even(board),
        filter_high_low(board),
        filter_group_spread(board),
        filter_consecutive(board),
        filter_no_recent_duplicate(board, df),
        filter_cluster_fit(board, cluster_centroids),
    ]
    passed = sum(1 for r in results if r["passed"])
    return {
        "passed_count": passed,
        "total": len(results),
        "all_passed": passed == len(results),
        "results": results,
        "confidence": f"{passed}/{len(results)}",
    }


def get_board_stats(board):
    """Calculate summary stats for a board."""
    sorted_board = sorted(board)
    odd = sum(1 for n in board if n % 2 == 1)
    even = 6 - odd
    low = sum(1 for n in board if n <= 24)
    high = 6 - low
    total = sum(board)

    groups = {}
    for n in board:
        if n <= 9:
            g = "1-9"
        elif n <= 19:
            g = "10-19"
        elif n <= 29:
            g = "20-29"
        elif n <= 39:
            g = "30-39"
        else:
            g = "40-49"
        groups[g] = groups.get(g, 0) + 1

    return {
        "numbers": sorted_board,
        "sum": total,
        "odd_even": f"{odd}/{even}",
        "high_low": f"{high}/{low}",
        "group_spread": groups,
        "group_count": len(groups),
    }
