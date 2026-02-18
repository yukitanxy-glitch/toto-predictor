"""
Singapore TOTO Lottery - Comprehensive Statistical Analysis Engine

Provides 13+ analysis functions covering frequency, pattern, temporal,
and correlation analyses on historical TOTO draw data.

Data schema expected:
    draw_number, date, day_of_week, num1-num6, additional_number,
    group1_prize, group1_winners, is_synthetic

Numbers range 1-49. Draws occur Monday and Thursday.
"""

import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_COLS = [f"num{i}" for i in range(1, 7)]
ALL_NUMBERS = list(range(1, 50))
LOW_BOUND = 24  # 1-24 = low, 25-49 = high


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the date column cast to datetime."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df


def _main_numbers_flat(df: pd.DataFrame) -> pd.Series:
    """Return a flat Series of every main number drawn (one row per number)."""
    return df[NUM_COLS].values.flatten()


def _rows_as_sets(df: pd.DataFrame) -> pd.Series:
    """Return a Series of frozensets, one per draw."""
    return df[NUM_COLS].apply(frozenset, axis=1)


def _rows_as_sorted_tuples(df: pd.DataFrame) -> pd.Series:
    """Return a Series of sorted tuples, one per draw."""
    return df[NUM_COLS].apply(lambda r: tuple(sorted(r)), axis=1)


def _recent_draws(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """Return draws within the last *months* months from the most recent draw."""
    df = _ensure_datetime(df)
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(months=months)
    return df[df["date"] >= cutoff]


# ===================================================================
# 1. Frequency Analysis
# ===================================================================

def frequency_analysis(df: pd.DataFrame) -> dict:
    """
    Count each number 1-49 as a main number across all draws.
    Also count appearances as the additional number.
    Compute overall frequency percentage and rank all 49 numbers.

    Returns
    -------
    dict with keys:
        main_counts   : dict {number: count}
        additional_counts : dict {number: count}
        overall_freq_pct  : dict {number: pct}
        ranked            : list of (number, main_count) sorted desc
        total_draws       : int
        dataframe         : pd.DataFrame summary
    """
    df = _ensure_datetime(df)
    total_draws = len(df)

    # Main number counts
    flat = _main_numbers_flat(df)
    main_counter = Counter(flat)
    main_counts = {n: main_counter.get(n, 0) for n in ALL_NUMBERS}

    # Additional number counts
    add_counter = Counter(df["additional_number"].dropna().astype(int))
    additional_counts = {n: add_counter.get(n, 0) for n in ALL_NUMBERS}

    # Overall frequency % (main only, out of 6 * total_draws possible slots)
    total_slots = 6 * total_draws if total_draws > 0 else 1
    overall_freq_pct = {n: round(100.0 * main_counts[n] / total_slots, 4)
                        for n in ALL_NUMBERS}

    ranked = sorted(main_counts.items(), key=lambda x: x[1], reverse=True)

    # Build a tidy DataFrame
    records = []
    for rank, (num, cnt) in enumerate(ranked, 1):
        records.append({
            "number": num,
            "main_count": cnt,
            "additional_count": additional_counts[num],
            "total_appearances": cnt + additional_counts[num],
            "main_freq_pct": overall_freq_pct[num],
            "rank": rank,
        })
    summary_df = pd.DataFrame(records)

    return {
        "main_counts": main_counts,
        "additional_counts": additional_counts,
        "overall_freq_pct": overall_freq_pct,
        "ranked": ranked,
        "total_draws": total_draws,
        "dataframe": summary_df,
    }


# ===================================================================
# 2. Hot / Cold / Overdue Numbers
# ===================================================================

def hot_cold_numbers(df: pd.DataFrame) -> dict:
    """
    Hot  : top-10 most frequent in last 3 months AND last 6 months.
    Cold : top-10 least frequent in last 3 months AND last 6 months.
    Overdue : numbers with the longest gap since their last appearance.

    Returns
    -------
    dict with keys:
        hot_3m, hot_6m       : list of (number, count)
        cold_3m, cold_6m     : list of (number, count)
        overdue              : list of (number, gap_in_draws)
    """
    df = _ensure_datetime(df)
    df_sorted = df.sort_values("date").reset_index(drop=True)

    results = {}
    for label, months in [("3m", 3), ("6m", 6)]:
        recent = _recent_draws(df_sorted, months)
        flat = _main_numbers_flat(recent)
        counter = Counter(flat)
        full = {n: counter.get(n, 0) for n in ALL_NUMBERS}
        ranked_desc = sorted(full.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(full.items(), key=lambda x: x[1])

        results[f"hot_{label}"] = ranked_desc[:10]
        results[f"cold_{label}"] = ranked_asc[:10]

    # Overdue: gap since last appearance (in number of draws)
    total = len(df_sorted)
    last_seen = {}
    for idx, row in df_sorted.iterrows():
        for col in NUM_COLS:
            last_seen[int(row[col])] = idx

    overdue = []
    for n in ALL_NUMBERS:
        if n in last_seen:
            gap = total - 1 - last_seen[n]
        else:
            gap = total  # never appeared
        overdue.append((n, gap))
    overdue.sort(key=lambda x: x[1], reverse=True)
    results["overdue"] = overdue

    return results


# ===================================================================
# 3. Pair & Triplet Analysis
# ===================================================================

def pair_triplet_analysis(df: pd.DataFrame) -> dict:
    """
    Top 30 most frequent 2-number pairs.
    Top 15 most frequent 3-number triplets.
    For each: frequency count and last draw date.

    Returns
    -------
    dict with keys:
        top_pairs    : list of dicts {pair, count, last_date}
        top_triplets : list of dicts {triplet, count, last_date}
    """
    df = _ensure_datetime(df)
    df_sorted = df.sort_values("date").reset_index(drop=True)

    pair_counter: Counter = Counter()
    triplet_counter: Counter = Counter()
    pair_last: dict = {}
    triplet_last: dict = {}

    for _, row in df_sorted.iterrows():
        nums = tuple(sorted(int(row[c]) for c in NUM_COLS))
        draw_date = row["date"]

        for pair in combinations(nums, 2):
            pair_counter[pair] += 1
            pair_last[pair] = draw_date
        for triplet in combinations(nums, 3):
            triplet_counter[triplet] += 1
            triplet_last[triplet] = draw_date

    top_pairs = []
    for pair, count in pair_counter.most_common(30):
        top_pairs.append({
            "pair": pair,
            "count": count,
            "last_date": pair_last[pair],
        })

    top_triplets = []
    for triplet, count in triplet_counter.most_common(15):
        top_triplets.append({
            "triplet": triplet,
            "count": count,
            "last_date": triplet_last[triplet],
        })

    return {
        "top_pairs": top_pairs,
        "top_triplets": top_triplets,
    }


# ===================================================================
# 4. Odd / Even Distribution
# ===================================================================

def odd_even_distribution(df: pd.DataFrame) -> dict:
    """
    Frequency of each odd/even split (6/0 through 0/6).
    Most common and rarest splits.  Percentages.

    Returns
    -------
    dict with keys:
        distribution : dict {(odd, even): count}
        percentages  : dict {(odd, even): pct}
        most_common  : (odd, even)
        rarest       : (odd, even)
        dataframe    : pd.DataFrame
    """
    df = _ensure_datetime(df)
    total = len(df)
    dist: Counter = Counter()

    for _, row in df.iterrows():
        nums = [int(row[c]) for c in NUM_COLS]
        odd = sum(1 for n in nums if n % 2 == 1)
        even = 6 - odd
        dist[(odd, even)] += 1

    # Ensure all splits present
    for o in range(7):
        key = (o, 6 - o)
        if key not in dist:
            dist[key] = 0

    pcts = {k: round(100.0 * v / total, 2) if total else 0.0
            for k, v in dist.items()}

    most_common = max(dist, key=dist.get)
    rarest = min(dist, key=dist.get)

    records = [{"odd": k[0], "even": k[1], "count": v, "pct": pcts[k]}
               for k, v in sorted(dist.items())]
    summary_df = pd.DataFrame(records)

    return {
        "distribution": dict(dist),
        "percentages": pcts,
        "most_common": most_common,
        "rarest": rarest,
        "dataframe": summary_df,
    }


# ===================================================================
# 5. High / Low Distribution
# ===================================================================

def high_low_distribution(df: pd.DataFrame) -> dict:
    """
    Low = 1-24, High = 25-49.
    Same structure as odd/even analysis.

    Returns
    -------
    dict with keys:
        distribution, percentages, most_common, rarest, dataframe
    """
    df = _ensure_datetime(df)
    total = len(df)
    dist: Counter = Counter()

    for _, row in df.iterrows():
        nums = [int(row[c]) for c in NUM_COLS]
        low = sum(1 for n in nums if n <= LOW_BOUND)
        high = 6 - low
        dist[(low, high)] += 1

    for lo in range(7):
        key = (lo, 6 - lo)
        if key not in dist:
            dist[key] = 0

    pcts = {k: round(100.0 * v / total, 2) if total else 0.0
            for k, v in dist.items()}

    most_common = max(dist, key=dist.get)
    rarest = min(dist, key=dist.get)

    records = [{"low": k[0], "high": k[1], "count": v, "pct": pcts[k]}
               for k, v in sorted(dist.items())]
    summary_df = pd.DataFrame(records)

    return {
        "distribution": dict(dist),
        "percentages": pcts,
        "most_common": most_common,
        "rarest": rarest,
        "dataframe": summary_df,
    }


# ===================================================================
# 6. Sum Range Analysis
# ===================================================================

def sum_range_analysis(df: pd.DataFrame) -> dict:
    """
    Sum of 6 main numbers per draw.
    Statistics: mean, median, mode, std, min, max.
    Optimal zone covering ~70 % of wins, tighter 50 % zone.

    Returns
    -------
    dict with keys:
        stats        : dict of descriptive stats
        zone_70      : (low, high) covering central 70 %
        zone_50      : (low, high) covering central 50 %
        sums_series  : pd.Series of per-draw sums
    """
    df = _ensure_datetime(df)
    sums = df[NUM_COLS].sum(axis=1).astype(int)

    mode_result = stats.mode(sums, keepdims=True)
    mode_val = int(mode_result.mode[0])

    descriptive = {
        "mean": round(float(sums.mean()), 2),
        "median": float(sums.median()),
        "mode": mode_val,
        "std": round(float(sums.std()), 2),
        "min": int(sums.min()),
        "max": int(sums.max()),
        "skewness": round(float(sums.skew()), 4),
        "kurtosis": round(float(sums.kurtosis()), 4),
    }

    # Central zones via percentiles
    p15, p25, p75, p85 = np.percentile(sums, [15, 25, 75, 85])
    zone_70 = (int(np.floor(p15)), int(np.ceil(p85)))
    zone_50 = (int(np.floor(p25)), int(np.ceil(p75)))

    return {
        "stats": descriptive,
        "zone_70": zone_70,
        "zone_50": zone_50,
        "sums_series": sums,
    }


# ===================================================================
# 7. Consecutive Number Patterns
# ===================================================================

def consecutive_patterns(df: pd.DataFrame) -> dict:
    """
    How often consecutive pairs (e.g. 5-6) appear in a draw.
    How often consecutive triplets (e.g. 5-6-7) appear.
    Most common consecutive pairs.

    Returns
    -------
    dict with keys:
        draws_with_consecutive_pair    : int
        draws_with_consecutive_triplet : int
        pair_pct, triplet_pct          : float
        consecutive_pair_counts        : Counter {(a,b): count}
        top_consecutive_pairs          : list of ((a,b), count) top-15
    """
    df = _ensure_datetime(df)
    total = len(df)
    pair_draws = 0
    triplet_draws = 0
    pair_counter: Counter = Counter()

    for _, row in df.iterrows():
        nums = sorted(int(row[c]) for c in NUM_COLS)
        has_pair = False
        has_triplet = False

        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                pair_counter[(nums[i], nums[i + 1])] += 1
                has_pair = True
                # Check for triplet
                if i + 2 < len(nums) and nums[i + 2] - nums[i + 1] == 1:
                    has_triplet = True

        if has_pair:
            pair_draws += 1
        if has_triplet:
            triplet_draws += 1

    return {
        "draws_with_consecutive_pair": pair_draws,
        "draws_with_consecutive_triplet": triplet_draws,
        "pair_pct": round(100.0 * pair_draws / total, 2) if total else 0.0,
        "triplet_pct": round(100.0 * triplet_draws / total, 2) if total else 0.0,
        "consecutive_pair_counts": pair_counter,
        "top_consecutive_pairs": pair_counter.most_common(15),
        "total_draws": total,
    }


# ===================================================================
# 8. Number Gap Analysis
# ===================================================================

def number_gap_analysis(df: pd.DataFrame) -> dict:
    """
    For each number 1-49: average gap, max gap, min gap, current gap.
    Flag overdue numbers (current gap > 2x average gap).

    Returns
    -------
    dict with keys:
        gap_stats : dict {number: {avg_gap, max_gap, min_gap, current_gap, overdue}}
        overdue   : list of numbers flagged overdue
        dataframe : pd.DataFrame
    """
    df = _ensure_datetime(df)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    total = len(df_sorted)

    # Build appearance indices for each number
    appearances: dict = {n: [] for n in ALL_NUMBERS}
    for idx, row in df_sorted.iterrows():
        for col in NUM_COLS:
            appearances[int(row[col])].append(idx)

    gap_stats = {}
    overdue_list = []

    for n in ALL_NUMBERS:
        idxs = appearances[n]
        if len(idxs) < 2:
            # Not enough data
            current_gap = (total - 1 - idxs[0]) if idxs else total
            gap_stats[n] = {
                "avg_gap": float(current_gap),
                "max_gap": current_gap,
                "min_gap": current_gap,
                "current_gap": current_gap,
                "appearances": len(idxs),
                "overdue": True,
            }
            overdue_list.append(n)
            continue

        gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
        avg_gap = float(np.mean(gaps))
        current_gap = total - 1 - idxs[-1]
        is_overdue = current_gap > 2 * avg_gap

        gap_stats[n] = {
            "avg_gap": round(avg_gap, 2),
            "max_gap": int(max(gaps)),
            "min_gap": int(min(gaps)),
            "current_gap": int(current_gap),
            "appearances": len(idxs),
            "overdue": is_overdue,
        }
        if is_overdue:
            overdue_list.append(n)

    records = [{"number": n, **v} for n, v in sorted(gap_stats.items())]
    summary_df = pd.DataFrame(records)

    return {
        "gap_stats": gap_stats,
        "overdue": sorted(overdue_list),
        "dataframe": summary_df,
    }


# ===================================================================
# 9. Decade Group Spread
# ===================================================================

def decade_group_spread(df: pd.DataFrame) -> dict:
    """
    Groups: 1-9, 10-19, 20-29, 30-39, 40-49.
    Count how many numbers from each group per draw.
    Most common spread pattern. Spreads that rarely win.

    Returns
    -------
    dict with keys:
        pattern_counts : Counter {pattern_tuple: count}
        most_common    : tuple pattern
        rare_patterns  : list of patterns appearing <= 1 % of draws
        per_draw       : pd.DataFrame (draw_number, g1-g5 counts, pattern)
        group_totals   : dict {group_label: total_count_across_all_draws}
    """
    df = _ensure_datetime(df)
    total = len(df)
    group_labels = ["1-9", "10-19", "20-29", "30-39", "40-49"]

    def _group_index(n: int) -> int:
        if n <= 9:
            return 0
        elif n <= 19:
            return 1
        elif n <= 29:
            return 2
        elif n <= 39:
            return 3
        else:
            return 4

    pattern_counter: Counter = Counter()
    per_draw_records = []
    group_totals = Counter()

    for _, row in df.iterrows():
        nums = [int(row[c]) for c in NUM_COLS]
        counts = [0, 0, 0, 0, 0]
        for n in nums:
            gi = _group_index(n)
            counts[gi] += 1
            group_totals[group_labels[gi]] += 1

        pattern = tuple(counts)
        pattern_counter[pattern] += 1
        per_draw_records.append({
            "draw_number": row.get("draw_number", None),
            "g_1_9": counts[0],
            "g_10_19": counts[1],
            "g_20_29": counts[2],
            "g_30_39": counts[3],
            "g_40_49": counts[4],
            "pattern": pattern,
        })

    most_common = pattern_counter.most_common(1)[0][0] if pattern_counter else ()
    threshold = max(1, int(0.01 * total))
    rare_patterns = [p for p, c in pattern_counter.items() if c <= threshold]

    per_draw_df = pd.DataFrame(per_draw_records)

    return {
        "pattern_counts": pattern_counter,
        "most_common": most_common,
        "rare_patterns": rare_patterns,
        "per_draw": per_draw_df,
        "group_totals": dict(group_totals),
    }


# ===================================================================
# 10. Day-of-Week Analysis
# ===================================================================

def day_of_week_analysis(df: pd.DataFrame) -> dict:
    """
    Monday vs Thursday frequency distributions.
    Chi-squared test for significance of difference.

    Returns
    -------
    dict with keys:
        monday_freq, thursday_freq : dict {number: count}
        monday_draws, thursday_draws : int
        chi2_stat, chi2_pvalue       : float
        significant                  : bool (p < 0.05)
        dataframe                    : pd.DataFrame comparison
    """
    df = _ensure_datetime(df)

    # Identify day from date column (more reliable than day_of_week string)
    df_work = df.copy()
    df_work["_dow"] = df_work["date"].dt.dayofweek  # 0=Mon, 3=Thu

    mon_df = df_work[df_work["_dow"] == 0]
    thu_df = df_work[df_work["_dow"] == 3]

    # If data also has other days (shouldn't for TOTO, but be safe)
    # we only compare Mon and Thu
    mon_flat = _main_numbers_flat(mon_df)
    thu_flat = _main_numbers_flat(thu_df)

    mon_counter = Counter(mon_flat)
    thu_counter = Counter(thu_flat)

    mon_freq = {n: mon_counter.get(n, 0) for n in ALL_NUMBERS}
    thu_freq = {n: thu_counter.get(n, 0) for n in ALL_NUMBERS}

    # Chi-squared test: observed vs expected under uniform-across-days assumption
    observed_mon = np.array([mon_freq[n] for n in ALL_NUMBERS], dtype=float)
    observed_thu = np.array([thu_freq[n] for n in ALL_NUMBERS], dtype=float)

    # Combine into contingency-style: compare distributions
    # Use scipy chi2_contingency on a 2 x 49 table
    contingency = np.array([observed_mon, observed_thu])
    # Avoid zero rows / columns
    col_mask = contingency.sum(axis=0) > 0
    contingency_clean = contingency[:, col_mask]

    if contingency_clean.size > 0 and contingency_clean.sum() > 0:
        chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(contingency_clean)
    else:
        chi2_stat, chi2_p, dof = 0.0, 1.0, 0

    records = []
    for n in ALL_NUMBERS:
        records.append({
            "number": n,
            "monday_count": mon_freq[n],
            "thursday_count": thu_freq[n],
            "total": mon_freq[n] + thu_freq[n],
        })
    summary_df = pd.DataFrame(records)

    return {
        "monday_freq": mon_freq,
        "thursday_freq": thu_freq,
        "monday_draws": len(mon_df),
        "thursday_draws": len(thu_df),
        "chi2_stat": round(float(chi2_stat), 4),
        "chi2_pvalue": round(float(chi2_p), 6),
        "chi2_dof": int(dof),
        "significant": chi2_p < 0.05,
        "dataframe": summary_df,
    }


# ===================================================================
# 11. Temporal Drift
# ===================================================================

def temporal_drift(df: pd.DataFrame) -> dict:
    """
    Compare first-5-years vs last-5-years frequency.
    Monthly and quarterly trend analysis.

    Returns
    -------
    dict with keys:
        early_freq, late_freq : dict {number: count}
        drift                 : dict {number: late - early normalised delta}
        biggest_risers        : list top-10 numbers gaining frequency
        biggest_fallers       : list top-10 numbers losing frequency
        monthly_trend         : pd.DataFrame (year_month, number, count)
        quarterly_trend       : pd.DataFrame (year_quarter, number, count)
    """
    df = _ensure_datetime(df)
    df_sorted = df.sort_values("date").reset_index(drop=True)

    min_date = df_sorted["date"].min()
    max_date = df_sorted["date"].max()
    early_cutoff = min_date + pd.DateOffset(years=5)
    late_cutoff = max_date - pd.DateOffset(years=5)

    early_df = df_sorted[df_sorted["date"] < early_cutoff]
    late_df = df_sorted[df_sorted["date"] >= late_cutoff]

    early_flat = _main_numbers_flat(early_df)
    late_flat = _main_numbers_flat(late_df)

    early_counter = Counter(early_flat)
    late_counter = Counter(late_flat)

    early_total = max(len(early_df), 1)
    late_total = max(len(late_df), 1)

    early_freq = {n: early_counter.get(n, 0) for n in ALL_NUMBERS}
    late_freq = {n: late_counter.get(n, 0) for n in ALL_NUMBERS}

    # Normalised delta: (late_rate - early_rate) where rate = count / draws
    drift = {}
    for n in ALL_NUMBERS:
        early_rate = early_freq[n] / early_total
        late_rate = late_freq[n] / late_total
        drift[n] = round(late_rate - early_rate, 6)

    sorted_drift = sorted(drift.items(), key=lambda x: x[1], reverse=True)
    biggest_risers = sorted_drift[:10]
    biggest_fallers = sorted_drift[-10:][::-1]  # worst first

    # Monthly trend: count of each number per month
    df_monthly = df_sorted.copy()
    df_monthly["year_month"] = df_monthly["date"].dt.to_period("M").astype(str)
    monthly_records = []
    for ym, group in df_monthly.groupby("year_month"):
        flat = _main_numbers_flat(group)
        c = Counter(flat)
        for n in ALL_NUMBERS:
            monthly_records.append({
                "year_month": ym,
                "number": n,
                "count": c.get(n, 0),
            })
    monthly_df = pd.DataFrame(monthly_records)

    # Quarterly trend
    df_quarterly = df_sorted.copy()
    df_quarterly["year_quarter"] = df_quarterly["date"].dt.to_period("Q").astype(str)
    quarterly_records = []
    for yq, group in df_quarterly.groupby("year_quarter"):
        flat = _main_numbers_flat(group)
        c = Counter(flat)
        for n in ALL_NUMBERS:
            quarterly_records.append({
                "year_quarter": yq,
                "number": n,
                "count": c.get(n, 0),
            })
    quarterly_df = pd.DataFrame(quarterly_records)

    return {
        "early_freq": early_freq,
        "late_freq": late_freq,
        "early_draws": len(early_df),
        "late_draws": len(late_df),
        "drift": drift,
        "biggest_risers": biggest_risers,
        "biggest_fallers": biggest_fallers,
        "monthly_trend": monthly_df,
        "quarterly_trend": quarterly_df,
    }


# ===================================================================
# 12. Jackpot Correlation
# ===================================================================

def jackpot_correlation(df: pd.DataFrame) -> dict:
    """
    Compare draws with high jackpots vs normal jackpots.
    Analyses number distributions, sum ranges, odd/even for both groups.

    Returns
    -------
    dict with keys:
        has_prize_data  : bool
        high_jackpot_threshold : float
        high_draws, normal_draws : int
        high_freq, normal_freq   : dict {number: count}
        high_sum_stats, normal_sum_stats : dict
        comparison_df   : pd.DataFrame
    """
    df = _ensure_datetime(df)

    # Check if prize data is meaningful
    prize_col = "group1_prize"
    if prize_col not in df.columns:
        return {"has_prize_data": False}

    prizes = pd.to_numeric(df[prize_col], errors="coerce")
    valid_prizes = prizes.dropna()
    if len(valid_prizes) == 0 or valid_prizes.max() == 0:
        return {"has_prize_data": False}

    threshold = float(valid_prizes.quantile(0.75))
    df_work = df.copy()
    df_work["_prize_numeric"] = prizes

    high_df = df_work[df_work["_prize_numeric"] >= threshold]
    normal_df = df_work[df_work["_prize_numeric"] < threshold]

    high_flat = _main_numbers_flat(high_df)
    normal_flat = _main_numbers_flat(normal_df)

    high_counter = Counter(high_flat)
    normal_counter = Counter(normal_flat)

    high_freq = {n: high_counter.get(n, 0) for n in ALL_NUMBERS}
    normal_freq = {n: normal_counter.get(n, 0) for n in ALL_NUMBERS}

    # Sum stats for both groups
    def _sum_stats(sub_df):
        if len(sub_df) == 0:
            return {}
        s = sub_df[NUM_COLS].sum(axis=1)
        return {
            "mean": round(float(s.mean()), 2),
            "median": float(s.median()),
            "std": round(float(s.std()), 2),
        }

    high_sum = _sum_stats(high_df)
    normal_sum = _sum_stats(normal_df)

    # Comparison DataFrame
    records = []
    high_total = max(len(high_df), 1)
    normal_total = max(len(normal_df), 1)
    for n in ALL_NUMBERS:
        records.append({
            "number": n,
            "high_jackpot_count": high_freq[n],
            "high_jackpot_rate": round(high_freq[n] / high_total, 4),
            "normal_count": normal_freq[n],
            "normal_rate": round(normal_freq[n] / normal_total, 4),
        })
    comparison_df = pd.DataFrame(records)

    return {
        "has_prize_data": True,
        "high_jackpot_threshold": threshold,
        "high_draws": len(high_df),
        "normal_draws": len(normal_df),
        "high_freq": high_freq,
        "normal_freq": normal_freq,
        "high_sum_stats": high_sum,
        "normal_sum_stats": normal_sum,
        "comparison_df": comparison_df,
    }


# ===================================================================
# 13. Burst / Dormancy Detection
# ===================================================================

def burst_dormancy_detection(df: pd.DataFrame) -> dict:
    """
    For each number 1-49 classify its current state:
        BURST   - appeared 2+ times in the last 10 draws
        ACTIVE  - appeared 1 time  in the last 10 draws AND current gap < avg gap
        NORMAL  - current gap within [0.5x .. 1.5x] of average gap
        COOLING - current gap in (1.5x .. 2x] of average gap
        DORMANT - absent for >= 2x its average gap

    Returns
    -------
    dict with keys:
        classifications : dict {number: {state, recent_count, avg_gap, current_gap}}
        burst_numbers   : list
        dormant_numbers : list
        summary_df      : pd.DataFrame
    """
    df = _ensure_datetime(df)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    total = len(df_sorted)
    last_10 = df_sorted.tail(10) if total >= 10 else df_sorted

    # Count appearances in last 10 draws
    last_10_flat = _main_numbers_flat(last_10)
    recent_counter = Counter(last_10_flat)

    # Build appearance indices for average gap calculation
    appearances: dict = {n: [] for n in ALL_NUMBERS}
    for idx, row in df_sorted.iterrows():
        for col in NUM_COLS:
            appearances[int(row[col])].append(idx)

    classifications = {}
    burst_numbers = []
    dormant_numbers = []

    for n in ALL_NUMBERS:
        recent_count = recent_counter.get(n, 0)
        idxs = appearances[n]

        if len(idxs) < 2:
            avg_gap = float(total)
            current_gap = (total - 1 - idxs[0]) if idxs else total
        else:
            gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
            avg_gap = float(np.mean(gaps))
            current_gap = total - 1 - idxs[-1]

        # Classification logic
        if recent_count >= 2:
            state = "BURST"
            burst_numbers.append(n)
        elif current_gap >= 2 * avg_gap:
            state = "DORMANT"
            dormant_numbers.append(n)
        elif current_gap > 1.5 * avg_gap:
            state = "COOLING"
        elif recent_count >= 1 and current_gap <= avg_gap:
            state = "ACTIVE"
        else:
            state = "NORMAL"

        classifications[n] = {
            "state": state,
            "recent_count_last_10": recent_count,
            "avg_gap": round(avg_gap, 2),
            "current_gap": int(current_gap),
            "total_appearances": len(idxs),
        }

    records = [{"number": n, **v} for n, v in sorted(classifications.items())]
    summary_df = pd.DataFrame(records)

    return {
        "classifications": classifications,
        "burst_numbers": sorted(burst_numbers),
        "dormant_numbers": sorted(dormant_numbers),
        "summary_df": summary_df,
    }


# ===================================================================
# 14. Positional Frequency (Bonus)
# ===================================================================

def positional_frequency(df: pd.DataFrame) -> dict:
    """
    For each position (num1 through num6, sorted ascending within each draw),
    compute frequency of each number appearing in that slot.

    Returns
    -------
    dict with keys:
        position_freq : dict {position: Counter}
        dataframe     : pd.DataFrame (number, pos1_count ... pos6_count)
    """
    df = _ensure_datetime(df)
    pos_counters = {col: Counter() for col in NUM_COLS}

    for _, row in df.iterrows():
        for col in NUM_COLS:
            pos_counters[col][int(row[col])] += 1

    records = []
    for n in ALL_NUMBERS:
        rec = {"number": n}
        for col in NUM_COLS:
            rec[f"{col}_count"] = pos_counters[col].get(n, 0)
        records.append(rec)

    return {
        "position_freq": {col: dict(c) for col, c in pos_counters.items()},
        "dataframe": pd.DataFrame(records),
    }


# ===================================================================
# 15. Last-N-Draws Repeat Analysis (Bonus)
# ===================================================================

def repeat_from_previous(df: pd.DataFrame, window: int = 1) -> dict:
    """
    For each draw, count how many numbers repeated from the previous
    *window* draws.  Summarise the distribution.

    Returns
    -------
    dict with keys:
        repeat_distribution : Counter {repeat_count: frequency}
        avg_repeats         : float
        per_draw            : pd.DataFrame
    """
    df = _ensure_datetime(df)
    df_sorted = df.sort_values("date").reset_index(drop=True)

    repeat_counts = []
    repeat_dist: Counter = Counter()

    for i in range(window, len(df_sorted)):
        current_set = set(int(df_sorted.loc[i, c]) for c in NUM_COLS)
        prev_sets = set()
        for j in range(max(0, i - window), i):
            for c in NUM_COLS:
                prev_sets.add(int(df_sorted.loc[j, c]))

        repeats = len(current_set & prev_sets)
        repeat_counts.append({
            "draw_index": i,
            "draw_number": df_sorted.loc[i, "draw_number"],
            "repeats_from_prev": repeats,
        })
        repeat_dist[repeats] += 1

    per_draw_df = pd.DataFrame(repeat_counts)
    avg_repeats = per_draw_df["repeats_from_prev"].mean() if len(per_draw_df) else 0.0

    return {
        "repeat_distribution": dict(repeat_dist),
        "avg_repeats": round(float(avg_repeats), 3),
        "per_draw": per_draw_df,
    }


# ===================================================================
# Master Runner
# ===================================================================

def get_full_analysis(df: pd.DataFrame) -> dict:
    """
    Run every analysis function and return a dict of all results.

    Parameters
    ----------
    df : pd.DataFrame
        TOTO draw data matching the expected schema.

    Returns
    -------
    dict mapping analysis name -> result dict
    """
    df = _ensure_datetime(df)

    results = {}

    print("[Analysis] Running frequency analysis ...")
    results["frequency"] = frequency_analysis(df)

    print("[Analysis] Running hot/cold/overdue analysis ...")
    results["hot_cold"] = hot_cold_numbers(df)

    print("[Analysis] Running pair & triplet analysis ...")
    results["pairs_triplets"] = pair_triplet_analysis(df)

    print("[Analysis] Running odd/even distribution ...")
    results["odd_even"] = odd_even_distribution(df)

    print("[Analysis] Running high/low distribution ...")
    results["high_low"] = high_low_distribution(df)

    print("[Analysis] Running sum range analysis ...")
    results["sum_range"] = sum_range_analysis(df)

    print("[Analysis] Running consecutive patterns ...")
    results["consecutive"] = consecutive_patterns(df)

    print("[Analysis] Running number gap analysis ...")
    results["gaps"] = number_gap_analysis(df)

    print("[Analysis] Running decade group spread ...")
    results["decade_spread"] = decade_group_spread(df)

    print("[Analysis] Running day-of-week analysis ...")
    results["day_of_week"] = day_of_week_analysis(df)

    print("[Analysis] Running temporal drift analysis ...")
    results["temporal_drift"] = temporal_drift(df)

    print("[Analysis] Running jackpot correlation ...")
    results["jackpot"] = jackpot_correlation(df)

    print("[Analysis] Running burst/dormancy detection ...")
    results["burst_dormancy"] = burst_dormancy_detection(df)

    print("[Analysis] Running positional frequency ...")
    results["positional"] = positional_frequency(df)

    print("[Analysis] Running repeat-from-previous analysis ...")
    results["repeats"] = repeat_from_previous(df, window=1)

    print("[Analysis] All analyses complete.")
    return results


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    CSV_PATH = os.path.join(DATA_DIR, "toto_results.csv")

    if not os.path.exists(CSV_PATH):
        print(f"Data file not found at {CSV_PATH}")
        print("Run the scraper first:  python -m src.scraper")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])

    all_results = get_full_analysis(df)

    # Quick summary printout
    print("\n" + "=" * 60)
    print("TOTO STATISTICAL ANALYSIS SUMMARY")
    print("=" * 60)

    freq = all_results["frequency"]
    print(f"\nTotal draws analysed: {freq['total_draws']}")
    print(f"Top-5 most frequent main numbers: "
          f"{freq['ranked'][:5]}")
    print(f"Bottom-5 least frequent main numbers: "
          f"{freq['ranked'][-5:]}")

    hc = all_results["hot_cold"]
    print(f"\nHot numbers (last 3 months): "
          f"{[n for n, _ in hc['hot_3m']]}")
    print(f"Cold numbers (last 3 months): "
          f"{[n for n, _ in hc['cold_3m']]}")
    print(f"Most overdue: "
          f"{hc['overdue'][:5]}")

    sr = all_results["sum_range"]
    print(f"\nSum range stats: {sr['stats']}")
    print(f"70% zone: {sr['zone_70']}")
    print(f"50% zone: {sr['zone_50']}")

    oe = all_results["odd_even"]
    print(f"\nMost common odd/even split: {oe['most_common']}")

    hl = all_results["high_low"]
    print(f"Most common low/high split: {hl['most_common']}")

    cp = all_results["consecutive"]
    print(f"\nDraws with consecutive pair: {cp['pair_pct']}%")
    print(f"Draws with consecutive triplet: {cp['triplet_pct']}%")

    bd = all_results["burst_dormancy"]
    print(f"\nBurst numbers: {bd['burst_numbers']}")
    print(f"Dormant numbers: {bd['dormant_numbers']}")

    dow = all_results["day_of_week"]
    print(f"\nDay-of-week chi-squared p-value: {dow['chi2_pvalue']} "
          f"({'Significant' if dow['significant'] else 'Not significant'})")

    td = all_results["temporal_drift"]
    print(f"\nBiggest risers (early vs late): "
          f"{[(n, round(d, 4)) for n, d in td['biggest_risers'][:5]]}")
    print(f"Biggest fallers: "
          f"{[(n, round(d, 4)) for n, d in td['biggest_fallers'][:5]]}")

    jp = all_results["jackpot"]
    if jp.get("has_prize_data"):
        print(f"\nHigh jackpot threshold: ${jp['high_jackpot_threshold']:,.0f}")
        print(f"High jackpot draws: {jp['high_draws']}, "
              f"Normal draws: {jp['normal_draws']}")

    print("\n" + "=" * 60)
    print("Full results dict contains keys:", list(all_results.keys()))
    print("=" * 60)
