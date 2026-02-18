"""
Backtesting Engine for Singapore TOTO Predictor

Walk-forward validation: trains on past data, predicts next draw,
measures accuracy. Never uses future data.
"""
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def count_matches(predicted, actual):
    """Count how many numbers match between predicted board and actual draw."""
    return len(set(predicted) & set(actual))


def check_additional(predicted_additional, actual_additional):
    """Check if additional number matches."""
    return predicted_additional == actual_additional


def determine_prize_group(main_matches, additional_match):
    """Determine which prize group a result would win."""
    if main_matches == 6:
        return 1
    elif main_matches == 5 and additional_match:
        return 2
    elif main_matches == 5:
        return 3
    elif main_matches == 4 and additional_match:
        return 4
    elif main_matches == 4:
        return 5
    elif main_matches == 3 and additional_match:
        return 6
    elif main_matches == 3:
        return 7
    return None


def run_backtest(df, holdout_months=6, verbose=True):
    """
    Run walk-forward backtesting.

    Args:
        df: Full dataset
        holdout_months: Number of months to hold out for testing
        verbose: Print progress

    Returns:
        Dict with comprehensive backtest results
    """
    from src.models.weighted_scoring import predict as ws_predict
    from src.models.monte_carlo import predict as mc_predict
    from src.models.markov_chain import predict as mk_predict
    from src.models.ensemble import predict as ensemble_predict
    from src.predictor import generate_all_boards

    df = df.sort_values("date").reset_index(drop=True)
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_months * 30)

    test_df = df[df["date"] > cutoff_date].copy()
    if len(test_df) < 5:
        warnings.warn(f"Only {len(test_df)} test draws available. Results may be unreliable.")

    if verbose:
        print(f"\n{'='*60}")
        print("BACKTESTING ENGINE")
        print(f"{'='*60}")
        print(f"Total draws: {len(df)}")
        print(f"Training cutoff: {cutoff_date.strftime('%Y-%m-%d')}")
        print(f"Test draws: {len(test_df)}")
        print(f"{'='*60}\n")

    results = {
        "board1": [], "board2": [], "board3": [], "random": [],
        "model_scores": {"weighted": [], "monte_carlo": [], "markov": []},
    }

    for i, (idx, test_row) in enumerate(test_df.iterrows()):
        # Training data: everything before this draw
        train_df = df[df["date"] < test_row["date"]].copy()
        if len(train_df) < 50:
            continue

        actual_numbers = [int(test_row[f"num{j}"]) for j in range(1, 7)]
        actual_additional = int(test_row["additional_number"])

        if verbose and i % 10 == 0:
            print(f"  Backtesting draw {i+1}/{len(test_df)} "
                  f"(date: {test_row['date'].strftime('%Y-%m-%d')})...")

        try:
            # Run lightweight models (skip RF and LSTM for speed)
            ws_result = ws_predict(train_df)
            mc_result = mc_predict(train_df)
            mk_result = mk_predict(train_df)

            # Ensemble from available models
            model_results = {
                "weighted_scoring": ws_result,
                "monte_carlo": mc_result,
                "markov_chain": mk_result,
            }
            model_weights = {
                "weighted_scoring": 1.0,
                "monte_carlo": 1.0,
                "markov_chain": 1.0,
            }
            ens_result = ensemble_predict(train_df, model_results, model_weights)
            ensemble_rankings = ens_result["rankings"]

            # Generate boards
            boards = generate_all_boards(ensemble_rankings, train_df)

            for bi, board_key in enumerate(["board1", "board2", "board3"]):
                board = boards["boards"][bi]
                matches = count_matches(board["numbers"], actual_numbers)
                add_match = check_additional(
                    board.get("additional_number", 0), actual_additional
                )
                prize = determine_prize_group(matches, add_match)
                results[board_key].append({
                    "draw_number": test_row["draw_number"],
                    "date": test_row["date"],
                    "predicted": board["numbers"],
                    "actual": actual_numbers,
                    "matches": matches,
                    "additional_match": add_match,
                    "prize_group": prize,
                })

            # Per-model tracking
            for model_name, model_key in [
                ("weighted", "weighted_scoring"),
                ("monte_carlo", "monte_carlo"),
                ("markov", "markov_chain"),
            ]:
                top6 = model_results[model_key]["top_numbers"][:6]
                matches = count_matches(top6, actual_numbers)
                results["model_scores"][model_name].append(matches)

        except Exception as e:
            if verbose:
                print(f"    Error on draw {i+1}: {e}")
            continue

        # Random baseline
        random_board = sorted(np.random.choice(range(1, 50), 6, replace=False))
        random_matches = count_matches(random_board, actual_numbers)
        results["random"].append({
            "matches": random_matches,
        })

    # ── Aggregate metrics ────────────────────────────────────────────────
    summary = _compute_summary(results, verbose)
    _save_results(results, summary)

    return summary


def _compute_summary(results, verbose=True):
    """Compute aggregate backtest metrics."""
    summary = {}

    for board_key in ["board1", "board2", "board3"]:
        if not results[board_key]:
            summary[board_key] = {"avg_matches": 0, "distribution": {}}
            continue

        matches = [r["matches"] for r in results[board_key]]
        dist = {}
        for m in range(7):
            count = matches.count(m)
            dist[m] = {"count": count, "pct": 100 * count / len(matches) if matches else 0}

        prizes_won = [r["prize_group"] for r in results[board_key] if r["prize_group"]]
        best_match = max(matches) if matches else 0

        summary[board_key] = {
            "avg_matches": np.mean(matches) if matches else 0,
            "std_matches": np.std(matches) if matches else 0,
            "distribution": dist,
            "best_single": best_match,
            "prizes_won": len(prizes_won),
            "total_draws": len(matches),
        }

    # Random baseline
    random_matches = [r["matches"] for r in results["random"]]
    summary["random"] = {
        "avg_matches": np.mean(random_matches) if random_matches else 0,
        "std_matches": np.std(random_matches) if random_matches else 0,
    }

    # Per-model breakdown
    summary["model_breakdown"] = {}
    best_model = None
    best_avg = 0
    for model_name, scores in results["model_scores"].items():
        if scores:
            avg = np.mean(scores)
            summary["model_breakdown"][model_name] = {
                "avg_matches": avg,
                "total_predictions": len(scores),
            }
            if avg > best_avg:
                best_avg = avg
                best_model = model_name
    summary["best_model"] = best_model

    # Statistical significance: t-test Board 1 vs Random
    if results["board1"] and results["random"]:
        b1_matches = [r["matches"] for r in results["board1"]]
        rand_matches = [r["matches"] for r in results["random"]]
        if len(b1_matches) > 1 and len(rand_matches) > 1:
            t_stat, p_value = stats.ttest_ind(b1_matches, rand_matches)
            summary["significance"] = {
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "significant_at_005": p_value < 0.05,
                "significant_at_010": p_value < 0.10,
            }
            # Confidence interval for the difference
            diff = np.mean(b1_matches) - np.mean(rand_matches)
            se = np.sqrt(np.var(b1_matches)/len(b1_matches) + np.var(rand_matches)/len(rand_matches))
            ci_low = diff - 1.96 * se
            ci_high = diff + 1.96 * se
            summary["significance"]["mean_diff"] = round(diff, 4)
            summary["significance"]["ci_95"] = (round(ci_low, 4), round(ci_high, 4))

    if verbose:
        _print_summary(summary)

    return summary


def _print_summary(summary):
    """Print a formatted backtest report."""
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS SUMMARY")
    print(f"{'='*60}")

    for bk in ["board1", "board2", "board3"]:
        s = summary.get(bk, {})
        if not s:
            continue
        print(f"\n{bk.upper()}:")
        print(f"  Average matches: {s.get('avg_matches', 0):.3f} / 6")
        print(f"  Best single draw: {s.get('best_single', 0)} matches")
        print(f"  Prizes won: {s.get('prizes_won', 0)} / {s.get('total_draws', 0)} draws")
        dist = s.get("distribution", {})
        for m in range(7):
            d = dist.get(m, {})
            print(f"    {m} matches: {d.get('count', 0)} ({d.get('pct', 0):.1f}%)")

    rs = summary.get("random", {})
    print(f"\nRANDOM BASELINE:")
    print(f"  Average matches: {rs.get('avg_matches', 0):.3f} / 6")

    mb = summary.get("model_breakdown", {})
    if mb:
        print(f"\nPER-MODEL BREAKDOWN:")
        for name, data in mb.items():
            print(f"  {name}: avg {data['avg_matches']:.3f} matches")
        print(f"  Best model: {summary.get('best_model', 'N/A')}")

    sig = summary.get("significance", {})
    if sig:
        print(f"\nSTATISTICAL SIGNIFICANCE (Board 1 vs Random):")
        print(f"  t-statistic: {sig.get('t_statistic', 'N/A')}")
        print(f"  p-value: {sig.get('p_value', 'N/A')}")
        print(f"  Mean difference: {sig.get('mean_diff', 'N/A')}")
        ci = sig.get("ci_95", ("N/A", "N/A"))
        print(f"  95% CI: ({ci[0]}, {ci[1]})")
        if sig.get("significant_at_005"):
            print("  ✓ Significant at p < 0.05")
        elif sig.get("significant_at_010"):
            print("  ~ Marginally significant at p < 0.10")
        else:
            print("  ✗ Not statistically significant")

    print(f"\n{'='*60}")


def _save_results(results, summary):
    """Save backtest results to CSV."""
    rows = []
    for bk in ["board1", "board2", "board3"]:
        for r in results[bk]:
            rows.append({
                "board": bk,
                "draw_number": r["draw_number"],
                "date": r["date"],
                "predicted": str(r["predicted"]),
                "actual": str(r["actual"]),
                "matches": r["matches"],
                "additional_match": r["additional_match"],
                "prize_group": r["prize_group"],
            })

    if rows:
        bt_df = pd.DataFrame(rows)
        path = os.path.join(DATA_DIR, "backtest_results.csv")
        os.makedirs(DATA_DIR, exist_ok=True)
        bt_df.to_csv(path, index=False)
        print(f"\nBacktest results saved to {path}")


if __name__ == "__main__":
    from src.scraper import load_data
    df = load_data()
    run_backtest(df, holdout_months=6, verbose=True)
