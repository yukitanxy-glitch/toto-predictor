"""
Ensemble Model for Singapore TOTO Prediction

Collects top 15 numbers from each individual model, then scores by
consensus count weighted by model performance. Ties are broken using
the weighted scoring model's rankings.

Default model weights reflect relative trustworthiness:
- WeightedScoring:  0.20 (strong baseline, interpretable)
- RandomForest:     0.20 (ML with feature engineering)
- LSTM/MLP:         0.15 (neural network patterns)
- MonteCarlo:       0.18 (probabilistic simulation)
- MarkovChain:      0.12 (sequential patterns)
- ClusterAnalysis:  0.15 (structural patterns)
"""

import traceback
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

ALL_NUMBERS = list(range(1, 50))
TOP_N_PER_MODEL = 15

# Default model weights
DEFAULT_MODEL_WEIGHTS = {
    "WeightedScoring": 0.20,
    "RandomForest": 0.20,
    "MLP_Neural_Network": 0.15,
    "LSTM": 0.15,
    "MonteCarlo": 0.18,
    "MarkovChain": 0.12,
    "ClusterAnalysis": 0.15,
}


def _run_individual_model(model_module, model_name, df):
    """
    Safely run an individual model and return its result.
    Returns None if the model fails.
    """
    try:
        print(f"\n  [Ensemble] Running {model_name}...")
        result = model_module.predict(df)
        if result and "rankings" in result and "top_numbers" in result:
            return result
        else:
            print(f"  [Ensemble] WARNING: {model_name} returned invalid result format.")
            return None
    except Exception as e:
        print(f"  [Ensemble] ERROR in {model_name}: {e}")
        traceback.print_exc()
        return None


def _collect_model_results(df):
    """
    Import and run all 6 individual models, collecting their results.
    """
    results = {}

    # Import models
    model_configs = [
        ("WeightedScoring", "weighted_scoring"),
        ("RandomForest", "random_forest"),
        ("LSTM/MLP", "lstm_model"),
        ("MonteCarlo", "monte_carlo"),
        ("MarkovChain", "markov_chain"),
        ("ClusterAnalysis", "cluster_analysis"),
    ]

    for display_name, module_name in model_configs:
        try:
            # Dynamic import from same package
            import importlib
            module = importlib.import_module(f".{module_name}", package="src.models")
            result = _run_individual_model(module, display_name, df)
            if result:
                model_key = result.get("model_name", display_name)
                results[model_key] = result
        except ImportError as e:
            print(f"  [Ensemble] Could not import {module_name}: {e}")
            # Try relative import as fallback
            try:
                import importlib
                module = importlib.import_module(f"models.{module_name}")
                result = _run_individual_model(module, display_name, df)
                if result:
                    model_key = result.get("model_name", display_name)
                    results[model_key] = result
            except ImportError:
                # Try direct import as last resort
                try:
                    import importlib
                    module = importlib.import_module(module_name)
                    result = _run_individual_model(module, display_name, df)
                    if result:
                        model_key = result.get("model_name", display_name)
                        results[model_key] = result
                except ImportError:
                    print(f"  [Ensemble] FAILED to import {module_name} by any method.")

    return results


def _compute_ensemble_scores(model_results, model_weights=None):
    """
    Compute ensemble scores for all 49 numbers based on model results.

    For each number:
    1. Count how many models have it in their top 15 (consensus count)
    2. For models that rank it, compute a weighted rank score
    3. Combine consensus count with weighted rank scores

    Parameters
    ----------
    model_results : dict of {model_name: result_dict}
    model_weights : dict of {model_name: weight}, optional

    Returns
    -------
    dict of {number: ensemble_score}
    """
    if model_weights is None:
        model_weights = DEFAULT_MODEL_WEIGHTS

    # Normalize weights to sum to 1 for active models
    active_models = set(model_results.keys())
    active_weights = {k: v for k, v in model_weights.items() if k in active_models}

    # If model names don't match exactly, try partial matching
    if not active_weights:
        for model_name in active_models:
            for weight_key, weight_val in model_weights.items():
                if weight_key.lower() in model_name.lower() or model_name.lower() in weight_key.lower():
                    active_weights[model_name] = weight_val
                    break
            if model_name not in active_weights:
                active_weights[model_name] = 1.0 / len(active_models)

    total_weight = sum(active_weights.values())
    if total_weight > 0:
        active_weights = {k: v / total_weight for k, v in active_weights.items()}

    # For each number, compute:
    # 1. Consensus score: weighted count of models that have it in top 15
    # 2. Rank score: weighted average of normalized rank positions

    consensus_scores = {n: 0.0 for n in ALL_NUMBERS}
    rank_scores = {n: 0.0 for n in ALL_NUMBERS}
    raw_scores = {n: 0.0 for n in ALL_NUMBERS}

    for model_name, result in model_results.items():
        weight = active_weights.get(model_name, 0.1)
        rankings = result.get("rankings", [])

        if not rankings:
            continue

        # Get top 15 numbers for consensus
        top_15_nums = set(num for num, _ in rankings[:TOP_N_PER_MODEL])

        # Create rank lookup (1-indexed)
        rank_lookup = {num: rank + 1 for rank, (num, _) in enumerate(rankings)}

        # Create score lookup
        score_lookup = {num: score for num, score in rankings}

        # Max score for normalization
        max_score = max(score for _, score in rankings) if rankings else 1.0
        min_score = min(score for _, score in rankings) if rankings else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        for n in ALL_NUMBERS:
            # Consensus: is this number in top 15?
            if n in top_15_nums:
                consensus_scores[n] += weight

            # Rank score: inverse of rank position, weighted
            if n in rank_lookup:
                rank = rank_lookup[n]
                # Inverse rank score: top rank (1) gets score 1.0, last rank gets ~0
                inverse_rank = 1.0 - (rank - 1) / 49.0
                rank_scores[n] += weight * inverse_rank

            # Raw score: normalized model score
            if n in score_lookup:
                normalized = (score_lookup[n] - min_score) / score_range
                raw_scores[n] += weight * normalized

    # Combine: 40% consensus, 30% rank, 30% raw score
    ensemble_scores = {}
    for n in ALL_NUMBERS:
        ensemble_scores[n] = (
            0.40 * consensus_scores[n] +
            0.30 * rank_scores[n] +
            0.30 * raw_scores[n]
        )

    return ensemble_scores, consensus_scores, rank_scores, raw_scores


def _tiebreak_with_weighted_scoring(rankings, model_results):
    """
    Break ties in ensemble rankings using the weighted scoring model.
    """
    ws_result = model_results.get("WeightedScoring", None)
    if ws_result is None:
        return rankings

    ws_rank_lookup = {}
    for rank, (num, _) in enumerate(ws_result.get("rankings", [])):
        ws_rank_lookup[num] = rank

    # Stable sort: for items with same score, prefer lower weighted scoring rank
    def sort_key(item):
        num, score = item
        ws_rank = ws_rank_lookup.get(num, 999)
        return (-score, ws_rank)

    return sorted(rankings, key=sort_key)


def predict(df, model_results=None, model_weights=None):
    """
    Run ensemble prediction combining all individual models.

    Parameters
    ----------
    df : pd.DataFrame
        Historical TOTO data.
    model_results : dict, optional
        Pre-computed results from individual models.
        If None, all models will be run.
    model_weights : dict, optional
        Custom weights for each model.
        If None, default weights are used.

    Returns
    -------
    dict with:
        'rankings': list of (number, score) sorted by score descending
        'top_numbers': list of top 6 numbers
    """
    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL")
    print("=" * 70)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Total draws: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Collect individual model results
    if model_results is None:
        print("\n  [Ensemble] Running all individual models...")
        model_results = _collect_model_results(df)
    else:
        print(f"  [Ensemble] Using {len(model_results)} pre-computed model results.")

    n_models = len(model_results)
    print(f"\n  [Ensemble] Successfully collected results from {n_models} models:")
    for name in model_results:
        top = model_results[name].get("top_numbers", [])
        print(f"    - {name}: top 6 = {top}")

    if n_models == 0:
        print("  [Ensemble] ERROR: No model results available. Returning uniform scores.")
        uniform_score = 1.0 / 49
        rankings = [(n, uniform_score) for n in ALL_NUMBERS]
        return {
            "rankings": rankings,
            "top_numbers": list(range(1, 7)),
            "model_name": "Ensemble",
            "n_models": 0,
        }

    # Compute ensemble scores
    print(f"\n  [Ensemble] Computing ensemble scores across {n_models} models...")
    ensemble_scores, consensus, rank_scores, raw_scores = _compute_ensemble_scores(
        model_results, model_weights
    )

    # Build and sort rankings
    rankings = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)

    # Tiebreak using weighted scoring
    rankings = _tiebreak_with_weighted_scoring(rankings, model_results)

    top_numbers = sorted([num for num, _ in rankings[:6]])

    # Consensus analysis
    print(f"\n  [Ensemble] Consensus Analysis:")
    for n in top_numbers:
        models_selecting = []
        for model_name, result in model_results.items():
            top_15 = [num for num, _ in result.get("rankings", [])[:TOP_N_PER_MODEL]]
            if n in top_15:
                models_selecting.append(model_name)
        print(f"    Number {n:2d}: selected by {len(models_selecting)}/{n_models} models "
              f"({', '.join(models_selecting)})")

    # Model agreement analysis
    all_top6 = []
    for result in model_results.values():
        all_top6.extend(result.get("top_numbers", []))
    top6_counts = Counter(all_top6)
    unanimous = [n for n, c in top6_counts.items() if c >= n_models]
    majority = [n for n, c in top6_counts.items() if c >= n_models / 2]

    print(f"\n  [Ensemble] Agreement Analysis:")
    print(f"    Numbers in ALL models' top 6: {sorted(unanimous) if unanimous else 'None'}")
    print(f"    Numbers in majority top 6: {sorted(majority) if majority else 'None'}")
    print(f"    Most agreed number: {top6_counts.most_common(1)[0] if top6_counts else 'N/A'}")

    print(f"\n  ENSEMBLE Top 6 numbers: {top_numbers}")
    print(f"  Top 15 rankings:")
    for i, (num, score) in enumerate(rankings[:15]):
        con = consensus.get(num, 0)
        rnk = rank_scores.get(num, 0)
        raw = raw_scores.get(num, 0)
        print(f"    {i+1:2d}. Number {num:2d} -> ensemble {score:.4f} "
              f"(consensus: {con:.3f}, rank: {rnk:.3f}, raw: {raw:.3f})")
    print("=" * 70)

    return {
        "rankings": rankings,
        "top_numbers": top_numbers,
        "model_name": "Ensemble",
        "n_models": n_models,
        "model_names": list(model_results.keys()),
        "individual_results": {
            name: {
                "top_numbers": r.get("top_numbers", []),
                "top_10": [(n, s) for n, s in r.get("rankings", [])[:10]],
            }
            for name, r in model_results.items()
        },
        "consensus_scores": consensus,
        "rank_scores": rank_scores,
        "raw_scores": raw_scores,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.scraper import load_data

    df = load_data()
    result = predict(df)

    print(f"\n{'='*70}")
    print(f"FINAL ENSEMBLE PREDICTION")
    print(f"{'='*70}")
    print(f"Top 6 numbers: {result['top_numbers']}")
    print(f"Models used: {result['n_models']}")
    for name in result.get('model_names', []):
        individual = result['individual_results'].get(name, {})
        print(f"  {name}: {individual.get('top_numbers', [])}")
    print(f"{'='*70}")
