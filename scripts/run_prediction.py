#!/usr/bin/env python3
"""
Standalone prediction script.
Runs the full pipeline and prints 3 boards without starting the web app.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import load_data
from src.models.weighted_scoring import predict as ws_predict
from src.models.monte_carlo import predict as mc_predict
from src.models.markov_chain import predict as mk_predict
from src.models.ensemble import predict as ensemble_predict
from src.predictor import generate_all_boards


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} draws")

    print("\nRunning Weighted Scoring model...")
    ws = ws_predict(df)

    print("Running Monte Carlo simulation...")
    mc = mc_predict(df)

    print("Running Markov Chain model...")
    mk = mk_predict(df)

    print("Running Ensemble...")
    model_results = {
        "weighted_scoring": ws,
        "monte_carlo": mc,
        "markov_chain": mk,
    }
    ens = ensemble_predict(df, model_results)

    print("Generating boards...")
    boards = generate_all_boards(ens["rankings"], df)

    # Print results
    print(f"\n{'='*60}")
    print("TOTO PREDICTOR - PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {len(df)} draws")
    real_count = len(df[~df["is_synthetic"]]) if "is_synthetic" in df.columns else 0
    synth_count = len(df) - real_count
    print(f"Real: {real_count} | Synthetic: {synth_count}")
    print(f"\nNEXT DRAW: {boards['next_draw']['date']} ({boards['next_draw']['day']})")
    print(f"{'='*60}")

    for b in boards["boards"]:
        print(f"\n{b['name']}:")
        print(f"  Numbers: {', '.join(str(n) for n in b['numbers'])} + Additional: {b.get('additional_number', '?')}")
        print(f"  Confidence: {b['filter_result']['confidence']}")
        print(f"  Sum: {b['stats']['sum']} | Odd/Even: {b['stats']['odd_even']} | High/Low: {b['stats']['high_low']}")
        print(f"  Strategy: {b.get('strategy', '')}")
        print(f"  Filters:")
        for f in b["filter_result"]["results"]:
            status = "PASS" if f["passed"] else "FAIL"
            print(f"    [{status}] {f['name']}: {f['detail']}")

    # Ensemble top 15
    print(f"\n{'='*60}")
    print("ENSEMBLE RANKING - TOP 15")
    print(f"{'='*60}")
    for i, (num, score) in enumerate(ens["rankings"][:15]):
        print(f"  #{i+1:2d}. Number {num:2d} - Score: {score:.4f}")

    print(f"\n{'='*60}")
    print("DISCLAIMER: TOTO is a truly random lottery.")
    print("This model cannot guarantee wins.")
    print("Actual odds of Group 1: 1 in 13,983,816.")
    print("Play responsibly.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
