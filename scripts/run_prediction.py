#!/usr/bin/env python3
"""
Standalone prediction script.
Runs both the legacy ensemble pipeline AND the Quant Engine v4.0.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import load_data
from src.models.weighted_scoring import predict as ws_predict
from src.models.monte_carlo import predict as mc_predict
from src.models.markov_chain import predict as mk_predict
from src.models.ensemble import predict as ensemble_predict
from src.models.quant_engine_v4 import QuantEngineV4
from src.predictor import generate_all_boards


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} draws")
    real_count = len(df[~df["is_synthetic"]]) if "is_synthetic" in df.columns else 0
    synth_count = len(df) - real_count
    print(f"Real: {real_count} | Synthetic: {synth_count}")

    # ================================================================
    # LEGACY ENSEMBLE (for comparison)
    # ================================================================
    print(f"\n{'='*70}")
    print("LEGACY ENSEMBLE MODEL")
    print(f"{'='*70}")

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

    print(f"\nNEXT DRAW: {boards['next_draw']['date']} ({boards['next_draw']['day']})")

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
    print("LEGACY ENSEMBLE RANKING - TOP 15")
    print(f"{'='*60}")
    for i, (num, score) in enumerate(ens["rankings"][:15]):
        print(f"  #{i+1:2d}. Number {num:2d} - Score: {score:.4f}")

    # ================================================================
    # QUANT ENGINE v3.0
    # ================================================================
    print(f"\n\n{'='*70}")
    print("QUANT ENGINE v4.0 - BAYESIAN + ENTROPY + MEAN REVERSION + SEQUENCE")
    print(f"{'='*70}")

    engine = QuantEngineV4(df)
    report = engine.analyze()

    # Regime
    r = report['regime']
    print(f"\n  Regime: {r['regime']} (score: {r['regime_score']})")
    print(f"    Sum trend: {r['sum_trend']}, High ratio: {r['high_ratio']}, Odd ratio: {r['odd_ratio']}")

    # Edge numbers
    print(f"\n  Bayesian Edges:")
    for num, post in report['edge_numbers'][:10]:
        label = 'HOT' if post['p_hot'] > 0.6 else 'COLD'
        print(f"    Number {num:2d}: P(hot)={post['p_hot']:.3f} ({label}), "
              f"edge={post['edge_over_fair']:+.4f}")

    # Top prediction scores (accuracy-only)
    print(f"\n  Top 15 Prediction Scores (pre-EV):")
    for i, (num, s) in enumerate(report['top_prediction']):
        print(f"    #{i+1:2d}. Number {num:2d} - Pred: {s['prediction']:.4f} "
              f"(Bay:{s['bayesian']:.2f} Mom:{s['momentum']:.2f} "
              f"Seq:{s['sequence']:.2f} Rev:{s['reversion']:.2f} "
              f"Ent:{s['entropy']:.2f})")

    # Top composite scores (prediction + anti-pop)
    print(f"\n  Top 15 Composite Scores (with EV):")
    for i, (num, s) in enumerate(report['top_composite']):
        print(f"    #{i+1:2d}. Number {num:2d} - Composite: {s['composite']:.4f} "
              f"(Pred:{s['prediction']:.3f} AntiPop:{s['anti_pop']:.2f})")

    # Generate quant boards
    qboards = engine.generate_all_boards()

    for b in qboards['boards']:
        nums_str = ', '.join(str(n) for n in b['numbers'])
        ev = b['expected_value']
        print(f"\n  Board {b['board_number']} [{b['strategy']}]:")
        print(f"    Numbers: {nums_str}")
        print(f"    Confidence: {b.get('confidence', 0):.1%}")
        print(f"    Sum: {b['validation']['sum']} | "
              f"Odd/Even: {b['validation']['odd_count']}/{6-b['validation']['odd_count']} | "
              f"Decades: {b['validation']['decades']}")
        pop_label = ('UNPOPULAR [+EV]' if ev['popularity_ratio'] < 0.8
                     else 'Average' if ev['popularity_ratio'] < 1.2
                     else 'Popular [-EV]')
        print(f"    Popularity: {ev['popularity_ratio']:.3f} ({pop_label})")
        print(f"    Expected prize if win: ${ev['expected_prize_if_win']:,.0f}")

    print(f"\n  Additional number: {qboards['additional_number']}")
    cov = qboards['coverage']
    print(f"  Coverage: {cov['unique_numbers']}/49 numbers ({cov['coverage_pct']}%), "
          f"overlap penalty: {cov['overlap_penalty']:.3f}")

    print(f"\n{'='*70}")
    print("DISCLAIMER: TOTO is a random lottery. No model guarantees wins.")
    print("Odds: 1 in 13,983,816. The Quant Engine maximizes expected")
    print("prize value (not probability). Play responsibly.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
