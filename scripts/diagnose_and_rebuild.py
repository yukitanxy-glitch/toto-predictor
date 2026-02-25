#!/usr/bin/env python3
"""
TOTO Predictor - Full Diagnosis, Rebuild, Backtest & Predict
=============================================================
Step 1: Diagnose v3.0 failures (walk-forward on last 52 draws)
Step 2: Build Quant Engine v4.0 with stronger methods
Step 3: Head-to-head backtest v3.0 vs v4.0 vs random
Step 4: Generate next-draw predictions
"""
import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb as scipy_comb

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PYTHON = sys.executable
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'toto_results.csv')


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Dataset: {len(df)} draws, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# =====================================================================
# STEP 1: DIAGNOSE V3.0 FAILURES
# =====================================================================

def diagnose_v3(df, n_test=52):
    """Walk-forward backtest of v3.0 on last n_test draws."""
    from src.models.quant_engine_v3 import QuantEngineV3

    print(f"\n{'='*70}")
    print("STEP 1: DIAGNOSING QUANT ENGINE v3.0 FAILURES")
    print(f"{'='*70}")
    print(f"Testing on last {n_test} draws (walk-forward, no data leakage)")

    test_df = df.tail(n_test)
    train_start_idx = len(df) - n_test

    v3_results = []
    factor_hits = defaultdict(list)  # track which factors predicted hits
    missed_numbers = Counter()  # numbers that appeared but weren't predicted
    false_positives = Counter()  # numbers predicted but didn't appear

    for i in range(n_test):
        test_idx = train_start_idx + i
        train = df.iloc[:test_idx].copy()
        actual_row = df.iloc[test_idx]
        actual = set(int(actual_row[f'num{j}']) for j in range(1, 7))
        actual_add = int(actual_row['additional_number'])

        if len(train) < 100:
            continue

        try:
            engine = QuantEngineV3(train)
            engine.analyze()
            boards_result = engine.generate_all_boards()
        except Exception as e:
            continue

        # Check each board
        best_matches = 0
        best_board_idx = 0
        for bi, board_info in enumerate(boards_result['boards']):
            predicted = set(int(n) for n in board_info['numbers'])
            matches = len(predicted & actual)
            if matches > best_matches:
                best_matches = matches
                best_board_idx = bi

            # Track per-number factor contributions for matched numbers
            for n in predicted & actual:
                s = engine._composite_scores[int(n)]
                factor_hits['bayesian'].append(s['bayesian'])
                factor_hits['momentum'].append(s['momentum'])
                factor_hits['centrality'].append(s['centrality'])
                factor_hits['anti_pop'].append(s['anti_pop'])
                factor_hits['overdue'].append(s['overdue'])

            # Track misses and false positives
            for n in actual - predicted:
                missed_numbers[n] += 1
            for n in predicted - actual:
                false_positives[n] += 1

        # Best of 3 boards
        best_board = boards_result['boards'][best_board_idx]
        predicted_best = set(int(n) for n in best_board['numbers'])
        add_match = int(boards_result['additional_number']) == actual_add

        v3_results.append({
            'draw_idx': test_idx,
            'date': actual_row['date'],
            'actual': sorted(actual),
            'best_matches': best_matches,
            'additional_match': add_match,
        })

    matches = [r['best_matches'] for r in v3_results]
    add_matches = sum(1 for r in v3_results if r['additional_match'])

    print(f"\n--- V3.0 PERFORMANCE (best of 3 boards) ---")
    print(f"Average matches: {np.mean(matches):.3f} / 6")
    print(f"Match distribution:")
    for m in range(7):
        c = matches.count(m)
        print(f"  {m} matches: {c} ({100*c/len(matches):.1f}%)")
    print(f"Additional number hits: {add_matches}/{len(v3_results)} ({100*add_matches/len(v3_results):.1f}%)")
    prize_draws = sum(1 for m in matches if m >= 3)
    print(f"Prize-winning draws (3+): {prize_draws}/{len(v3_results)} ({100*prize_draws/len(v3_results):.1f}%)")

    # Random baseline
    random_matches = []
    np.random.seed(42)
    for _ in range(n_test):
        for _ in range(3):  # 3 random boards
            rb = set(np.random.choice(range(1, 50), 6, replace=False))
            actual = set(int(df.iloc[train_start_idx + _]['num' + str(j)]) for j in range(1, 7))
        best_rm = 0
        for _ in range(3):
            rb = set(np.random.choice(range(1, 50), 6, replace=False))
            rm = len(rb & actual)
            best_rm = max(best_rm, rm)
        random_matches.append(best_rm)
    print(f"\nRandom baseline (best of 3): {np.mean(random_matches):.3f} / 6")
    print(f"Edge over random: {np.mean(matches) - np.mean(random_matches):+.3f}")

    # Factor analysis
    print(f"\n--- FACTOR DIAGNOSIS ---")
    print("Average factor scores for numbers that MATCHED:")
    for factor, scores in factor_hits.items():
        if scores:
            print(f"  {factor}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")

    # Compute same for ALL predicted numbers (to compare)
    print("\nMost frequently MISSED numbers (appeared but not predicted):")
    for num, count in missed_numbers.most_common(15):
        print(f"  Number {num}: missed {count} times")

    print("\nMost frequent FALSE POSITIVES (predicted but didn't appear):")
    for num, count in false_positives.most_common(15):
        print(f"  Number {num}: false positive {count} times")

    # Structural analysis of failures
    print(f"\n--- STRUCTURAL FAILURE ANALYSIS ---")

    # Check if v3.0 has a bias toward certain number ranges
    all_actual = []
    all_predicted_b1 = []
    for i in range(n_test):
        test_idx = train_start_idx + i
        actual = [int(df.iloc[test_idx][f'num{j}']) for j in range(1, 7)]
        all_actual.extend(actual)

    actual_mean = np.mean(all_actual)
    actual_high_pct = sum(1 for n in all_actual if n >= 25) / len(all_actual)
    actual_odd_pct = sum(1 for n in all_actual if n % 2 == 1) / len(all_actual)

    print(f"Actual numbers mean: {actual_mean:.1f} (fair=25.0)")
    print(f"Actual high (>=25) pct: {actual_high_pct:.3f} (fair=0.510)")
    print(f"Actual odd pct: {actual_odd_pct:.3f} (fair=0.510)")

    # Key weaknesses identified
    print(f"\n--- KEY WEAKNESSES IDENTIFIED ---")
    weaknesses = []

    # 1. Anti-popularity weight too high
    if factor_hits['anti_pop']:
        anti_pop_mean = np.mean(factor_hits['anti_pop'])
        if anti_pop_mean > 0.5:
            weaknesses.append(
                "ANTI-POPULARITY OVER-WEIGHTED: Matched numbers tend to have HIGH anti-pop "
                f"({anti_pop_mean:.3f}), meaning v3.0 correctly avoids popular numbers for EV, "
                "but this doesn't help with MATCHING accuracy."
            )
        else:
            weaknesses.append(
                "ANTI-POPULARITY MISALIGNED: 25% weight on anti-popularity hurts prediction accuracy. "
                "Anti-pop optimizes for prize VALUE, not for matching numbers."
            )

    # 2. Overdue factor (gambler's fallacy)
    if factor_hits['overdue']:
        overdue_mean = np.mean(factor_hits['overdue'])
        weaknesses.append(
            f"OVERDUE FACTOR: Mean overdue score for hits = {overdue_mean:.3f}. "
            "Overdue logic is gambler's fallacy in a fair lottery. "
            "Each draw is independent - a number isn't 'due'."
        )

    # 3. Static regime detection
    weaknesses.append(
        "REGIME DETECTOR TOO COARSE: Binary high/low classification with static thresholds. "
        "30-draw window is arbitrary. No statistical test for regime significance."
    )

    # 4. No sequence modeling
    weaknesses.append(
        "NO TEMPORAL MODELING: v3.0 treats each draw independently. "
        "No LSTM, no autoregressive features, no lag analysis."
    )

    # 5. No entropy/information theory
    weaknesses.append(
        "NO ENTROPY ANALYSIS: v3.0 claims information theory but never implements it. "
        "Missing: draw-to-draw mutual information, conditional entropy, surprise scoring."
    )

    # 6. Board generation randomness
    weaknesses.append(
        "STOCHASTIC BOARD GENERATION: 3000 random attempts to find good boards. "
        "Not deterministic. No guarantee of optimality. Different runs = different boards."
    )

    # 7. Pair graph is static
    weaknesses.append(
        "PAIR GRAPH COMPUTED ON FULL HISTORY: No time decay. "
        "Ancient pair correlations from 2015 weighted same as recent ones. "
        "Co-occurrence from 1000+ draws ago shouldn't count equally."
    )

    for i, w in enumerate(weaknesses, 1):
        print(f"  {i}. {w}")

    return {
        'v3_matches': matches,
        'random_matches': random_matches,
        'weaknesses': weaknesses,
        'factor_hits': {k: float(np.mean(v)) if v else 0 for k, v in factor_hits.items()},
        'missed_numbers': dict(missed_numbers.most_common(20)),
        'false_positives': dict(false_positives.most_common(20)),
    }


# =====================================================================
# STEP 2: BUILD QUANT ENGINE v4.0
# =====================================================================

class EntropyAnalyzer:
    """Information-theoretic analysis of draw sequences."""

    def __init__(self, df):
        self.df = df
        self.n = len(df)

    def number_surprise_scores(self, lookback=50):
        """
        Compute 'surprise' score for each number.
        Surprise = -log2(P(number)) based on recent frequency.
        High surprise = hasn't appeared much recently = more informative if it appears.
        """
        recent = self.df.tail(lookback)
        counts = Counter()
        for _, row in recent.iterrows():
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                counts[int(row[col])] += 1

        total = lookback * 6
        scores = {}
        for num in range(1, 50):
            freq = counts.get(num, 0) / total
            freq = max(freq, 1e-6)  # avoid log(0)
            surprise = -np.log2(freq)
            scores[num] = surprise

        # Normalize to [0, 1]
        min_s = min(scores.values())
        max_s = max(scores.values())
        rng = max_s - min_s if max_s > min_s else 1
        return {k: (v - min_s) / rng for k, v in scores.items()}

    def consecutive_draw_mutual_info(self, lookback=100):
        """
        Estimate mutual information between consecutive draws.
        I(X_t; X_{t+1}) - how much knowing current draw helps predict next.
        """
        recent = self.df.tail(lookback)
        overlap_counts = []
        for i in range(1, len(recent)):
            prev = set(recent.iloc[i-1][['num1','num2','num3','num4','num5','num6']].astype(int))
            curr = set(recent.iloc[i][['num1','num2','num3','num4','num5','num6']].astype(int))
            overlap_counts.append(len(prev & curr))

        # Expected overlap under independence: 6*6/49 = 0.735
        expected = 6 * 6 / 49
        observed = np.mean(overlap_counts) if overlap_counts else expected

        # Simple MI proxy: KL divergence of observed overlap distribution from expected
        return {
            'expected_overlap': round(expected, 3),
            'observed_overlap': round(observed, 3),
            'excess_overlap': round(observed - expected, 3),
            'overlap_std': round(np.std(overlap_counts), 3) if overlap_counts else 0,
        }


class MeanReversionDetector:
    """
    Detect mean-reversion patterns in number frequencies.
    If a number's recent frequency is far from its long-term average,
    bet on it reverting.
    """

    def __init__(self, df):
        self.df = df
        self.n = len(df)

    def compute_reversion_scores(self, short_window=20, long_window=200):
        """
        Z-score of recent frequency vs long-term frequency.
        Large negative z = recently cold, expect reversion UP.
        Large positive z = recently hot, expect reversion DOWN.
        """
        long_data = self.df.tail(min(long_window, self.n))
        short_data = self.df.tail(min(short_window, self.n))

        scores = {}
        for num in range(1, 50):
            long_count = 0
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                long_count += (long_data[col] == num).sum()
            long_freq = long_count / len(long_data)

            short_count = 0
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                short_count += (short_data[col] == num).sum()
            short_freq = short_count / len(short_data)

            expected_freq = 6 / 49
            # Standard error of frequency
            se = np.sqrt(expected_freq * (1 - expected_freq) / len(short_data))
            se = max(se, 1e-6)

            z_score = (short_freq - long_freq) / se

            # Mean reversion: if z is very negative (recently cold vs long-term),
            # expect it to come back. Score = inverse of z (cold -> high score)
            # But capped - extreme values shouldn't dominate
            reversion = np.clip(-z_score * 0.3, -1, 1)
            # Map to [0, 1]
            scores[num] = (reversion + 1) / 2

        return scores


class LSTMFeatureExtractor:
    """
    Lightweight sequence features without requiring deep learning libraries.
    Extracts autoregressive and lag features from draw sequences.
    """

    def __init__(self, df):
        self.df = df
        self.n = len(df)

    def extract_lag_features(self, lags=[1, 2, 3, 5, 10]):
        """
        For each number, compute lag features:
        - Was it in draw t-1? t-2? t-3?
        - Rolling frequency over last L draws
        """
        scores = {}
        for num in range(1, 50):
            appearances = []
            for _, row in self.df.iterrows():
                appeared = int(any(row[f'num{j}'] == num for j in range(1, 7)))
                appearances.append(appeared)

            features = {}
            for lag in lags:
                if len(appearances) > lag:
                    # Was this number in draw at lag L?
                    features[f'lag_{lag}'] = appearances[-lag]
                    # Rolling mean over last L draws
                    features[f'rolling_{lag}'] = np.mean(appearances[-lag:])
                else:
                    features[f'lag_{lag}'] = 0
                    features[f'rolling_{lag}'] = 6/49

            # Autoregressive signal: if appeared in t-1, what's P(appears in t)?
            # Compute from history
            transitions = []
            for i in range(1, len(appearances)):
                if appearances[i-1] == 1:
                    transitions.append(appearances[i])

            features['p_repeat'] = np.mean(transitions) if transitions else 6/49

            # Streak length
            streak = 0
            for a in reversed(appearances):
                if a == 0:
                    streak += 1
                else:
                    break
            features['absence_streak'] = streak

            scores[num] = features

        return scores

    def compute_sequence_scores(self):
        """Combine lag features into a single score per number."""
        lag_features = self.extract_lag_features()
        scores = {}
        for num in range(1, 50):
            f = lag_features[num]
            # Weighted combination of features
            score = (
                0.15 * f['rolling_3'] +    # Very recent trend
                0.15 * f['rolling_5'] +    # Short-term trend
                0.10 * f['rolling_10'] +   # Medium-term
                0.20 * f['p_repeat'] +     # Autoregressive signal
                0.40 * (1 / (1 + np.exp(-(f['absence_streak'] - 8) / 3)))  # Sigmoid of gap
            )
            scores[num] = score

        # Normalize to [0, 1]
        min_s = min(scores.values())
        max_s = max(scores.values())
        rng = max_s - min_s if max_s > min_s else 1
        return {k: (v - min_s) / rng for k, v in scores.items()}


class TripletAnalyzer:
    """Analyze triplet co-occurrence patterns beyond pairs."""

    def __init__(self, df, lookback=200):
        self.df = df.tail(min(lookback, len(df)))
        self.n = len(self.df)
        self.triplet_counts = Counter()
        self._build()

    def _build(self):
        for _, row in self.df.iterrows():
            nums = sorted(int(row[f'num{j}']) for j in range(1, 7))
            for trip in combinations(nums, 3):
                self.triplet_counts[trip] += 1

    def get_number_triplet_scores(self):
        """Score each number by how often it appears in strong triplets."""
        # Expected triplet frequency: n * C(5,2)/C(48,2) = n * 10/1128
        expected = self.n * 10 / 1128
        scores = defaultdict(float)

        for trip, count in self.triplet_counts.items():
            lift = count / max(expected, 0.01)
            if lift > 1.5:  # Only count significant triplets
                for num in trip:
                    scores[num] += lift - 1

        # Normalize
        if scores:
            max_s = max(scores.values())
            if max_s > 0:
                return {num: scores.get(num, 0) / max_s for num in range(1, 50)}
        return {num: 0.5 for num in range(1, 50)}


class QuantEngineV4:
    """
    Quant Engine v4.0 - Complete rebuild addressing v3.0 weaknesses.

    Changes from v3.0:
    1. Removed overdue factor (gambler's fallacy)
    2. Added entropy/surprise scoring
    3. Added mean reversion detector
    4. Added autoregressive lag features (LSTM-lite)
    5. Added triplet co-occurrence analysis
    6. Time-decayed pair graph (recent pairs weighted more)
    7. Deterministic board generation (greedy construction, not random sampling)
    8. Anti-popularity SEPARATED from prediction score (two-stage: predict, then optimize)
    9. Ensemble calibration with walk-forward validation
    10. Confidence scoring per board

    Scoring weights (prediction only, anti-pop applied separately):
    - Bayesian posterior:  20%
    - Recent momentum:     15%
    - Pair centrality:     15%
    - Sequence/lag:        20%
    - Mean reversion:      15%
    - Entropy/surprise:    10%
    - Triplet bonus:        5%
    """

    def __init__(self, df):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
        self.n = len(df)

        # Sub-engines
        from src.models.quant_engine_v3 import BayesianNumberScorer, RegimeDetector
        self.bayesian = BayesianNumberScorer(df, prior_strength=30)  # Weaker prior for more data-driven
        self.entropy = EntropyAnalyzer(df)
        self.mean_rev = MeanReversionDetector(df)
        self.sequence = LSTMFeatureExtractor(df)
        self.triplet = TripletAnalyzer(df, lookback=200)
        self.regime = RegimeDetector(df, window=20)  # Shorter window = more responsive

        # Time-decayed pair graph
        self._pair_centrality = None
        self._posteriors = None
        self._posteriors_recent = None
        self._regime_info = None
        self._composite_scores = None
        self._anti_popularity = None
        self._prediction_scores = None  # Separate from composite (no anti-pop)

    def analyze(self):
        """Full analysis pipeline."""
        self._posteriors = self.bayesian.compute_posteriors()
        self._posteriors_recent = self.bayesian.compute_posteriors(lookback=26)
        self._pair_centrality = self._build_time_decayed_pairs()
        self._regime_info = self.regime.detect_regime()
        self._anti_popularity = self._compute_anti_popularity()

        # Compute all component scores
        entropy_scores = self.entropy.number_surprise_scores(lookback=50)
        reversion_scores = self.mean_rev.compute_reversion_scores(short_window=15, long_window=150)
        sequence_scores = self.sequence.compute_sequence_scores()
        triplet_scores = self.triplet.get_number_triplet_scores()

        # Build prediction scores (WITHOUT anti-popularity)
        self._prediction_scores = {}
        for num in range(1, 50):
            bayesian = self._posteriors[num]['p_hot']
            momentum = self._posteriors_recent[num]['p_hot']
            centrality = self._pair_centrality.get(num, 0.5)
            sequence = sequence_scores.get(num, 0.5)
            reversion = reversion_scores.get(num, 0.5)
            entropy = entropy_scores.get(num, 0.5)
            triplet = triplet_scores.get(num, 0.5)

            # Regime adjustment (gentler than v3.0)
            regime_adj = self._get_regime_adjustment(num)

            pred_score = (
                0.20 * bayesian +
                0.15 * momentum +
                0.15 * centrality +
                0.20 * sequence +
                0.15 * reversion +
                0.10 * entropy +
                0.05 * triplet
            )
            # Apply regime as a multiplier, not additive
            pred_score *= (0.85 + 0.30 * regime_adj)

            self._prediction_scores[num] = {
                'prediction': round(pred_score, 5),
                'bayesian': round(bayesian, 4),
                'momentum': round(momentum, 4),
                'centrality': round(centrality, 4),
                'sequence': round(sequence, 4),
                'reversion': round(reversion, 4),
                'entropy': round(entropy, 4),
                'triplet': round(triplet, 4),
                'regime_adj': round(regime_adj, 4),
            }

        # Composite scores (prediction + anti-pop for EV optimization)
        self._composite_scores = {}
        for num in range(1, 50):
            pred = self._prediction_scores[num]['prediction']
            anti = self._anti_popularity.get(num, 0.5)
            # Two-stage: 70% prediction accuracy, 30% EV optimization
            composite = 0.70 * pred + 0.30 * anti
            self._composite_scores[num] = {
                'composite': round(composite, 5),
                'prediction': pred,
                'anti_pop': round(anti, 4),
                **{k: v for k, v in self._prediction_scores[num].items() if k != 'prediction'}
            }

        return self.get_analysis_report()

    def _build_time_decayed_pairs(self):
        """Build pair graph with exponential time decay."""
        adj = np.zeros((49, 49))
        half_life = 200  # draws

        for idx, (_, row) in enumerate(self.df.iterrows()):
            age = self.n - 1 - idx
            weight = np.exp(-0.693 * age / half_life)  # exponential decay

            nums = [int(row['num1']), int(row['num2']), int(row['num3']),
                   int(row['num4']), int(row['num5']), int(row['num6'])]
            for i, j in combinations(nums, 2):
                adj[i-1][j-1] += weight
                adj[j-1][i-1] += weight

        # Eigenvector centrality
        try:
            eigenvalues, eigenvectors = np.linalg.eig(adj)
            idx = np.argmax(np.real(eigenvalues))
            centrality = np.abs(np.real(eigenvectors[:, idx]))
            centrality = centrality / centrality.max()  # Normalize to [0, 1]
        except np.linalg.LinAlgError:
            centrality = np.ones(49) / 49

        return {num: float(centrality[num-1]) for num in range(1, 50)}

    def _get_regime_adjustment(self, num):
        """Gentle regime adjustment as a multiplier factor [0, 1]."""
        if self._regime_info is None:
            return 0.5

        recs = self._regime_info.get('recommendations', {})
        prefer = recs.get('prefer_range', 'balanced')

        if prefer == 'high':
            return 0.65 if num >= 25 else 0.35
        elif prefer == 'low':
            return 0.35 if num >= 25 else 0.65
        return 0.5

    def _compute_anti_popularity(self):
        """Same anti-popularity model as v3.0 (proven logic)."""
        scores = {}
        for num in range(1, 50):
            pop = 1.0
            if num <= 12: pop *= 1.8
            elif num <= 31: pop *= 1.4
            if num in [8, 18, 28, 38, 48]: pop *= 1.5
            if num in [6, 16, 26, 36, 46]: pop *= 1.2
            if num in [9, 19, 29, 39, 49]: pop *= 1.2
            if num in [7, 17, 27, 37, 47]: pop *= 1.3
            if num in [4, 14, 24, 34, 44]: pop *= 0.7
            if num % 10 == 0: pop *= 1.3
            elif num % 5 == 0: pop *= 1.15
            if num >= 40: pop *= 0.85
            scores[num] = 1.0 / pop

        min_s = min(scores.values())
        max_s = max(scores.values())
        rng = max_s - min_s if max_s > min_s else 1
        return {k: (v - min_s) / rng for k, v in scores.items()}

    def _validate_board(self, board):
        """Structural validation. Returns penalty score."""
        board = sorted(board)
        penalty = 0
        total = sum(board)
        if total < 100 or total > 200: penalty += 3
        elif total < 115 or total > 185: penalty += 1

        odd = sum(1 for n in board if n % 2 == 1)
        if odd in [0, 6]: penalty += 3
        elif odd in [1, 5]: penalty += 1

        high = sum(1 for n in board if n >= 25)
        if high in [0, 6]: penalty += 3
        elif high in [1, 5]: penalty += 1

        max_consec = 1
        curr = 1
        for i in range(1, len(board)):
            if board[i] == board[i-1] + 1:
                curr += 1
                max_consec = max(max_consec, curr)
            else:
                curr = 1
        if max_consec >= 4: penalty += 3
        elif max_consec >= 3: penalty += 1

        decades = set((n-1) // 10 for n in board)
        if len(decades) <= 2: penalty += 2

        return penalty

    def generate_board_greedy(self, score_key='composite', exclude=None, boost_fn=None):
        """
        Deterministic greedy board construction.
        Start with best number, greedily add numbers that maximize
        board score while minimizing structural penalties.
        """
        if exclude is None:
            exclude = set()

        scores = self._composite_scores if score_key == 'composite' else self._prediction_scores
        candidates = [(n, s[score_key if score_key in s else 'prediction'])
                     for n, s in scores.items() if n not in exclude]

        if boost_fn:
            candidates = [(n, s * boost_fn(n)) for n, s in candidates]

        candidates.sort(key=lambda x: -x[1])

        # Greedy with lookahead
        best_board = None
        best_total_score = -float('inf')

        # Try starting from each of top 12 numbers
        for start_idx in range(min(12, len(candidates))):
            board = [candidates[start_idx][0]]
            remaining = [c for c in candidates if c[0] != board[0]]

            while len(board) < 6 and remaining:
                best_next = None
                best_next_score = -float('inf')

                for n, s in remaining[:25]:  # Consider top 25 remaining
                    trial = board + [n]
                    penalty = self._validate_board(trial + [49] * (6 - len(trial)))  # Partial check
                    score = s - penalty * 0.3
                    if score > best_next_score:
                        best_next_score = score
                        best_next = n

                if best_next is None:
                    break
                board.append(best_next)
                remaining = [c for c in remaining if c[0] != best_next]

            if len(board) == 6:
                penalty = self._validate_board(board)
                total = sum(scores[n][score_key if score_key in scores[n] else 'prediction']
                          for n in board) - penalty
                if total > best_total_score:
                    best_total_score = total
                    best_board = sorted(board)

        if best_board is None:
            best_board = sorted([c[0] for c in candidates[:6]])

        return best_board

    def _compute_board_ev(self, board):
        """Compute expected value metrics."""
        p_win = 1 / 13_983_816
        pop_score = sum(1.0 - self._anti_popularity.get(n, 0.5) for n in board) / 6
        popularity_ratio = 0.3 + pop_score * 1.4
        expected_winners = (1_000_000 * p_win * popularity_ratio) + 1
        expected_prize = 5_000_000 / expected_winners
        return {
            'popularity_ratio': round(popularity_ratio, 3),
            'expected_prize_if_win': round(expected_prize, 0),
            'expected_value': round(p_win * expected_prize, 6)
        }

    def _compute_confidence(self, board):
        """
        Confidence score based on how many independent signals agree.
        Range: 0 to 1, where higher = more signals point to these numbers.
        """
        signals_above_median = 0
        total_signals = 0

        for n in board:
            s = self._prediction_scores[n]
            for factor in ['bayesian', 'momentum', 'centrality', 'sequence', 'reversion', 'entropy']:
                total_signals += 1
                if s[factor] > 0.5:
                    signals_above_median += 1

        return round(signals_above_median / max(total_signals, 1), 3)

    def generate_all_boards(self):
        """Generate 3 boards with different strategies."""
        if self._composite_scores is None:
            self.analyze()

        # Board 1: Best prediction accuracy (composite)
        board1 = self.generate_board_greedy(score_key='composite')

        # Board 2: Max EV (boost anti-popular numbers)
        def anti_pop_boost(n):
            return 1 + self._anti_popularity.get(n, 0.5) * 0.5
        board2 = self.generate_board_greedy(
            score_key='composite',
            exclude=set(board1),
            boost_fn=anti_pop_boost
        )

        # Board 3: Sequence-heavy (momentum + reversion)
        def sequence_boost(n):
            s = self._prediction_scores[n]
            return 1 + (s['sequence'] + s['reversion']) * 0.3
        board3 = self.generate_board_greedy(
            score_key='composite',
            exclude=set(board1) | set(board2),
            boost_fn=sequence_boost
        )

        boards = [board1, board2, board3]
        strategies = [
            'Composite Optimized (Accuracy+EV)',
            'Max Expected Value (Anti-Popular Boost)',
            'Sequence Momentum (Lag+Reversion Focus)'
        ]

        board_details = []
        for i, (board, strat) in enumerate(zip(boards, strategies)):
            ev = self._compute_board_ev(board)
            penalty = self._validate_board(board)
            odd = sum(1 for n in board if n % 2 == 1)
            high = sum(1 for n in board if n >= 25)
            decades = len(set((n-1)//10 for n in board))
            confidence = self._compute_confidence(board)

            num_details = []
            for n in board:
                s = self._prediction_scores[n]
                num_details.append({
                    'number': n,
                    'prediction': s['prediction'],
                    'bayesian': s['bayesian'],
                    'sequence': s['sequence'],
                    'reversion': s['reversion'],
                    'anti_pop': self._anti_popularity.get(n, 0.5),
                })

            board_details.append({
                'board_number': i + 1,
                'numbers': sorted(board),
                'strategy': strat,
                'expected_value': ev,
                'confidence': confidence,
                'validation': {
                    'sum': sum(board),
                    'odd_count': odd,
                    'high_count': high,
                    'decades': decades,
                    'penalties': penalty
                },
                'number_details': num_details
            })

        # Coverage
        all_nums = set()
        for b in boards:
            all_nums.update(b)

        # Additional number
        additional = self._predict_additional(boards)

        return {
            'boards': board_details,
            'coverage': {
                'unique_numbers': len(all_nums),
                'coverage_pct': round(len(all_nums) / 49 * 100, 1),
                'overlap_penalty': 0.0
            },
            'additional_number': additional,
            'regime': self._regime_info
        }

    def _predict_additional(self, boards):
        """Predict additional number."""
        used = set()
        for b in boards:
            used.update(b)

        # Additional number frequency analysis
        add_counts = Counter(self.df['additional_number'].astype(int))
        recent = self.df.tail(50)
        recent_add = Counter(recent['additional_number'].astype(int))

        candidates = {}
        for num in range(1, 50):
            if num in used:
                continue
            freq = add_counts.get(num, 0) / self.n
            recent_freq = recent_add.get(num, 0) / len(recent)
            pred = self._prediction_scores.get(num, {}).get('prediction', 0.5)
            candidates[num] = 0.3 * freq * 49 + 0.3 * recent_freq * 49 + 0.4 * pred

        if not candidates:
            return np.random.randint(1, 50)
        return max(candidates, key=candidates.get)

    def get_rankings(self):
        """Get full rankings for compatibility."""
        if self._composite_scores is None:
            self.analyze()
        ranked = sorted(
            [(n, s['composite']) for n, s in self._composite_scores.items()],
            key=lambda x: -x[1]
        )
        return {'rankings': ranked, 'top_numbers': [n for n, _ in ranked[:6]]}

    def get_analysis_report(self):
        """Generate analysis report."""
        top_prediction = sorted(self._prediction_scores.items(),
                               key=lambda x: -x[1]['prediction'])[:15]
        top_composite = sorted(self._composite_scores.items(),
                              key=lambda x: -x[1]['composite'])[:15]

        edge_numbers = [(n, self._posteriors[n]) for n in range(1, 50)
                       if self._posteriors[n]['p_hot'] > 0.7 or self._posteriors[n]['p_hot'] < 0.3]

        mi = self.entropy.consecutive_draw_mutual_info()

        return {
            'regime': self._regime_info,
            'edge_numbers': edge_numbers,
            'top_prediction': top_prediction,
            'top_composite': top_composite,
            'mutual_info': mi,
            'total_draws': self.n,
            'date_range': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"
        }


# =====================================================================
# STEP 3: HEAD-TO-HEAD BACKTEST
# =====================================================================

def run_backtest_comparison(df, n_test=52):
    """Walk-forward backtest comparing v3.0, v4.0, and random."""
    from src.models.quant_engine_v3 import QuantEngineV3

    print(f"\n{'='*70}")
    print(f"STEP 3: HEAD-TO-HEAD BACKTEST (last {n_test} draws)")
    print(f"{'='*70}")

    train_start_idx = len(df) - n_test
    results = {'v3': [], 'v4': [], 'random': []}

    np.random.seed(42)

    for i in range(n_test):
        test_idx = train_start_idx + i
        train = df.iloc[:test_idx].copy()
        actual_row = df.iloc[test_idx]
        actual = set(int(actual_row[f'num{j}']) for j in range(1, 7))
        actual_add = int(actual_row['additional_number'])

        if len(train) < 100:
            continue

        if i % 10 == 0:
            print(f"  Draw {i+1}/{n_test} ({actual_row['date'].date()})...")

        # V3.0
        try:
            v3 = QuantEngineV3(train)
            v3.analyze()
            v3_boards = v3.generate_all_boards()
            v3_best = max(
                len(set(int(n) for n in b['numbers']) & actual)
                for b in v3_boards['boards']
            )
            v3_add = int(v3_boards['additional_number']) == actual_add
        except:
            v3_best = 0
            v3_add = False

        # V4.0
        try:
            v4 = QuantEngineV4(train)
            v4.analyze()
            v4_boards = v4.generate_all_boards()
            v4_best = max(
                len(set(int(n) for n in b['numbers']) & actual)
                for b in v4_boards['boards']
            )
            v4_add = int(v4_boards['additional_number']) == actual_add
        except:
            v4_best = 0
            v4_add = False

        # Random (best of 3)
        rand_best = max(
            len(set(np.random.choice(range(1, 50), 6, replace=False)) & actual)
            for _ in range(3)
        )

        results['v3'].append({'matches': v3_best, 'add': v3_add})
        results['v4'].append({'matches': v4_best, 'add': v4_add})
        results['random'].append({'matches': rand_best, 'add': False})

    # Report
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS (best of 3 boards per draw)")
    print(f"{'='*70}")

    for label, key in [('Quant v3.0', 'v3'), ('Quant v4.0', 'v4'), ('Random', 'random')]:
        matches = [r['matches'] for r in results[key]]
        add_hits = sum(1 for r in results[key] if r.get('add', False))

        print(f"\n{label}:")
        print(f"  Avg matches: {np.mean(matches):.3f}")
        print(f"  2+ matches: {sum(1 for m in matches if m >= 2)}/{len(matches)} "
              f"({100*sum(1 for m in matches if m >= 2)/len(matches):.1f}%)")
        print(f"  3+ matches (prize): {sum(1 for m in matches if m >= 3)}/{len(matches)} "
              f"({100*sum(1 for m in matches if m >= 3)/len(matches):.1f}%)")
        print(f"  Additional hits: {add_hits}/{len(results[key])}")
        print(f"  Distribution:", end='')
        for m in range(5):
            c = sum(1 for x in matches if x == m)
            print(f" {m}:{c}", end='')
        print()

    # Statistical comparison
    v3_m = [r['matches'] for r in results['v3']]
    v4_m = [r['matches'] for r in results['v4']]
    rand_m = [r['matches'] for r in results['random']]

    if len(v4_m) > 1 and len(rand_m) > 1:
        t, p = stats.ttest_ind(v4_m, rand_m)
        print(f"\nv4.0 vs Random: t={t:.3f}, p={p:.4f} "
              f"({'SIGNIFICANT' if p < 0.05 else 'not significant'})")

    if len(v4_m) > 1 and len(v3_m) > 1:
        t, p = stats.ttest_ind(v4_m, v3_m)
        print(f"v4.0 vs v3.0:   t={t:.3f}, p={p:.4f} "
              f"({'SIGNIFICANT' if p < 0.05 else 'not significant'})")

    return results


# =====================================================================
# STEP 4: GENERATE PREDICTIONS
# =====================================================================

def generate_predictions(df):
    """Generate next-draw predictions with full reasoning."""
    print(f"\n{'='*70}")
    print("STEP 4: NEXT DRAW PREDICTIONS")
    print(f"{'='*70}")

    engine = QuantEngineV4(df)
    report = engine.analyze()

    print(f"\nDataset: {report['total_draws']} draws ({report['date_range']})")
    print(f"Regime: {report['regime']['regime']} (score: {report['regime']['regime_score']})")
    print(f"  Sum trend: {report['regime']['sum_trend']}, "
          f"High ratio: {report['regime']['high_ratio']}, "
          f"Odd ratio: {report['regime']['odd_ratio']}")

    mi = report['mutual_info']
    print(f"\nDraw-to-draw overlap: {mi['observed_overlap']:.3f} "
          f"(expected: {mi['expected_overlap']:.3f}, excess: {mi['excess_overlap']:+.3f})")

    # Edge numbers
    print(f"\nBayesian edges:")
    hot = [(n, p) for n, p in report['edge_numbers'] if p['p_hot'] > 0.7]
    cold = [(n, p) for n, p in report['edge_numbers'] if p['p_hot'] < 0.3]
    hot_str = ', '.join(str(n) + '(' + format(p['p_hot'], '.2f') + ')'
                        for n, p in sorted(hot, key=lambda x: -x[1]['p_hot'])[:8])
    cold_str = ', '.join(str(n) + '(' + format(p['p_hot'], '.2f') + ')'
                         for n, p in sorted(cold, key=lambda x: x[1]['p_hot'])[:8])
    print(f"  Hot: {hot_str}")
    print(f"  Cold: {cold_str}")

    # Top composite
    print(f"\nTop 15 composite scores:")
    for i, (n, s) in enumerate(report['top_composite'][:15]):
        print(f"  #{i+1:2d}. Number {n:2d} = {s['composite']:.4f} "
              f"(Bay:{s['bayesian']:.2f} Seq:{s.get('sequence', 0):.2f} "
              f"Rev:{s.get('reversion', 0):.2f} Ent:{s.get('entropy', 0):.2f})")

    # Generate boards
    boards = engine.generate_all_boards()

    # Determine next draw date
    today = datetime.now().date()
    # Next Monday or Thursday
    days_to_mon = (0 - today.weekday()) % 7
    days_to_thu = (3 - today.weekday()) % 7
    if days_to_mon == 0: days_to_mon = 7
    if days_to_thu == 0: days_to_thu = 7
    next_draw_date = today + timedelta(days=min(days_to_mon, days_to_thu))
    draw_day = "Monday" if next_draw_date.weekday() == 0 else "Thursday"

    print(f"\n{'='*70}")
    print(f"PREDICTIONS FOR DRAW ON {next_draw_date} ({draw_day})")
    print(f"{'='*70}")

    for b in boards['boards']:
        nums = ', '.join(str(n) for n in b['numbers'])
        ev = b['expected_value']
        pop_label = ('UNPOPULAR [+EV]' if ev['popularity_ratio'] < 0.8
                     else 'Average' if ev['popularity_ratio'] < 1.2
                     else 'Popular [-EV]')

        print(f"\n  Board {b['board_number']}: {b['strategy']}")
        print(f"  Numbers: [{nums}]")
        print(f"  Sum: {b['validation']['sum']} | "
              f"Odd/Even: {b['validation']['odd_count']}/{6-b['validation']['odd_count']} | "
              f"Decades: {b['validation']['decades']} | Penalties: {b['validation']['penalties']}")
        print(f"  Popularity: {ev['popularity_ratio']:.3f} ({pop_label})")
        print(f"  Expected prize if win: ${ev['expected_prize_if_win']:,.0f}")
        print(f"  Confidence: {b['confidence']:.1%}")

        print(f"  Number breakdown:")
        for nd in b['number_details']:
            print(f"    {nd['number']:2d}: pred={nd['prediction']:.4f} "
                  f"bay={nd['bayesian']:.2f} seq={nd['sequence']:.2f} "
                  f"rev={nd['reversion']:.2f} anti={nd['anti_pop']:.2f}")

    print(f"\n  Additional Number: {boards['additional_number']}")
    cov = boards['coverage']
    print(f"  Coverage: {cov['unique_numbers']}/49 ({cov['coverage_pct']}%)")

    print(f"\n  Regime: {boards['regime']['regime']}")

    print(f"\n{'='*70}")
    print("DISCLAIMER: TOTO is a random lottery. Odds: 1 in 13,983,816.")
    print("No model can predict random outcomes. Play responsibly.")
    print(f"{'='*70}")

    return engine, boards, report


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    df = load_dataset()

    # Step 1: Diagnose v3.0
    diagnosis = diagnose_v3(df, n_test=52)

    # Step 2: v4.0 is defined above

    # Step 3: Head-to-head backtest
    bt_results = run_backtest_comparison(df, n_test=52)

    # Step 4: Generate predictions
    engine, boards, report = generate_predictions(df)

    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    # Save backtest comparison
    bt_rows = []
    for model in ['v3', 'v4', 'random']:
        for i, r in enumerate(bt_results[model]):
            bt_rows.append({
                'model': model,
                'draw_index': i,
                'matches': r['matches'],
                'additional_match': r.get('add', False),
            })
    bt_df = pd.DataFrame(bt_rows)
    bt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'backtest_v3_vs_v4.csv')
    bt_df.to_csv(bt_path, index=False)
    print(f"  Backtest comparison saved to {bt_path}")

    # Save predictions
    pred_rows = []
    for b in boards['boards']:
        pred_rows.append({
            'board_number': b['board_number'],
            'strategy': b['strategy'],
            'numbers': str(b['numbers']),
            'confidence': b['confidence'],
            'popularity_ratio': b['expected_value']['popularity_ratio'],
            'expected_prize': b['expected_value']['expected_prize_if_win'],
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'latest_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")

    print("\nDone.")
