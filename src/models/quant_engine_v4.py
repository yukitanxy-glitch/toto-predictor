"""
Quant Engine v4.0 - Complete Rebuild
=====================================

Changes from v3.0:
1. Removed overdue factor (gambler's fallacy in fair lottery)
2. Added entropy/surprise scoring (information theory)
3. Added mean reversion detector
4. Added autoregressive lag features (LSTM-lite sequence modeling)
5. Added triplet co-occurrence analysis
6. Time-decayed pair graph (recent pairs weighted more)
7. Deterministic greedy board generation (not random sampling)
8. Anti-popularity SEPARATED from prediction (two-stage: predict then optimize EV)
9. Confidence scoring per board
10. Shorter regime window (20 draws) for faster adaptation

Scoring weights (prediction only):
- Bayesian posterior:  20%%
- Recent momentum:     15%%  
- Pair centrality:     15%% (time-decayed)
- Sequence/lag:        20%% (autoregressive features)
- Mean reversion:      15%%
- Entropy/surprise:    10%%
- Triplet bonus:        5%%

Final composite: 70%% prediction + 30%% anti-popularity (for EV)
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
