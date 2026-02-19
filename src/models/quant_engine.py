"""
Quant-Grade TOTO Prediction Engine v2.0

A fundamentally different approach to lottery prediction.
Instead of trying to predict winning numbers (impossible in a fair lottery),
this engine maximizes EXPECTED VALUE by:

1. Statistical Edge Detection - Find any micro-deviation from uniform
2. Expected Value Optimization - Pick numbers that maximize payout if won
3. Coverage Optimization - Spread boards for maximum prize capture
4. Anti-Correlation Strategy - Avoid what other players pick
5. Proper Backtesting - Never fool ourselves with overfitting

Key Innovation: We don't try to beat randomness.
We assume TOTO is fair and optimize for PRIZE VALUE instead.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.optimize import minimize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class QuantEdgeDetector:
    """
    Phase 1: Detect any statistically significant deviations from uniform.
    Uses rigorous hypothesis testing - not pattern-matching on noise.
    """

    def __init__(self, df, significance=0.05):
        self.df = df
        self.significance = significance
        self.n_draws = len(df)
        self.numbers = list(range(1, 50))
        self.expected_freq = self.n_draws * 6 / 49  # ~128.6 for 1050 draws

        # Extract all main numbers
        self.all_nums = []
        for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
            self.all_nums.extend(df[col].values)
        self.freq = Counter(self.all_nums)

    def test_uniformity(self):
        """Chi-squared test: are all numbers equally likely?"""
        observed = [self.freq.get(n, 0) for n in self.numbers]
        chi2, p_value = stats.chisquare(observed)
        return {
            'test': 'Chi-squared uniformity',
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < self.significance,
            'interpretation': 'Numbers are NOT uniformly distributed' if p_value < self.significance
                            else 'Numbers appear uniformly distributed (no edge)'
        }

    def test_serial_independence(self):
        """Test if consecutive draws are independent (runs test on each number)."""
        significant_numbers = []
        for num in self.numbers:
            # Binary sequence: did this number appear in each draw?
            appears = []
            for _, row in self.df.iterrows():
                draw_nums = {row['num1'], row['num2'], row['num3'],
                           row['num4'], row['num5'], row['num6']}
                appears.append(1 if num in draw_nums else 0)

            # Wald-Wolfowitz runs test
            n1 = sum(appears)
            n0 = len(appears) - n1
            if n1 == 0 or n0 == 0:
                continue

            # Count runs
            runs = 1
            for i in range(1, len(appears)):
                if appears[i] != appears[i-1]:
                    runs += 1

            # Expected runs and variance under independence
            n = len(appears)
            expected_runs = 1 + (2 * n1 * n0) / n
            var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
            if var_runs <= 0:
                continue

            z = (runs - expected_runs) / np.sqrt(var_runs)
            p = 2 * (1 - stats.norm.cdf(abs(z)))

            if p < self.significance / 49:  # Bonferroni correction
                significant_numbers.append({
                    'number': num,
                    'runs': runs,
                    'expected_runs': expected_runs,
                    'z_score': z,
                    'p_value': p,
                    'pattern': 'clustering' if z < 0 else 'alternating'
                })

        return {
            'test': 'Serial independence (runs test, Bonferroni-corrected)',
            'numbers_tested': 49,
            'significant_count': len(significant_numbers),
            'significant_numbers': significant_numbers,
            'has_edge': len(significant_numbers) > 0
        }

    def test_pair_dependence(self):
        """Test if any number pairs appear together more than expected."""
        # Observed pair frequencies
        pair_counts = Counter()
        for _, row in self.df.iterrows():
            nums = sorted([row['num1'], row['num2'], row['num3'],
                          row['num4'], row['num5'], row['num6']])
            for pair in combinations(nums, 2):
                pair_counts[pair] += 1

        # Expected pair frequency: C(47,4)/C(49,6) * n_draws
        # = n_draws * 6 * 5 / (49 * 48) = n_draws * 30/2352
        expected_pair = self.n_draws * 30 / 2352

        # Chi-squared test on pair frequencies
        n_pairs = len(list(combinations(range(1, 50), 2)))  # 1176
        observed_pairs = [pair_counts.get((i, j), 0)
                         for i in range(1, 50) for j in range(i+1, 50)]
        chi2, p_value = stats.chisquare(observed_pairs)

        # Find most over-represented pairs
        top_pairs = []
        for pair, count in pair_counts.most_common(20):
            z = (count - expected_pair) / np.sqrt(expected_pair * (1 - 30/2352))
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            top_pairs.append({
                'pair': pair,
                'observed': count,
                'expected': round(expected_pair, 2),
                'z_score': round(z, 3),
                'p_value': round(p, 4),
                'significant': p < self.significance / n_pairs
            })

        return {
            'test': 'Pair dependence (chi-squared)',
            'chi2': chi2,
            'p_value': p_value,
            'expected_per_pair': round(expected_pair, 2),
            'top_pairs': top_pairs,
            'any_significant_after_correction': any(p['significant'] for p in top_pairs)
        }

    def test_recency_effect(self):
        """Test if recently drawn numbers are more/less likely to appear again."""
        # For each draw, compute: of the 6 numbers drawn, how many appeared
        # in the previous 1, 3, 5, 10 draws?
        lookbacks = [1, 3, 5, 10]
        results = {}

        for lb in lookbacks:
            overlaps = []
            for i in range(lb, self.n_draws):
                current = set()
                row = self.df.iloc[i]
                current = {row['num1'], row['num2'], row['num3'],
                          row['num4'], row['num5'], row['num6']}

                recent = set()
                for j in range(max(0, i - lb), i):
                    prev = self.df.iloc[j]
                    recent.update({prev['num1'], prev['num2'], prev['num3'],
                                  prev['num4'], prev['num5'], prev['num6']})

                overlap = len(current & recent)
                overlaps.append(overlap)

            # Expected overlap: hypergeometric
            # Recent pool has min(lb*6, 49) unique numbers
            # We're drawing 6 from 49, how many from the recent pool?
            recent_pool_size = min(lb * 6, 49)
            expected = 6 * recent_pool_size / 49
            observed_mean = np.mean(overlaps)

            # One-sample t-test
            t_stat, p_val = stats.ttest_1samp(overlaps, expected)

            results[f'lookback_{lb}'] = {
                'lookback_draws': lb,
                'expected_overlap': round(expected, 3),
                'observed_mean_overlap': round(observed_mean, 3),
                'difference': round(observed_mean - expected, 3),
                't_statistic': round(t_stat, 3),
                'p_value': round(p_val, 4),
                'significant': p_val < self.significance
            }

        return {
            'test': 'Recency effect (overlap with recent draws)',
            'results': results
        }

    def compute_number_edge_scores(self):
        """
        Combine all edge detections into a per-number score.
        Positive = slight statistical advantage. Zero = no edge.
        """
        scores = np.zeros(49)

        # 1. Frequency deviation (z-scores from expected)
        for i, num in enumerate(self.numbers):
            observed = self.freq.get(num, 0)
            z = (observed - self.expected_freq) / np.sqrt(self.expected_freq)
            # Only count if the overall uniformity test is significant
            uniformity = self.test_uniformity()
            if uniformity['significant']:
                scores[i] += z * 0.3  # Small weight on frequency edge

        # 2. Serial dependence edge
        serial = self.test_serial_independence()
        for item in serial['significant_numbers']:
            idx = item['number'] - 1
            if item['pattern'] == 'clustering':
                # Number tends to cluster - check if it appeared recently
                last_seen = 0
                for j in range(len(self.df) - 1, -1, -1):
                    row = self.df.iloc[j]
                    if item['number'] in {row['num1'], row['num2'], row['num3'],
                                         row['num4'], row['num5'], row['num6']}:
                        last_seen = len(self.df) - 1 - j
                        break
                if last_seen <= 3:  # Recently appeared and clusters
                    scores[idx] += 0.5
            elif item['pattern'] == 'alternating':
                # Number alternates - check if it was absent recently
                last_seen = 0
                for j in range(len(self.df) - 1, -1, -1):
                    row = self.df.iloc[j]
                    if item['number'] in {row['num1'], row['num2'], row['num3'],
                                         row['num4'], row['num5'], row['num6']}:
                        last_seen = len(self.df) - 1 - j
                        break
                if last_seen >= 5:  # Absent recently and alternates
                    scores[idx] += 0.5

        return {num: round(float(scores[num - 1]), 4) for num in self.numbers}

    def run_all_tests(self):
        """Run complete edge detection suite."""
        results = {
            'uniformity': self.test_uniformity(),
            'serial_independence': self.test_serial_independence(),
            'pair_dependence': self.test_pair_dependence(),
            'recency_effect': self.test_recency_effect(),
            'edge_scores': self.compute_number_edge_scores()
        }

        # Summary
        has_any_edge = (results['uniformity']['significant'] or
                       results['serial_independence']['has_edge'] or
                       results['pair_dependence']['any_significant_after_correction'])

        results['summary'] = {
            'any_statistical_edge_detected': has_any_edge,
            'recommendation': 'Use edge-adjusted probabilities' if has_any_edge
                            else 'No statistical edge found - optimize for expected prize value instead'
        }

        return results


class ExpectedValueOptimizer:
    """
    Phase 2: Maximize expected prize value.

    Key insight: If we can't change P(winning), we maximize E[prize|winning].
    This means picking numbers that OTHER PLAYERS avoid.

    Prize(board) = JackpotPool / N_winners_sharing

    If we pick unpopular numbers, fewer people share our prize group.
    """

    def __init__(self, df):
        self.df = df
        self.n_draws = len(df)

        # Estimate number popularity (what other players likely pick)
        self.popularity_scores = self._estimate_popularity()

    def _estimate_popularity(self):
        """
        Estimate how popular each number is among other TOTO players.
        Based on behavioral research:
        - Birthday bias: numbers 1-31 are heavily over-picked
        - Lucky numbers: 7, 8 (Chinese lucky), 13 (contrarian), 28 (prosperity)
        - Cultural: 4 avoided (death in Chinese), 14, 24 avoided
        - Patterns: sequences, multiples of 7, round numbers
        - Visual: numbers at edges of the bet slip
        """
        popularity = np.ones(49)

        for num in range(1, 50):
            score = 1.0

            # Birthday effect (1-31 are dates)
            if num <= 12:  # Month numbers
                score *= 1.8
            elif num <= 31:  # Day numbers
                score *= 1.4

            # Chinese lucky numbers (Singapore is ~75% Chinese)
            if num in [8, 18, 28, 38, 48]:  # 8 = prosperity
                score *= 1.5
            if num in [6, 16, 26, 36, 46]:  # 6 = smooth/easy
                score *= 1.2
            if num in [9, 19, 29, 39, 49]:  # 9 = longevity
                score *= 1.2

            # Number 7 - universally lucky
            if num in [7, 17, 27, 37, 47]:
                score *= 1.3

            # Chinese unlucky (4 = death)
            if num in [4, 14, 24, 34, 44]:
                score *= 0.7

            # Round numbers (multiples of 5, 10)
            if num % 10 == 0:
                score *= 1.3
            elif num % 5 == 0:
                score *= 1.15

            # Low numbers overall more popular
            if num <= 10:
                score *= 1.2

            # Very high numbers less picked
            if num >= 40:
                score *= 0.85

            # Consecutive clusters (people pick 1,2,3 or 5,6,7)
            # Already captured by low number bias

            popularity[num - 1] = score

        # Normalize to sum to 49 (average = 1.0)
        popularity = popularity / popularity.mean()
        return popularity

    def compute_anti_popularity_scores(self):
        """
        Score each number inversely to its estimated popularity.
        Higher score = fewer other players pick this number =
        higher expected prize if it wins.
        """
        # Inverse popularity (less popular = higher EV)
        inv_pop = 1.0 / self.popularity_scores
        # Normalize to [0, 1]
        inv_pop = (inv_pop - inv_pop.min()) / (inv_pop.max() - inv_pop.min())
        return {num: round(float(inv_pop[num - 1]), 4) for num in range(1, 50)}

    def estimate_expected_value(self, board, jackpot_estimate=5_000_000):
        """
        Estimate expected value of a specific 6-number board.

        E[V] = P(win) * E[prize | win]
        E[prize | win] = Jackpot / E[n_winners]
        E[n_winners] is proportional to how popular the combination is.

        P(win) is always 1/13,983,816 for any combination.
        But E[prize | win] varies based on popularity.
        """
        # P(winning jackpot) - same for all combinations
        p_win = 1 / 13_983_816

        # Popularity of this board = product of individual popularities
        # (simplified: assume independence of player selections)
        board_popularity = 1.0
        for num in board:
            board_popularity *= self.popularity_scores[num - 1]

        # Average board popularity (for calibration)
        # A random board would have popularity ~1.0^6 = 1.0
        avg_popularity = 1.0

        # Expected number of co-winners is proportional to popularity
        # Base: assume ~1M tickets sold per draw (typical for SG TOTO)
        tickets_per_draw = 1_000_000
        base_expected_winners = tickets_per_draw * p_win  # ~0.07

        popularity_ratio = board_popularity / avg_popularity
        expected_winners = base_expected_winners * popularity_ratio + 1  # +1 for us

        # Expected prize per winner
        expected_prize = jackpot_estimate / expected_winners

        # Expected value of this ticket
        ev = p_win * expected_prize

        # Ticket cost
        ticket_cost = 1.0  # $1 per board

        return {
            'board': sorted(board),
            'p_win': p_win,
            'board_popularity': round(board_popularity, 4),
            'expected_co_winners': round(expected_winners, 4),
            'expected_prize_if_win': round(expected_prize, 0),
            'expected_value': round(ev, 6),
            'ev_ratio': round(ev / ticket_cost, 4),
            'popularity_ratio': round(popularity_ratio, 4)
        }


class CoverageOptimizer:
    """
    Phase 3: Optimize multiple boards for maximum coverage.

    Instead of picking similar numbers across boards,
    maximize the probability of getting at least one prize
    across all boards combined.
    """

    def __init__(self, n_boards=3):
        self.n_boards = n_boards

    def compute_coverage_score(self, boards):
        """
        Score a set of boards on how well they cover the number space.
        Higher = better diversification.
        """
        all_numbers = set()
        for board in boards:
            all_numbers.update(board)

        # Unique number coverage (out of 49)
        coverage = len(all_numbers) / 49

        # Decade group coverage per board
        decade_coverage = 0
        for board in boards:
            decades = set()
            for n in board:
                decades.add((n - 1) // 10)
            decade_coverage += len(decades) / 5
        decade_coverage /= len(boards)

        # Parity diversity across boards
        parity_diversity = 0
        for board in boards:
            odd = sum(1 for n in board if n % 2 == 1)
            parity_diversity += 1 - abs(odd - 3) / 3
        parity_diversity /= len(boards)

        # Overlap penalty (boards sharing numbers)
        overlap_penalty = 0
        for i in range(len(boards)):
            for j in range(i + 1, len(boards)):
                shared = len(set(boards[i]) & set(boards[j]))
                overlap_penalty += shared / 6
        max_pairs = len(boards) * (len(boards) - 1) / 2
        if max_pairs > 0:
            overlap_penalty /= max_pairs

        # Combined score
        score = (0.35 * coverage +
                0.25 * decade_coverage +
                0.15 * parity_diversity +
                0.25 * (1 - overlap_penalty))

        return {
            'total_score': round(score, 4),
            'unique_numbers': len(all_numbers),
            'coverage_pct': round(coverage * 100, 1),
            'decade_coverage': round(decade_coverage, 3),
            'parity_balance': round(parity_diversity, 3),
            'overlap_penalty': round(overlap_penalty, 3)
        }


class QuantPredictor:
    """
    Master class: Combines all three optimization phases.
    """

    def __init__(self, df):
        self.df = df
        self.edge_detector = QuantEdgeDetector(df)
        self.ev_optimizer = ExpectedValueOptimizer(df)
        self.coverage_optimizer = CoverageOptimizer(n_boards=3)

        # Run edge detection once
        self.edge_results = None
        self.edge_scores = None
        self.anti_pop_scores = None

    def analyze(self):
        """Run full edge detection analysis."""
        self.edge_results = self.edge_detector.run_all_tests()
        self.edge_scores = self.edge_results['edge_scores']
        self.anti_pop_scores = self.ev_optimizer.compute_anti_popularity_scores()
        return self.edge_results

    def _compute_composite_scores(self, strategy='balanced'):
        """
        Compute per-number composite score combining:
        - Edge score (statistical deviation, if any)
        - Anti-popularity (expected value)
        - Structural fitness (sum/parity/spread contribution)
        """
        if self.edge_scores is None:
            self.analyze()

        composite = {}
        for num in range(1, 50):
            edge = self.edge_scores.get(num, 0)
            anti_pop = self.anti_pop_scores.get(num, 0.5)

            if strategy == 'edge_focus':
                # Maximize statistical edge (if any exists)
                score = 0.6 * edge + 0.3 * anti_pop + 0.1 * 0.5
            elif strategy == 'value_focus':
                # Maximize expected prize value
                score = 0.15 * edge + 0.7 * anti_pop + 0.15 * 0.5
            elif strategy == 'balanced':
                # Balance edge and value
                score = 0.35 * edge + 0.45 * anti_pop + 0.20 * 0.5
            else:
                score = 0.33 * edge + 0.34 * anti_pop + 0.33 * 0.5

            composite[num] = score

        return composite

    def _validate_board(self, board, df=None):
        """Soft validation: score a board on structural properties."""
        board = sorted(board)
        penalties = 0
        details = []

        # Sum check (historical 70% zone: roughly 115-185)
        total = sum(board)
        if total < 100 or total > 200:
            penalties += 2
            details.append(f'Sum {total} is extreme')
        elif total < 115 or total > 185:
            penalties += 1
            details.append(f'Sum {total} is outside 70% zone')

        # Odd/even (penalize extremes, not hard reject)
        odd = sum(1 for n in board if n % 2 == 1)
        if odd in [0, 6]:
            penalties += 3
            details.append(f'Odd/Even {odd}/{6-odd} is extreme')
        elif odd in [1, 5]:
            penalties += 1
            details.append(f'Odd/Even {odd}/{6-odd} is borderline')

        # High/low
        high = sum(1 for n in board if n >= 25)
        if high in [0, 6]:
            penalties += 3
            details.append(f'High/Low extreme')
        elif high in [1, 5]:
            penalties += 1

        # Consecutive numbers
        max_consec = 1
        current = 1
        for i in range(1, len(board)):
            if board[i] == board[i-1] + 1:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 1
        if max_consec >= 4:
            penalties += 3
            details.append(f'{max_consec} consecutive numbers')
        elif max_consec >= 3:
            penalties += 1

        # Decade spread
        decades = set((n - 1) // 10 for n in board)
        if len(decades) <= 2:
            penalties += 2
            details.append(f'Only {len(decades)} decade groups')

        return {
            'board': board,
            'penalties': penalties,
            'valid': penalties <= 2,
            'details': details,
            'sum': total,
            'odd_count': odd,
            'high_count': high,
            'decades': len(decades),
            'max_consecutive': max_consec
        }

    def generate_board(self, strategy='balanced', exclude_numbers=None, max_attempts=2000):
        """
        Generate a single optimized board.

        Uses weighted random sampling with composite scores,
        then validates with soft constraints.
        """
        if exclude_numbers is None:
            exclude_numbers = set()

        scores = self._compute_composite_scores(strategy)

        # Remove excluded numbers
        candidates = {n: s for n, s in scores.items() if n not in exclude_numbers}

        best_board = None
        best_penalty = float('inf')
        best_ev = 0

        for _ in range(max_attempts):
            # Convert scores to sampling weights
            numbers = list(candidates.keys())
            weights = np.array([max(candidates[n], 0.01) for n in numbers])
            weights = weights / weights.sum()

            # Sample 6 numbers
            board = sorted(np.random.choice(numbers, size=6, replace=False, p=weights))

            # Validate
            validation = self._validate_board(board)
            if validation['valid']:
                ev_info = self.ev_optimizer.estimate_expected_value(board)
                penalty = validation['penalties']
                ev = ev_info['expected_value']

                # Keep the board with lowest penalties and highest EV
                score = penalty - ev * 1000000  # Scale EV to be comparable
                if penalty < best_penalty or (penalty == best_penalty and ev > best_ev):
                    best_board = board
                    best_penalty = penalty
                    best_ev = ev

                if penalty == 0:
                    break

        if best_board is None:
            # Fallback: just take top 6 by score
            ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            best_board = sorted([n for n, _ in ranked[:6]])

        return best_board

    def generate_all_boards(self):
        """
        Generate 3 optimized boards with maximum coverage.

        Board 1: Edge + Value balanced (best overall expected value)
        Board 2: Pure value focus (maximum anti-popularity)
        Board 3: Edge focus + contrarian (exploit any detected edges)
        """
        if self.edge_scores is None:
            self.analyze()

        # Board 1: Balanced strategy
        board1 = self.generate_board(strategy='balanced')

        # Board 2: Value focus (avoid board 1 numbers for coverage)
        board2 = self.generate_board(strategy='value_focus',
                                     exclude_numbers=set(board1))

        # Board 3: Edge focus (avoid boards 1 & 2)
        board3 = self.generate_board(strategy='edge_focus',
                                     exclude_numbers=set(board1) | set(board2))

        # Compute metrics for all boards
        boards = [board1, board2, board3]
        board_info = []
        for i, board in enumerate(boards):
            ev = self.ev_optimizer.estimate_expected_value(board)
            val = self._validate_board(board)
            board_info.append({
                'board_number': i + 1,
                'numbers': sorted(board),
                'strategy': ['Balanced Edge+Value', 'Max Prize Value', 'Edge Exploiter'][i],
                'expected_value': ev,
                'validation': val
            })

        # Coverage analysis
        coverage = self.coverage_optimizer.compute_coverage_score(boards)

        # Predict additional number (7th ball)
        additional = self._predict_additional(boards)

        return {
            'boards': board_info,
            'coverage': coverage,
            'additional_number': additional,
            'edge_analysis_summary': self.edge_results['summary']
        }

    def _predict_additional(self, boards):
        """
        Predict the additional number.
        Use anti-popularity + frequency, excluding all board numbers.
        """
        used = set()
        for b in boards:
            used.update(b)

        # Score remaining numbers
        candidates = {}
        for num in range(1, 50):
            if num in used:
                continue
            freq_score = self.edge_detector.freq.get(num, 0) / self.edge_detector.expected_freq
            anti_pop = self.anti_pop_scores.get(num, 0.5)
            candidates[num] = 0.4 * freq_score + 0.6 * anti_pop

        if not candidates:
            return np.random.randint(1, 50)

        # Return top candidate
        return max(candidates, key=candidates.get)

    def get_rankings(self):
        """Get full number rankings for compatibility with ensemble."""
        if self.edge_scores is None:
            self.analyze()

        scores = self._compute_composite_scores('balanced')
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {
            'rankings': ranked,
            'top_numbers': [n for n, _ in ranked[:6]]
        }

    def backtest(self, test_size=100, verbose=True):
        """
        Walk-forward backtest: for each test draw, use only prior data.
        Compare our method vs random vs frequency-only.
        """
        n = len(self.df)
        train_start = n - test_size

        quant_matches = []
        random_matches = []
        freq_matches = []
        quant_prizes = []

        for i in range(train_start, n):
            train_df = self.df.iloc[:i].copy()
            test_row = self.df.iloc[i]
            actual = set([test_row['num1'], test_row['num2'], test_row['num3'],
                         test_row['num4'], test_row['num5'], test_row['num6']])

            # Our quant prediction
            qp = QuantPredictor(train_df)
            qp.analyze()
            boards_result = qp.generate_all_boards()

            best_quant = 0
            for b in boards_result['boards']:
                matches = len(set(b['numbers']) & actual)
                best_quant = max(best_quant, matches)

            quant_matches.append(best_quant)

            # Prize group for best board
            prize = self._match_to_prize(best_quant, False)
            quant_prizes.append(prize)

            # Random baseline (best of 3 random boards)
            best_random = 0
            for _ in range(3):
                rand_board = sorted(np.random.choice(range(1, 50), 6, replace=False))
                matches = len(set(rand_board) & actual)
                best_random = max(best_random, matches)
            random_matches.append(best_random)

            # Frequency-only baseline
            all_nums = []
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                all_nums.extend(train_df[col].values)
            freq = Counter(all_nums)
            freq_board = sorted([n for n, _ in freq.most_common(6)])
            freq_match = len(set(freq_board) & actual)
            freq_matches.append(freq_match)

            if verbose and (i - train_start) % 20 == 0:
                print(f'  Backtesting draw {i - train_start + 1}/{test_size}...')

        # Compute statistics
        quant_arr = np.array(quant_matches)
        random_arr = np.array(random_matches)
        freq_arr = np.array(freq_matches)

        # T-test: quant vs random
        t_stat, p_val = stats.ttest_ind(quant_arr, random_arr)

        # Match distribution
        quant_dist = Counter(quant_matches)
        random_dist = Counter(random_matches)

        return {
            'test_size': test_size,
            'quant_avg_matches': round(float(quant_arr.mean()), 3),
            'random_avg_matches': round(float(random_arr.mean()), 3),
            'freq_avg_matches': round(float(freq_arr.mean()), 3),
            'quant_distribution': dict(sorted(quant_dist.items())),
            'random_distribution': dict(sorted(random_dist.items())),
            'quant_any_prize_pct': round(sum(1 for m in quant_matches if m >= 3) / test_size * 100, 1),
            'random_any_prize_pct': round(sum(1 for m in random_matches if m >= 3) / test_size * 100, 1),
            't_statistic': round(t_stat, 3),
            'p_value': round(p_val, 4),
            'significant_improvement': p_val < 0.05 and quant_arr.mean() > random_arr.mean()
        }

    def _match_to_prize(self, matches, has_additional):
        """Convert match count to prize group."""
        if matches == 6:
            return 1  # Jackpot
        elif matches == 5 and has_additional:
            return 2
        elif matches == 5:
            return 3
        elif matches == 4 and has_additional:
            return 4
        elif matches == 4:
            return 5
        elif matches == 3 and has_additional:
            return 6
        elif matches == 3:
            return 7
        return 0  # No prize


def run_quant_prediction(df=None):
    """Main entry point for quant prediction."""
    if df is None:
        df = pd.read_csv('data/toto_results.csv')

    print("=" * 70)
    print("QUANT-GRADE TOTO PREDICTION ENGINE v2.0")
    print("=" * 70)
    print(f"Dataset: {len(df)} real draws")
    print(f"Period: {df['date'].min()} to {df['date'].max()}")
    print()

    predictor = QuantPredictor(df)

    # Phase 1: Edge Detection
    print("PHASE 1: STATISTICAL EDGE DETECTION")
    print("-" * 50)
    results = predictor.analyze()

    uni = results['uniformity']
    print(f"  Uniformity test: chi2={uni['chi2']:.2f}, p={uni['p_value']:.4f}")
    print(f"  -> {uni['interpretation']}")

    serial = results['serial_independence']
    print(f"  Serial independence: {serial['significant_count']}/49 numbers show dependence")

    pairs = results['pair_dependence']
    print(f"  Pair dependence: {'Some pairs significant' if pairs['any_significant_after_correction'] else 'No significant pairs'}")

    recency = results['recency_effect']
    for key, val in recency['results'].items():
        if val['significant']:
            print(f"  Recency ({key}): SIGNIFICANT - overlap {val['observed_mean_overlap']:.3f} vs expected {val['expected_overlap']:.3f}")

    print(f"\n  VERDICT: {results['summary']['recommendation']}")

    # Phase 2: Generate Boards
    print(f"\n{'=' * 70}")
    print("PHASE 2: OPTIMIZED BOARD GENERATION")
    print("-" * 50)

    boards = predictor.generate_all_boards()

    for b in boards['boards']:
        nums_str = ', '.join(str(n) for n in b['numbers'])
        ev = b['expected_value']
        print(f"\n  Board {b['board_number']} [{b['strategy']}]:")
        print(f"    Numbers: {nums_str}")
        print(f"    Sum: {b['validation']['sum']} | "
              f"Odd/Even: {b['validation']['odd_count']}/{6-b['validation']['odd_count']} | "
              f"Decades: {b['validation']['decades']}")
        print(f"    Popularity ratio: {ev['popularity_ratio']:.3f} "
              f"({'UNPOPULAR - good' if ev['popularity_ratio'] < 0.8 else 'Average' if ev['popularity_ratio'] < 1.2 else 'Popular - risky'})")
        print(f"    Expected prize if win: ${ev['expected_prize_if_win']:,.0f} "
              f"(vs ~$4.7M for average combo)")

    print(f"\n  Additional number prediction: {boards['additional_number']}")

    # Coverage
    cov = boards['coverage']
    print(f"\n  Coverage Analysis:")
    print(f"    Unique numbers across all boards: {cov['unique_numbers']}/49 ({cov['coverage_pct']}%)")
    print(f"    Board overlap penalty: {cov['overlap_penalty']:.3f} (lower = better)")
    print(f"    Overall coverage score: {cov['total_score']:.3f}")

    print(f"\n{'=' * 70}")
    print("DISCLAIMER: TOTO is a random lottery. No model guarantees wins.")
    print(f"Group 1 odds: 1 in 13,983,816 per combination.")
    print(f"This model maximizes EXPECTED PRIZE VALUE, not win probability.")
    print(f"{'=' * 70}")

    return predictor, boards


if __name__ == '__main__':
    predictor, boards = run_quant_prediction()
