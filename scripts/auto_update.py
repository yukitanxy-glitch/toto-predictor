#!/usr/bin/env python3
"""
TOTO Self-Learning Auto-Update Pipeline v3.0

This script is designed to run automatically after each TOTO draw.
It handles the FULL lifecycle:

1. DETECT  - Knows today's date and next draw date
2. FETCH   - Scrapes latest results from Lottolyzer (proven source)
3. VALIDATE - Ensures data integrity before adding
4. SCORE   - Checks how previous prediction performed
5. ADAPT   - Adjusts model weights based on performance history
6. RETRAIN - Runs all models on updated dataset
7. PREDICT - Generates new boards for next draw
8. LOG     - Saves predictions and performance metrics
9. EXPORT  - Updates CSV and prediction log for Streamlit app

Designed to run via GitHub Actions cron (Mon & Thu after 7:30 PM SGT)
or manually via: python scripts/auto_update.py
"""

import os
import sys
import json
import csv
from datetime import datetime, timedelta, date
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "toto_results.csv")
PREDICTIONS_LOG = os.path.join(DATA_DIR, "predictions_log.csv")
PERFORMANCE_LOG = os.path.join(DATA_DIR, "performance_log.csv")
MODEL_WEIGHTS_FILE = os.path.join(DATA_DIR, "adaptive_weights.json")


# ══════════════════════════════════════════════════════════════════════════
# DRAW DATE AWARENESS
# ══════════════════════════════════════════════════════════════════════════

class DrawCalendar:
    """Knows TOTO draw schedule: Monday & Thursday, 6:30 PM SGT."""

    DRAW_DAYS = [0, 3]  # Monday=0, Thursday=3

    @staticmethod
    def get_today():
        """Get today's date."""
        return date.today()

    @staticmethod
    def is_draw_day(d=None):
        """Check if a date is a TOTO draw day."""
        if d is None:
            d = date.today()
        return d.weekday() in DrawCalendar.DRAW_DAYS

    @staticmethod
    def get_last_draw_date(from_date=None):
        """Get the most recent draw date on or before from_date."""
        if from_date is None:
            from_date = date.today()
        d = from_date
        while d.weekday() not in DrawCalendar.DRAW_DAYS:
            d -= timedelta(days=1)
        return d

    @staticmethod
    def get_next_draw_date(from_date=None):
        """Get the next draw date after from_date."""
        if from_date is None:
            from_date = date.today()
        d = from_date + timedelta(days=1)
        while d.weekday() not in DrawCalendar.DRAW_DAYS:
            d += timedelta(days=1)
        return d

    @staticmethod
    def get_missing_draw_dates(last_data_date, up_to_date=None):
        """Get all draw dates between last_data_date and up_to_date."""
        if up_to_date is None:
            up_to_date = date.today()

        if isinstance(last_data_date, str):
            last_data_date = datetime.strptime(last_data_date, '%Y-%m-%d').date()

        missing = []
        d = last_data_date + timedelta(days=1)
        while d <= up_to_date:
            if d.weekday() in DrawCalendar.DRAW_DAYS:
                missing.append(d)
            d += timedelta(days=1)
        return missing

    @staticmethod
    def format_schedule():
        """Print upcoming draw schedule."""
        today = date.today()
        lines = []
        lines.append(f"  Today: {today.strftime('%Y-%m-%d (%A)')}")
        lines.append(f"  Is draw day: {'YES' if DrawCalendar.is_draw_day() else 'No'}")
        lines.append(f"  Last draw: {DrawCalendar.get_last_draw_date().strftime('%Y-%m-%d (%A)')}")
        lines.append(f"  Next draw: {DrawCalendar.get_next_draw_date().strftime('%Y-%m-%d (%A)')}")

        lines.append(f"\n  Upcoming draws:")
        d = today
        count = 0
        while count < 6:
            d += timedelta(days=1)
            if d.weekday() in DrawCalendar.DRAW_DAYS:
                lines.append(f"    {d.strftime('%Y-%m-%d (%A)')}")
                count += 1
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════
# DATA FETCHER - Scrapes from Lottolyzer (proven reliable)
# ══════════════════════════════════════════════════════════════════════════

class LottolyzerFetcher:
    """Fetches TOTO results from en.lottolyzer.com (proven source)."""

    BASE_URL = "https://en.lottolyzer.com/history/singapore/toto/page/1/per-page/10/summary-view"

    @staticmethod
    def fetch_latest_draws(n_pages=1):
        """Fetch the most recent draws from Lottolyzer."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            print("[FETCH] requests/bs4 not installed, trying urllib...")
            return LottolyzerFetcher._fetch_with_urllib()

        results = []
        for page in range(1, n_pages + 1):
            url = f"https://en.lottolyzer.com/history/singapore/toto/page/{page}/per-page/20/summary-view"
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code != 200:
                    print(f"[FETCH] Page {page}: HTTP {resp.status_code}")
                    continue

                soup = BeautifulSoup(resp.text, 'html.parser')

                # Find all draw result rows
                rows = soup.select('table.history-summary tbody tr')
                if not rows:
                    # Try alternative selectors
                    rows = soup.select('tr[data-draw]')

                if not rows:
                    # Parse from text content
                    text = soup.get_text()
                    results.extend(LottolyzerFetcher._parse_text_content(text))
                    continue

                for row in rows:
                    try:
                        cells = row.find_all('td')
                        if len(cells) < 3:
                            continue

                        # Extract draw number, date, numbers
                        draw_num = int(cells[0].get_text(strip=True))
                        date_str = cells[1].get_text(strip=True)

                        # Numbers are usually in the 3rd cell
                        nums_text = cells[2].get_text(strip=True)
                        nums = [int(n.strip()) for n in nums_text.split(',') if n.strip().isdigit()]

                        if len(nums) >= 7:
                            main_nums = sorted(nums[:6])
                            additional = nums[6]
                        elif len(nums) == 6:
                            main_nums = sorted(nums)
                            additional = int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0
                        else:
                            continue

                        results.append({
                            'draw_number': draw_num,
                            'date': date_str,
                            'numbers': main_nums,
                            'additional': additional
                        })
                    except (ValueError, IndexError):
                        continue

                print(f"[FETCH] Page {page}: Found {len(results)} draws")

            except Exception as e:
                print(f"[FETCH] Error on page {page}: {e}")
                continue

        return results

    @staticmethod
    def _fetch_with_urllib():
        """Fallback fetcher using urllib."""
        import urllib.request
        import re

        url = "https://en.lottolyzer.com/history/singapore/toto/page/1/per-page/20/summary-view"
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode('utf-8')
                return LottolyzerFetcher._parse_text_content(html)
        except Exception as e:
            print(f"[FETCH] urllib fallback failed: {e}")
            return []

    @staticmethod
    def _parse_text_content(text):
        """Parse draw results from raw text/HTML content."""
        import re
        results = []

        # Pattern: draw_number followed by date followed by numbers
        # Look for patterns like: 4158 2026-02-19 8, 16, 17, 34, 38, 48 25
        pattern = r'(\d{4})\s+(\d{4}-\d{2}-\d{2})\s+(\d{1,2}(?:\s*,\s*\d{1,2}){5})\s+(\d{1,2})'
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                draw_num = int(match[0])
                date_str = match[1]
                nums = sorted([int(n.strip()) for n in match[2].split(',')])
                additional = int(match[3])

                if len(nums) == 6 and all(1 <= n <= 49 for n in nums) and 1 <= additional <= 49:
                    results.append({
                        'draw_number': draw_num,
                        'date': date_str,
                        'numbers': nums,
                        'additional': additional
                    })
            except ValueError:
                continue

        return results


# ══════════════════════════════════════════════════════════════════════════
# ADAPTIVE MODEL WEIGHTS (Self-Learning)
# ══════════════════════════════════════════════════════════════════════════

class AdaptiveWeights:
    """
    Learns optimal model weights from historical performance.
    After each draw, scores each model's prediction accuracy
    and adjusts weights using exponential moving average.
    """

    DEFAULT_WEIGHTS = {
        'quant_engine': 0.35,
        'weighted_scoring': 0.15,
        'monte_carlo': 0.15,
        'markov_chain': 0.10,
        'ensemble_legacy': 0.15,
        'random_baseline': 0.10
    }

    def __init__(self):
        self.weights = self._load_weights()
        self.performance_history = self._load_performance()

    def _load_weights(self):
        """Load saved weights or use defaults."""
        if os.path.exists(MODEL_WEIGHTS_FILE):
            try:
                with open(MODEL_WEIGHTS_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('weights', self.DEFAULT_WEIGHTS.copy())
            except (json.JSONDecodeError, KeyError):
                pass
        return self.DEFAULT_WEIGHTS.copy()

    def _load_performance(self):
        """Load performance history."""
        if os.path.exists(PERFORMANCE_LOG):
            try:
                return pd.read_csv(PERFORMANCE_LOG).to_dict('records')
            except Exception:
                pass
        return []

    def record_performance(self, draw_date, model_scores):
        """
        Record how each model performed on a specific draw.
        model_scores: dict of {model_name: n_matches}
        """
        entry = {'draw_date': str(draw_date)}
        entry.update(model_scores)
        self.performance_history.append(entry)

        # Save to CSV
        pd.DataFrame(self.performance_history).to_csv(PERFORMANCE_LOG, index=False)

    def update_weights(self, decay=0.95):
        """
        Update model weights based on recent performance.
        Uses exponential weighted average of match counts.
        """
        if len(self.performance_history) < 5:
            print("[ADAPT] Not enough history (<5 draws) to adapt weights. Using defaults.")
            return self.weights

        model_names = [k for k in self.DEFAULT_WEIGHTS.keys() if k != 'random_baseline']
        ewa_scores = {m: 0.0 for m in model_names}
        total_weight = 0.0

        for i, entry in enumerate(self.performance_history):
            age = len(self.performance_history) - 1 - i
            w = decay ** age
            total_weight += w

            for model in model_names:
                if model in entry:
                    try:
                        ewa_scores[model] += w * float(entry[model])
                    except (ValueError, TypeError):
                        pass

        if total_weight > 0:
            for model in model_names:
                ewa_scores[model] /= total_weight

        # Convert scores to weights (softmax-like)
        total = sum(max(s, 0.01) for s in ewa_scores.values())
        if total > 0:
            new_weights = {m: max(s, 0.01) / total for m, s in ewa_scores.items()}
        else:
            new_weights = self.DEFAULT_WEIGHTS.copy()

        # Blend with current weights (don't change too fast)
        blend = 0.3  # 30% new, 70% old
        for model in model_names:
            if model in self.weights and model in new_weights:
                self.weights[model] = (1 - blend) * self.weights[model] + blend * new_weights[model]

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        self._save_weights()
        return self.weights

    def _save_weights(self):
        """Save weights to file."""
        data = {
            'weights': self.weights,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'n_performance_records': len(self.performance_history)
        }
        os.makedirs(os.path.dirname(MODEL_WEIGHTS_FILE), exist_ok=True)
        with open(MODEL_WEIGHTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary(self):
        """Get human-readable weight summary."""
        lines = []
        for model, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            bar = '#' * int(weight * 50)
            lines.append(f"    {model:<20} {weight:.3f} {bar}")
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════
# MAIN AUTO-UPDATE PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_auto_update():
    """Full self-learning auto-update pipeline."""
    print("=" * 70)
    print("TOTO SELF-LEARNING AUTO-UPDATE PIPELINE v3.0")
    print("=" * 70)
    print(f"\n[SCHEDULE]")
    print(DrawCalendar.format_schedule())

    # Step 1: Load current dataset
    print(f"\n{'-' * 50}")
    print("[STEP 1] Loading current dataset...")
    df = pd.read_csv(CSV_PATH)
    last_date = df['date'].max()
    print(f"  Current: {len(df)} draws, last: {last_date}")

    # Step 2: Check for missing draws
    print(f"\n{'-' * 50}")
    print("[STEP 2] Checking for missing draws...")
    missing_dates = DrawCalendar.get_missing_draw_dates(last_date)

    if not missing_dates:
        print("  Dataset is up to date! No missing draws.")
    else:
        print(f"  Found {len(missing_dates)} missing draw date(s):")
        for md in missing_dates:
            print(f"    {md.strftime('%Y-%m-%d (%A)')}")

        # Step 3: Fetch missing draws
        print(f"\n{'-' * 50}")
        print("[STEP 3] Fetching latest results from Lottolyzer...")
        fetched = LottolyzerFetcher.fetch_latest_draws(n_pages=2)

        if fetched:
            print(f"  Fetched {len(fetched)} draws from Lottolyzer")

            # Filter to only missing dates
            new_rows = []
            existing_draw_nums = set(df['draw_number'].values)
            existing_dates = set(df['date'].values)

            for draw in fetched:
                if (draw['draw_number'] not in existing_draw_nums and
                    draw['date'] not in existing_dates):
                    # Validate
                    nums = draw['numbers']
                    if (len(nums) == 6 and len(set(nums)) == 6 and
                        all(1 <= n <= 49 for n in nums) and
                        1 <= draw['additional'] <= 49):

                        dt = datetime.strptime(draw['date'], '%Y-%m-%d')
                        new_rows.append({
                            'draw_number': draw['draw_number'],
                            'date': draw['date'],
                            'day_of_week': dt.strftime('%A'),
                            'num1': nums[0], 'num2': nums[1], 'num3': nums[2],
                            'num4': nums[3], 'num5': nums[4], 'num6': nums[5],
                            'additional_number': draw['additional'],
                            'group1_prize': 0,
                            'group1_winners': 0,
                            'is_synthetic': False
                        })

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                df = pd.concat([df, new_df], ignore_index=True)
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                df = df.sort_values('date').reset_index(drop=True)
                df.to_csv(CSV_PATH, index=False)
                print(f"  Added {len(new_rows)} new draw(s) to dataset")
                print(f"  Dataset now: {len(df)} draws")
            else:
                print("  No new draws to add (all already in dataset)")
        else:
            print("  Could not fetch from Lottolyzer. Data unchanged.")

    # Step 4: Score previous predictions
    print(f"\n{'-' * 50}")
    print("[STEP 4] Scoring previous predictions...")
    adaptive = AdaptiveWeights()

    if os.path.exists(PREDICTIONS_LOG):
        try:
            pred_log = pd.read_csv(PREDICTIONS_LOG)
            if len(pred_log) > 0:
                last_pred = pred_log.iloc[-1]
                target_date = str(last_pred.get('target_draw_date', ''))

                # Find actual result for that date
                actual_row = df[df['date'] == target_date]
                if len(actual_row) > 0:
                    actual = set([actual_row.iloc[0][f'num{i}'] for i in range(1, 7)])
                    print(f"  Scoring predictions for {target_date}:")
                    print(f"  Actual winning: {sorted(actual)}")

                    model_scores = {}
                    for col in ['board1', 'board2', 'board3',
                               'quant_board1', 'quant_board2', 'quant_board3']:
                        if col in last_pred and pd.notna(last_pred[col]):
                            try:
                                pred_nums = eval(str(last_pred[col]))
                                matches = len(set(pred_nums) & actual)
                                model_name = 'quant_engine' if 'quant' in col else 'ensemble_legacy'
                                model_scores[col] = matches
                                print(f"    {col}: {matches}/6 matches")

                                # Best match per model type
                                if model_name not in model_scores or matches > model_scores.get(model_name, 0):
                                    model_scores[model_name] = matches
                            except Exception:
                                pass

                    if model_scores:
                        adaptive.record_performance(target_date, model_scores)
                        print("  Performance recorded!")
                else:
                    print(f"  Target draw {target_date} not yet in dataset. Skipping scoring.")
            else:
                print("  No previous predictions to score.")
        except Exception as e:
            print(f"  Error reading prediction log: {e}")
    else:
        print("  No prediction log found. Starting fresh.")

    # Step 5: Adapt model weights
    print(f"\n{'-' * 50}")
    print("[STEP 5] Adapting model weights...")
    weights = adaptive.update_weights()
    print(f"  Current adaptive weights:")
    print(adaptive.get_summary())

    # Step 6: Retrain models and generate predictions
    print(f"\n{'-' * 50}")
    print("[STEP 6] Retraining models on updated data...")

    # Reload data (may have been updated)
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])

    # Run legacy models
    from src.models.weighted_scoring import predict as ws_predict
    from src.models.monte_carlo import predict as mc_predict
    from src.models.markov_chain import predict as mk_predict
    from src.models.ensemble import predict as ensemble_predict
    from src.predictor import generate_all_boards

    print("  Running Weighted Scoring...")
    ws = ws_predict(df)
    print("  Running Monte Carlo (1M simulations)...")
    mc = mc_predict(df)
    print("  Running Markov Chain...")
    mk = mk_predict(df)
    print("  Running Legacy Ensemble...")
    ens = ensemble_predict(df, {"weighted_scoring": ws, "monte_carlo": mc, "markov_chain": mk})
    legacy_boards = generate_all_boards(ens["rankings"], df)

    # Run Quant Engine v3.0
    print("  Running Quant Engine v3.0 (Bayesian + Pair Network + Regime)...")
    from src.models.quant_engine_v3 import QuantEngineV3
    qp = QuantEngineV3(df)
    qp.analyze()
    quant_boards = qp.generate_all_boards()

    # Step 7: Log predictions
    print(f"\n{'-' * 50}")
    print("[STEP 7] Logging predictions...")

    next_draw = DrawCalendar.get_next_draw_date()
    log_entry = {
        'date_predicted': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'target_draw_date': next_draw.strftime('%Y-%m-%d'),
        'target_draw_day': next_draw.strftime('%A'),
        'dataset_size': len(df),
        'board1': str(legacy_boards['boards'][0]['numbers']),
        'board2': str(legacy_boards['boards'][1]['numbers']),
        'board3': str(legacy_boards['boards'][2]['numbers']),
        'additional1': legacy_boards['boards'][0].get('additional_number'),
        'additional2': legacy_boards['boards'][1].get('additional_number'),
        'additional3': legacy_boards['boards'][2].get('additional_number'),
        'quant_board1': str(quant_boards['boards'][0]['numbers']),
        'quant_board2': str(quant_boards['boards'][1]['numbers']),
        'quant_board3': str(quant_boards['boards'][2]['numbers']),
        'quant_additional': quant_boards['additional_number'],
        'quant_coverage': quant_boards['coverage']['coverage_pct'],
        'model_weights': json.dumps(weights),
    }

    if os.path.exists(PREDICTIONS_LOG):
        log_df = pd.read_csv(PREDICTIONS_LOG)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_entry])
    log_df.to_csv(PREDICTIONS_LOG, index=False)

    # Step 8: Print results
    print(f"\n{'=' * 70}")
    print(f"  PREDICTIONS FOR NEXT DRAW: {next_draw.strftime('%Y-%m-%d (%A)')}")
    print(f"{'=' * 70}")

    print(f"\n  QUANT ENGINE v2.0 (Expected Value Optimized):")
    for b in quant_boards['boards']:
        nums = ', '.join(str(n) for n in b['numbers'])
        ev = b['expected_value']
        pop_label = ('UNPOPULAR [+EV]' if ev['popularity_ratio'] < 0.8
                     else 'Average' if ev['popularity_ratio'] < 1.2
                     else 'Popular [-EV]')
        print(f"    Board {b['board_number']} [{b['strategy']}]: {nums}")
        print(f"      Popularity: {ev['popularity_ratio']:.3f} ({pop_label}) | "
              f"Expected prize: ${ev['expected_prize_if_win']:,.0f}")

    print(f"\n  LEGACY ENSEMBLE:")
    for b in legacy_boards['boards']:
        nums = ', '.join(str(n) for n in b['numbers'])
        print(f"    {b['name']}: {nums}")
        print(f"      Confidence: {b['filter_result']['confidence']} | "
              f"Sum: {b['stats']['sum']}")

    print(f"\n  Additional number: {quant_boards['additional_number']}")
    cov = quant_boards['coverage']
    print(f"  Coverage: {cov['unique_numbers']}/49 ({cov['coverage_pct']}%)")

    print(f"\n{'=' * 70}")
    print(f"  Dataset: {len(df)} real draws")
    print(f"  Last draw: {df['date'].max()}")
    print(f"  Auto-update: ENABLED (GitHub Actions runs Mon & Thu)")
    print(f"  Self-learning: {'ACTIVE' if len(adaptive.performance_history) >= 5 else 'WARMING UP'} "
          f"({len(adaptive.performance_history)} performance records)")
    print(f"{'=' * 70}")

    return {
        'success': True,
        'draws_added': len(df) - 1050,  # rough count of new draws
        'next_draw': next_draw.strftime('%Y-%m-%d'),
        'quant_boards': quant_boards,
        'legacy_boards': legacy_boards,
        'adaptive_weights': weights
    }


if __name__ == '__main__':
    run_auto_update()
