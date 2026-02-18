"""
Singapore TOTO Historical Data Scraper

Attempts to collect historical TOTO results from multiple sources.
Falls back to generating realistic synthetic data for gaps.
"""
import os
import random
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "toto_results.csv")


def scrape_singapore_pools(start_year=2016, end_year=2026):
    """Try to scrape from Singapore Pools website via their results API."""
    all_results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/html, */*",
        "Referer": "https://www.singaporepools.com.sg/en/product/pages/toto_results.aspx",
    }

    # Singapore Pools has a search page that returns past results
    base_url = "https://www.singaporepools.com.sg/en/product/pages/toto_results.aspx"

    try:
        session = requests.Session()
        resp = session.get(base_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            # Try to parse results from the page
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 7:
                        try:
                            nums = [int(c.text.strip()) for c in cells[:7]]
                            if all(1 <= n <= 49 for n in nums):
                                all_results.append(nums)
                        except (ValueError, IndexError):
                            continue
            print(f"[Scraper] Fetched {len(all_results)} results from Singapore Pools")
    except Exception as e:
        print(f"[Scraper] Singapore Pools scraping failed: {e}")

    return all_results


def try_fetch_kaggle_data():
    """Try to fetch the Kaggle Singapore lottery dataset."""
    # Direct download URL for the Kaggle dataset (public CC0 license)
    urls = [
        "https://raw.githubusercontent.com/calven22/singapore-lottery-numbers/main/toto.csv",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 100:
                # Try to parse as CSV
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                print(f"[Scraper] Fetched {len(df)} rows from {url}")
                return df
        except Exception as e:
            print(f"[Scraper] Failed to fetch {url}: {e}")
    return None


def generate_synthetic_data(start_date, end_date, existing_dates=None):
    """
    Generate realistic synthetic TOTO data for date ranges without real data.
    Uses statistical distributions matching real TOTO patterns.
    """
    if existing_dates is None:
        existing_dates = set()

    synthetic_rows = []
    current = start_date
    draw_number = 3000  # Starting synthetic draw number

    while current <= end_date:
        # TOTO draws on Monday (0) and Thursday (3)
        if current.weekday() in (0, 3):
            if current not in existing_dates:
                # Generate 6 main numbers + 1 additional from 1-49
                # Use slightly non-uniform distribution to mimic real patterns
                weights = np.ones(49)
                # Slight bias: numbers in middle range appear slightly more
                for i in range(49):
                    num = i + 1
                    if 15 <= num <= 35:
                        weights[i] *= 1.05
                weights /= weights.sum()

                all_nums = np.random.choice(
                    range(1, 50), size=7, replace=False, p=weights
                )
                main_nums = sorted(all_nums[:6])
                additional = all_nums[6]

                # Generate realistic jackpot amounts
                group1_prize = random.choice([
                    1000000, 1500000, 2000000, 2500000, 3000000,
                    3500000, 4000000, 5000000, 6000000, 7000000,
                    8000000, 10000000, 12000000,
                ])
                group1_winners = random.choices([0, 1, 2, 3], weights=[60, 30, 8, 2])[0]

                day_name = "Monday" if current.weekday() == 0 else "Thursday"
                synthetic_rows.append({
                    "draw_number": draw_number,
                    "date": current.strftime("%Y-%m-%d"),
                    "day_of_week": day_name,
                    "num1": main_nums[0],
                    "num2": main_nums[1],
                    "num3": main_nums[2],
                    "num4": main_nums[3],
                    "num5": main_nums[4],
                    "num6": main_nums[5],
                    "additional_number": int(additional),
                    "group1_prize": group1_prize,
                    "group1_winners": group1_winners,
                    "is_synthetic": True,
                })
                draw_number += 1
        current += timedelta(days=1)

    return pd.DataFrame(synthetic_rows)


def collect_all_data():
    """
    Main collection function. Tries real sources first, fills gaps with synthetic.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    real_data = []
    real_dates = set()

    # --- Attempt 1: Try Kaggle / GitHub datasets ---
    kaggle_df = try_fetch_kaggle_data()
    if kaggle_df is not None and len(kaggle_df) > 0:
        # Normalize column names
        cols = kaggle_df.columns.str.lower().str.strip()
        kaggle_df.columns = cols
        # Try to map to our schema
        try:
            normalized = _normalize_external_df(kaggle_df)
            if normalized is not None and len(normalized) > 0:
                real_data.append(normalized)
                for d in normalized["date"]:
                    real_dates.add(pd.Timestamp(d).date())
        except Exception as e:
            print(f"[Scraper] Failed to normalize Kaggle data: {e}")

    # --- Attempt 2: Try Singapore Pools ---
    sp_results = scrape_singapore_pools()
    # (Singapore Pools typically blocks automated scraping, so this may be empty)

    # --- Build dataset ---
    start_date = datetime(2016, 1, 1).date()
    end_date = datetime(2026, 2, 17).date()  # Up to recent date

    if real_data:
        combined_real = pd.concat(real_data, ignore_index=True)
        combined_real["is_synthetic"] = False
        print(f"[Scraper] Total real data rows: {len(combined_real)}")
    else:
        combined_real = pd.DataFrame()
        print("[Scraper] No real data obtained from any source.")

    # Generate synthetic data for gaps
    print("[Scraper] Generating synthetic data for missing date ranges...")
    synthetic_df = generate_synthetic_data(start_date, end_date, real_dates)

    if len(combined_real) > 0:
        final_df = pd.concat([combined_real, synthetic_df], ignore_index=True)
    else:
        final_df = synthetic_df
        warnings.warn(
            "WARNING: Entire dataset is synthetic! No real TOTO data was obtained. "
            "Results are for demonstration purposes only."
        )

    # Sort by date
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df = final_df.sort_values("date").reset_index(drop=True)

    # Reassign draw numbers sequentially
    final_df["draw_number"] = range(3000, 3000 + len(final_df))

    # Save
    final_df.to_csv(CSV_PATH, index=False)

    # Print summary
    total = len(final_df)
    real_count = len(final_df[~final_df["is_synthetic"]])
    synth_count = len(final_df[final_df["is_synthetic"]])
    date_min = final_df["date"].min().strftime("%Y-%m-%d")
    date_max = final_df["date"].max().strftime("%Y-%m-%d")

    print("\n" + "=" * 50)
    print("TOTO DATA COLLECTION SUMMARY")
    print("=" * 50)
    print(f"Total draws collected: {total}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Real data: {real_count} ({100*real_count/total:.1f}%)")
    print(f"Synthetic data: {synth_count} ({100*synth_count/total:.1f}%)")
    if synth_count > 0:
        print("\n[!] WARNING: Synthetic data is labeled with is_synthetic=True")
        print("    Synthetic data follows correct TOTO format but is randomly generated.")
    print("=" * 50)

    return final_df


def _normalize_external_df(df):
    """Try to normalize an external DataFrame to our schema."""
    result = pd.DataFrame()

    # Try to find date column
    date_col = None
    for c in df.columns:
        if "date" in c or "draw" in c and "date" in c:
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c].head())
                date_col = c
                break
            except Exception:
                continue

    if date_col is None:
        return None

    result["date"] = pd.to_datetime(df[date_col])
    result["day_of_week"] = result["date"].dt.day_name()

    # Try to find number columns
    num_cols = []
    for c in df.columns:
        if c in ("num1", "num2", "num3", "num4", "num5", "num6",
                  "n1", "n2", "n3", "n4", "n5", "n6",
                  "number1", "number2", "number3", "number4", "number5", "number6",
                  "winning1", "winning2", "winning3", "winning4", "winning5", "winning6"):
            num_cols.append(c)

    if len(num_cols) < 6:
        # Try numeric columns
        for c in df.columns:
            if df[c].dtype in (np.int64, np.float64, int, float):
                vals = df[c].dropna()
                if len(vals) > 0 and vals.min() >= 1 and vals.max() <= 49:
                    num_cols.append(c)
                    if len(num_cols) >= 7:
                        break

    if len(num_cols) < 6:
        return None

    for i, c in enumerate(num_cols[:6]):
        result[f"num{i+1}"] = df[c].astype(int)

    # Additional number
    if len(num_cols) >= 7:
        result["additional_number"] = df[num_cols[6]].astype(int)
    else:
        # Look for additional/bonus column
        for c in df.columns:
            if "additional" in c or "bonus" in c or "extra" in c:
                result["additional_number"] = df[c].astype(int)
                break
        if "additional_number" not in result.columns:
            result["additional_number"] = 0

    # Sort numbers within each row
    for idx in result.index:
        nums = sorted([result.loc[idx, f"num{i}"] for i in range(1, 7)])
        for i, n in enumerate(nums):
            result.loc[idx, f"num{i+1}"] = n

    # Draw number
    draw_col = None
    for c in df.columns:
        if "draw" in c and "number" in c or c == "draw_no" or c == "drawno":
            draw_col = c
            break
    if draw_col:
        result["draw_number"] = df[draw_col]
    else:
        result["draw_number"] = range(1, len(result) + 1)

    # Prize data if available
    for c in df.columns:
        if "prize" in c or "jackpot" in c:
            result["group1_prize"] = df[c]
            break
    if "group1_prize" not in result.columns:
        result["group1_prize"] = 0

    for c in df.columns:
        if "winner" in c:
            result["group1_winners"] = df[c]
            break
    if "group1_winners" not in result.columns:
        result["group1_winners"] = 0

    return result


def load_data():
    """Load the TOTO dataset. If not collected yet, run collection first."""
    if not os.path.exists(CSV_PATH):
        return collect_all_data()
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    collect_all_data()
