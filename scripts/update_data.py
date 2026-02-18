#!/usr/bin/env python3
"""
Data Update Script for Singapore TOTO Predictor

1. Fetches latest draw result
2. Validates and appends to dataset
3. Checks previous prediction accuracy
4. Re-runs analysis and prediction pipeline
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime

from src.scraper import load_data, CSV_PATH, DATA_DIR
from src.analysis import get_full_analysis
from src.models.weighted_scoring import predict as ws_predict
from src.models.monte_carlo import predict as mc_predict
from src.models.markov_chain import predict as mk_predict
from src.models.ensemble import predict as ensemble_predict
from src.predictor import generate_all_boards


def fetch_latest_result():
    """
    Fetch the most recent TOTO draw result.
    Returns a dict with draw data, or None if fetch fails.
    """
    import requests
    from bs4 import BeautifulSoup

    url = "https://www.singaporepools.com.sg/en/product/pages/toto_results.aspx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            # Parse the latest result from the page
            # Singapore Pools page structure may vary
            print("[Update] Fetched Singapore Pools page successfully")
            print("[Update] Note: Automated parsing may not work — manual entry may be needed")
    except Exception as e:
        print(f"[Update] Could not fetch latest result: {e}")

    return None


def manual_add_result():
    """Allow manual entry of latest draw result."""
    print("\n--- Manual Draw Entry ---")
    try:
        draw_num = int(input("Draw number: "))
        date_str = input("Date (YYYY-MM-DD): ")
        nums = input("6 winning numbers (comma-separated): ").strip()
        nums = [int(n.strip()) for n in nums.split(",")]
        additional = int(input("Additional number: "))

        if len(nums) != 6:
            print("Error: Need exactly 6 numbers")
            return None
        if not all(1 <= n <= 49 for n in nums):
            print("Error: Numbers must be 1-49")
            return None
        if len(set(nums)) != 6:
            print("Error: Numbers must be unique")
            return None
        if not (1 <= additional <= 49):
            print("Error: Additional number must be 1-49")
            return None
        if additional in nums:
            print("Error: Additional number must not be in winning numbers")
            return None

        date = pd.Timestamp(date_str)
        day_name = date.day_name()

        return {
            "draw_number": draw_num,
            "date": date_str,
            "day_of_week": day_name,
            "num1": sorted(nums)[0],
            "num2": sorted(nums)[1],
            "num3": sorted(nums)[2],
            "num4": sorted(nums)[3],
            "num5": sorted(nums)[4],
            "num6": sorted(nums)[5],
            "additional_number": additional,
            "group1_prize": 0,
            "group1_winners": 0,
            "is_synthetic": False,
        }
    except (ValueError, KeyboardInterrupt):
        print("Entry cancelled.")
        return None


def check_previous_prediction(df, new_result):
    """Check how previous prediction performed against actual result."""
    log_path = os.path.join(DATA_DIR, "predictions_log.csv")
    if not os.path.exists(log_path):
        print("[Update] No previous predictions to check.")
        return

    log_df = pd.read_csv(log_path)
    if len(log_df) == 0:
        return

    last_pred = log_df.iloc[-1]
    actual = [new_result[f"num{i}"] for i in range(1, 7)]

    print(f"\n--- Previous Prediction Accuracy ---")
    for board_key in ["board1", "board2", "board3"]:
        if board_key in last_pred and pd.notna(last_pred[board_key]):
            pred_nums = eval(str(last_pred[board_key]))
            matches = len(set(pred_nums) & set(actual))
            print(f"  {board_key}: {matches}/6 matches — {pred_nums} vs {actual}")


def update_pipeline():
    """Full update pipeline."""
    print("=" * 60)
    print("TOTO PREDICTOR — DATA UPDATE")
    print("=" * 60)

    # Load existing data
    df = load_data()
    print(f"Current dataset: {len(df)} draws")

    # Try to fetch latest
    new_result = fetch_latest_result()
    if new_result is None:
        choice = input("\nManually enter latest draw? (y/n): ").strip().lower()
        if choice == "y":
            new_result = manual_add_result()

    if new_result:
        # Check previous prediction
        check_previous_prediction(df, new_result)

        # Append to dataset
        new_row = pd.DataFrame([new_result])
        df = pd.concat([df, new_row], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n✓ Added draw #{new_result['draw_number']} to dataset")
        print(f"  Dataset now has {len(df)} draws")
    else:
        print("\nNo new data added. Running prediction on existing data.")

    # Re-run analysis
    print("\n--- Running Analysis ---")
    analysis = get_full_analysis(df)
    print("✓ Analysis complete")

    # Re-run predictions
    print("\n--- Running Prediction Models ---")
    ws = ws_predict(df)
    mc = mc_predict(df)
    mk = mk_predict(df)

    model_results = {
        "weighted_scoring": ws,
        "monte_carlo": mc,
        "markov_chain": mk,
    }
    ens = ensemble_predict(df, model_results)
    print("✓ Models complete")

    # Generate boards
    boards = generate_all_boards(ens["rankings"], df)

    # Log predictions
    log_path = os.path.join(DATA_DIR, "predictions_log.csv")
    log_entry = {
        "date_predicted": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target_draw_date": boards["next_draw"]["date"],
        "board1": str(boards["boards"][0]["numbers"]),
        "board2": str(boards["boards"][1]["numbers"]),
        "board3": str(boards["boards"][2]["numbers"]),
        "additional1": boards["boards"][0].get("additional_number"),
        "additional2": boards["boards"][1].get("additional_number"),
        "additional3": boards["boards"][2].get("additional_number"),
    }
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_entry])
    log_df.to_csv(log_path, index=False)

    # Print results
    print(f"\n{'='*60}")
    print(f"NEXT DRAW: {boards['next_draw']['date']} ({boards['next_draw']['day']})")
    print(f"{'='*60}")
    for b in boards["boards"]:
        print(f"\n{b['name']}:")
        print(f"  Numbers: {b['numbers']} + Additional: {b.get('additional_number', '?')}")
        print(f"  Confidence: {b['filter_result']['confidence']}")
        print(f"  Sum: {b['stats']['sum']} | Odd/Even: {b['stats']['odd_even']} | High/Low: {b['stats']['high_low']}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    update_pipeline()
