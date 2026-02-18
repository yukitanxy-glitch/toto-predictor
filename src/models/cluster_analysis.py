"""
Cluster Analysis Model for Singapore TOTO Prediction

Represents each historical draw as a feature vector (6 main numbers + derived
features like sum, spread, odd/even count, high/low count, consecutive count,
decade distribution). Applies K-Means clustering with silhouette score
optimization to find the optimal number of clusters.

Prediction: identifies which cluster the recent draws belong to, then
scores numbers by their prevalence within that cluster and neighboring clusters.
"""

import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]
ALL_NUMBERS = list(range(1, 50))
TOTAL_NUMBERS = 49


def _extract_draw_numbers(row):
    """Extract the 6 main numbers from a draw row."""
    return [int(row[c]) for c in NUM_COLS]


def _build_draw_features(df):
    """
    Build a feature matrix where each row represents a draw with:
    - The 6 sorted numbers (num1-num6)
    - Sum of numbers
    - Spread (max - min)
    - Mean of numbers
    - Standard deviation
    - Odd count (out of 6)
    - High count (numbers >= 25)
    - Consecutive pair count
    - Decade distribution (5 features: count in each decade 1-9, 10-19, ..., 40-49)
    - Gap between consecutive numbers (5 features: num2-num1, num3-num2, ...)
    """
    df = df.sort_values("date").reset_index(drop=True)
    feature_rows = []

    for _, row in df.iterrows():
        nums = sorted(_extract_draw_numbers(row))

        # Base numbers (normalized to 0-1)
        base = [n / 49.0 for n in nums]

        # Derived features
        draw_sum = sum(nums) / 300.0
        spread = (nums[-1] - nums[0]) / 48.0
        mean_val = np.mean(nums) / 49.0
        std_val = np.std(nums) / 20.0
        odd_count = sum(1 for n in nums if n % 2 == 1) / 6.0
        high_count = sum(1 for n in nums if n >= 25) / 6.0

        # Consecutive pairs (e.g., 5,6 or 23,24)
        consecutive = sum(1 for j in range(len(nums) - 1) if nums[j + 1] - nums[j] == 1)
        consecutive_norm = consecutive / 5.0

        # Decade distribution
        decades = [0] * 5
        for n in nums:
            decade_idx = min((n - 1) // 10, 4)
            decades[decade_idx] += 1
        decades_norm = [d / 6.0 for d in decades]

        # Gaps between consecutive sorted numbers
        gaps = [(nums[j + 1] - nums[j]) / 48.0 for j in range(5)]

        feature_vec = (
            base +
            [draw_sum, spread, mean_val, std_val, odd_count, high_count, consecutive_norm] +
            decades_norm +
            gaps
        )
        feature_rows.append(feature_vec)

    feature_names = (
        [f"num{i}_norm" for i in range(1, 7)] +
        ["sum", "spread", "mean", "std", "odd_count", "high_count", "consecutive"] +
        [f"decade_{i}" for i in range(5)] +
        [f"gap_{i}" for i in range(1, 6)]
    )

    return np.array(feature_rows, dtype=np.float64), feature_names


def _optimize_k(X_scaled, k_range=(3, 15)):
    """
    Find optimal number of clusters using silhouette score.
    """
    print("  [Cluster] Optimizing number of clusters (k)...")
    best_k = k_range[0]
    best_score = -1
    results = []

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        results.append((k, score))
        print(f"    k={k:2d}: silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"  [Cluster] Optimal k={best_k} with silhouette={best_score:.4f}")
    return best_k, best_score, results


def _build_cluster_profiles(df, labels, n_clusters):
    """
    Build profile for each cluster: which numbers are most frequent,
    average stats, etc.
    """
    df = df.sort_values("date").reset_index(drop=True)
    profiles = {}

    for c in range(n_clusters):
        cluster_mask = labels == c
        cluster_df = df.iloc[cluster_mask]
        cluster_size = len(cluster_df)

        if cluster_size == 0:
            profiles[c] = {"size": 0, "number_freq": {}, "avg_sum": 0, "description": "empty"}
            continue

        # Number frequency within cluster
        num_counts = Counter()
        sums = []
        odd_counts = []
        high_counts = []

        for _, row in cluster_df.iterrows():
            nums = _extract_draw_numbers(row)
            for n in nums:
                num_counts[n] += 1
            sums.append(sum(nums))
            odd_counts.append(sum(1 for n in nums if n % 2 == 1))
            high_counts.append(sum(1 for n in nums if n >= 25))

        # Normalize to frequency
        num_freq = {n: num_counts.get(n, 0) / cluster_size for n in ALL_NUMBERS}

        avg_sum = np.mean(sums)
        avg_odd = np.mean(odd_counts)
        avg_high = np.mean(high_counts)

        # Describe the cluster
        top_nums = sorted(num_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nums_str = ", ".join([str(n) for n, _ in top_nums])

        if avg_sum < 130:
            sum_desc = "low-sum"
        elif avg_sum > 170:
            sum_desc = "high-sum"
        else:
            sum_desc = "mid-sum"

        description = f"{sum_desc}, avg_odd={avg_odd:.1f}, avg_high={avg_high:.1f}, top: [{top_nums_str}]"

        profiles[c] = {
            "size": cluster_size,
            "fraction": cluster_size / len(df),
            "number_freq": num_freq,
            "avg_sum": avg_sum,
            "avg_odd": avg_odd,
            "avg_high": avg_high,
            "description": description,
            "top_numbers": [n for n, _ in top_nums],
        }

    return profiles


def get_cluster_profiles(df):
    """
    Public function to get cluster profiles without prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Historical TOTO data.

    Returns
    -------
    dict with cluster profiles and metadata.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build features
    X, feature_names = _build_draw_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optimize k
    best_k, best_score, _ = _optimize_k(X_scaled)

    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    # Build profiles
    profiles = _build_cluster_profiles(df, labels, best_k)

    return {
        "n_clusters": best_k,
        "silhouette_score": best_score,
        "profiles": profiles,
        "labels": labels.tolist(),
        "centroids": kmeans.cluster_centers_.tolist(),
    }


def predict(df):
    """
    Run cluster analysis and predict based on recent cluster assignment.

    Parameters
    ----------
    df : pd.DataFrame
        Historical TOTO data.

    Returns
    -------
    dict with:
        'rankings': list of (number, score) sorted by score descending
        'top_numbers': list of top 6 numbers
    """
    print("\n" + "=" * 60)
    print("CLUSTER ANALYSIS MODEL")
    print("=" * 60)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Total draws: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Build feature matrix
    print("  [Cluster] Building draw feature matrix...")
    X, feature_names = _build_draw_features(df)
    print(f"  [Cluster] Feature matrix shape: {X.shape} ({len(feature_names)} features)")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optimize k
    best_k, best_score, k_results = _optimize_k(X_scaled)

    # Final clustering
    print(f"  [Cluster] Running KMeans with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    # Build cluster profiles
    print("  [Cluster] Building cluster profiles...")
    profiles = _build_cluster_profiles(df, labels, best_k)

    # Print cluster summaries
    for c in range(best_k):
        p = profiles[c]
        print(f"    Cluster {c}: {p['size']} draws ({p['fraction']*100:.1f}%) - {p['description']}")

    # Identify which cluster recent draws belong to
    n_recent = min(20, len(df))
    recent_labels = labels[-n_recent:]
    recent_cluster_counts = Counter(recent_labels)
    dominant_cluster = recent_cluster_counts.most_common(1)[0][0]

    print(f"\n  [Cluster] Recent {n_recent} draws cluster distribution: {dict(recent_cluster_counts)}")
    print(f"  [Cluster] Dominant recent cluster: {dominant_cluster}")

    # Score numbers based on:
    # 1. Frequency in dominant cluster (60%)
    # 2. Frequency in neighboring clusters by centroid distance (25%)
    # 3. Overall frequency across all clusters (15%)

    dominant_freq = profiles[dominant_cluster]["number_freq"]

    # Compute centroid distances from dominant cluster
    dominant_centroid = kmeans.cluster_centers_[dominant_cluster]
    cluster_distances = {}
    for c in range(best_k):
        if c != dominant_cluster:
            dist = np.linalg.norm(dominant_centroid - kmeans.cluster_centers_[c])
            cluster_distances[c] = dist

    # Weighted neighbor frequency
    neighbor_freq = {n: 0.0 for n in ALL_NUMBERS}
    if cluster_distances:
        max_dist = max(cluster_distances.values())
        for c, dist in cluster_distances.items():
            weight = 1.0 - (dist / (max_dist + 1e-6))  # closer = higher weight
            for n in ALL_NUMBERS:
                neighbor_freq[n] += weight * profiles[c]["number_freq"].get(n, 0)

        # Normalize
        total_weight = sum(1.0 - (d / (max_dist + 1e-6)) for d in cluster_distances.values())
        if total_weight > 0:
            neighbor_freq = {n: v / total_weight for n, v in neighbor_freq.items()}

    # Overall frequency
    overall_freq = {n: 0.0 for n in ALL_NUMBERS}
    total_draws = len(df)
    for _, row in df.iterrows():
        for n in _extract_draw_numbers(row):
            overall_freq[n] += 1
    overall_freq = {n: v / total_draws for n, v in overall_freq.items()}

    # Combine scores
    print("  [Cluster] Computing final scores (60% dominant, 25% neighbor, 15% overall)...")
    final_scores = {}
    for n in ALL_NUMBERS:
        score = (
            0.60 * dominant_freq.get(n, 0) +
            0.25 * neighbor_freq.get(n, 0) +
            0.15 * overall_freq.get(n, 0)
        )
        final_scores[n] = score

    # Build rankings
    rankings = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_numbers = sorted([num for num, _ in rankings[:6]])

    # PCA for visualization info
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_.sum()

    print(f"\n  PCA explained variance (2D): {explained_variance:.2%}")
    print(f"  Top 6 numbers: {top_numbers}")
    print(f"  Top 10 rankings:")
    for i, (num, score) in enumerate(rankings[:10]):
        dom = dominant_freq.get(num, 0)
        nbr = neighbor_freq.get(num, 0)
        ovr = overall_freq.get(num, 0)
        print(f"    {i+1:2d}. Number {num:2d} -> score {score:.4f} "
              f"(dom: {dom:.3f}, nbr: {nbr:.3f}, ovr: {ovr:.3f})")
    print("=" * 60)

    return {
        "rankings": rankings,
        "top_numbers": top_numbers,
        "model_name": "ClusterAnalysis",
        "n_clusters": best_k,
        "silhouette_score": best_score,
        "dominant_cluster": int(dominant_cluster),
        "cluster_profiles": profiles,
        "pca_explained_variance": explained_variance,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.scraper import load_data

    df = load_data()

    # Run prediction
    result = predict(df)
    print(f"\nFinal top 6: {result['top_numbers']}")

    # Also show cluster profiles
    print("\n--- Detailed Cluster Profiles ---")
    profile_result = get_cluster_profiles(df)
    for c, p in profile_result["profiles"].items():
        print(f"\nCluster {c} ({p['size']} draws):")
        print(f"  Description: {p['description']}")
        if p["size"] > 0:
            print(f"  Top numbers: {p.get('top_numbers', [])[:6]}")
