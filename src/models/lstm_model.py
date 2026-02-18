"""
LSTM / Neural Network Model for Singapore TOTO Prediction

Attempts to use TensorFlow LSTM if available; falls back to sklearn's
MLPClassifier (multi-layer perceptron) which provides a neural network
approach without the TensorFlow dependency.

Prepares sequences of past N draws as features. For each of the 49 numbers,
a binary classification is performed (will this number appear in the next draw?).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]
ALL_NUMBERS = list(range(1, 50))

# Sequence length: how many past draws to use as features
SEQUENCE_LENGTH = 15


def _check_tensorflow():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
        print(f"  [LSTM] TensorFlow {tf.__version__} detected.")
        return True
    except ImportError:
        return False


def _extract_draw_numbers(row):
    """Extract the 6 main numbers from a draw row."""
    return [int(row[c]) for c in NUM_COLS]


def _build_binary_matrix(df):
    """
    Convert draws into a binary matrix of shape (n_draws, 49)
    where entry [i, j] = 1 if number j+1 appeared in draw i.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n_draws = len(df)
    matrix = np.zeros((n_draws, 49), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        nums = _extract_draw_numbers(row)
        for n in nums:
            matrix[i, n - 1] = 1.0

    return matrix


def _build_derived_features(df):
    """
    Build additional per-draw features:
    - draw sum, odd count, high count, spread (max - min)
    - day of week encoded
    """
    df = df.sort_values("date").reset_index(drop=True)
    features = []

    day_map = {"Monday": 0, "Thursday": 1, "Tuesday": 2, "Wednesday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}

    for _, row in df.iterrows():
        nums = _extract_draw_numbers(row)
        draw_sum = sum(nums) / 300.0  # normalize
        odd_count = sum(1 for n in nums if n % 2 == 1) / 6.0
        high_count = sum(1 for n in nums if n >= 25) / 6.0
        spread = (max(nums) - min(nums)) / 48.0
        dow = day_map.get(str(row.get("day_of_week", "Monday")), 0) / 6.0
        features.append([draw_sum, odd_count, high_count, spread, dow])

    return np.array(features, dtype=np.float32)


def _prepare_sequences(binary_matrix, derived_features, seq_len=SEQUENCE_LENGTH):
    """
    Prepare sequences for the neural network.
    For each timestep t, the input is the flattened sequence of binary vectors
    and derived features from draws [t-seq_len, ..., t-1].
    The target is the binary vector at draw t.
    """
    n_draws = binary_matrix.shape[0]
    n_numbers = binary_matrix.shape[1]
    n_derived = derived_features.shape[1]

    X_sequences = []
    y_targets = []

    for i in range(seq_len, n_draws):
        # Flatten the sequence window: (seq_len, 49 + n_derived) -> flat vector
        seq_binary = binary_matrix[i - seq_len:i]  # (seq_len, 49)
        seq_derived = derived_features[i - seq_len:i]  # (seq_len, n_derived)
        seq_combined = np.concatenate([seq_binary, seq_derived], axis=1)  # (seq_len, 54)

        # Flatten for MLP (for LSTM we'd keep the 3D shape)
        X_sequences.append(seq_combined.flatten())
        y_targets.append(binary_matrix[i])

    return np.array(X_sequences), np.array(y_targets)


def _prepare_sequences_3d(binary_matrix, derived_features, seq_len=SEQUENCE_LENGTH):
    """
    Prepare 3D sequences for LSTM: (n_samples, seq_len, features).
    """
    n_draws = binary_matrix.shape[0]

    X_sequences = []
    y_targets = []

    for i in range(seq_len, n_draws):
        seq_binary = binary_matrix[i - seq_len:i]
        seq_derived = derived_features[i - seq_len:i]
        seq_combined = np.concatenate([seq_binary, seq_derived], axis=1)
        X_sequences.append(seq_combined)
        y_targets.append(binary_matrix[i])

    return np.array(X_sequences), np.array(y_targets)


def _train_lstm(X_train, y_train, X_pred):
    """Train a TensorFlow LSTM model and predict."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping

    print("  [LSTM] Building TensorFlow LSTM model...")

    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),
             return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(49, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Split last 10% for validation
    val_split = int(0.9 * len(X_train))
    X_t, X_v = X_train[:val_split], X_train[val_split:]
    y_t, y_v = y_train[:val_split], y_train[val_split:]

    print("  [LSTM] Training LSTM model...")
    model.fit(
        X_t, y_t,
        validation_data=(X_v, y_v),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    val_loss = model.evaluate(X_v, y_v, verbose=0)
    print(f"  [LSTM] Validation loss: {val_loss[0]:.4f}, accuracy: {val_loss[1]:.4f}")

    # Predict
    predictions = model.predict(X_pred, verbose=0)
    return predictions[0]  # shape (49,)


def _train_mlp(X_train, y_train, X_pred):
    """Train sklearn MLPClassifier as LSTM fallback."""
    print("  [LSTM/MLP] Training MLPClassifier (neural network fallback)...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Train separate MLP for each number (binary classification)
    # But for efficiency, we can train a single multi-output approach
    # by training one MLP per number in batches

    probabilities = np.zeros(49)

    # Group numbers into batches for progress reporting
    batch_size = 10
    for batch_start in range(0, 49, batch_size):
        batch_end = min(batch_start + batch_size, 49)
        batch_nums = list(range(batch_start, batch_end))

        for num_idx in batch_nums:
            y_num = y_train[:, num_idx]

            # Check if we have both classes
            if len(np.unique(y_num)) < 2:
                probabilities[num_idx] = np.mean(y_num)
                continue

            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42,
                learning_rate="adaptive",
                learning_rate_init=0.001,
                batch_size=min(64, len(X_train_scaled)),
            )

            try:
                mlp.fit(X_train_scaled, y_num)
                prob = mlp.predict_proba(X_pred_scaled)[0, 1]
                probabilities[num_idx] = prob
            except Exception:
                probabilities[num_idx] = np.mean(y_num)

        print(f"    Completed numbers {batch_start + 1}-{batch_end} of 49")

    return probabilities


def predict(df):
    """
    Train LSTM (or MLP fallback) and predict next draw probabilities.

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
    print("LSTM / NEURAL NETWORK MODEL")
    print("=" * 60)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Total draws: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")

    # Build binary matrix and derived features
    print("  [LSTM] Building binary matrix and derived features...")
    binary_matrix = _build_binary_matrix(df)
    derived_features = _build_derived_features(df)

    use_tensorflow = _check_tensorflow()

    if use_tensorflow:
        print("  [LSTM] Using TensorFlow LSTM backend.")
        X_3d, y_targets = _prepare_sequences_3d(binary_matrix, derived_features)
        print(f"  [LSTM] Sequence data shape: {X_3d.shape}")

        # Train on all but predict the last sequence
        X_train = X_3d[:-1]
        y_train = y_targets[:-1]
        X_pred = X_3d[-1:]

        probabilities = _train_lstm(X_train, y_train, X_pred)
    else:
        print("  [LSTM] TensorFlow not available. Using MLPClassifier fallback.")
        X_flat, y_targets = _prepare_sequences(binary_matrix, derived_features)
        print(f"  [LSTM/MLP] Flattened feature matrix shape: {X_flat.shape}")

        # Train on all but predict the last sequence
        X_train = X_flat[:-1]
        y_train = y_targets[:-1]
        X_pred = X_flat[-1:]

        probabilities = _train_mlp(X_train, y_train, X_pred)

    # Build rankings from probabilities
    rankings_dict = {n: probabilities[n - 1] for n in ALL_NUMBERS}
    rankings = sorted(rankings_dict.items(), key=lambda x: x[1], reverse=True)
    top_numbers = sorted([num for num, _ in rankings[:6]])

    # Compute validation score on last 10% of data
    n_val = max(int(len(y_targets) * 0.1), 5)
    val_y = y_targets[-n_val:]
    val_aucs = []
    for num_idx in range(49):
        y_col = val_y[:, num_idx]
        if len(np.unique(y_col)) >= 2:
            # Use the training probability as a rough estimate
            pred_col = np.full(len(y_col), probabilities[num_idx])
            try:
                auc = roc_auc_score(y_col, pred_col)
                val_aucs.append(auc)
            except ValueError:
                pass

    mean_val_auc = np.mean(val_aucs) if val_aucs else 0.5

    backend = "TensorFlow LSTM" if use_tensorflow else "MLPClassifier"
    print(f"\n  Backend: {backend}")
    print(f"  Approx. validation AUC: {mean_val_auc:.4f}")
    print(f"  Top 6 numbers: {top_numbers}")
    print(f"  Top 10 rankings:")
    for i, (num, prob) in enumerate(rankings[:10]):
        print(f"    {i+1:2d}. Number {num:2d} -> probability {prob:.4f}")
    print("=" * 60)

    return {
        "rankings": rankings,
        "top_numbers": top_numbers,
        "model_name": "LSTM" if use_tensorflow else "MLP_Neural_Network",
        "backend": backend,
        "validation_auc": mean_val_auc,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.scraper import load_data

    df = load_data()
    result = predict(df)
    print(f"\nFinal top 6: {result['top_numbers']}")
