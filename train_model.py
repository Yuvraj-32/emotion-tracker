"""
==========================================================
  EMOTION TRACKER — ML PRODUCTIVITY MODEL TRAINER
==========================================================
  Reads all session CSVs, engineers features, and trains
  a Random Forest classifier to predict productivity level.

  Productivity levels (derived from engagement score):
    HIGH   = engagement >= 65
    MEDIUM = 35 <= engagement < 65
    LOW    = engagement < 35

  Usage:
    python train_model.py

  Output:
    models/productivity_model.pkl
==========================================================
"""

import os
import csv
import glob
import json
import numpy as np
from collections import Counter
from datetime import datetime

# ── Optional: check for sklearn ──
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
except ImportError:
    print("ERROR: scikit-learn is required. Install with:")
    print("  pip install scikit-learn joblib")
    exit(1)

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "productivity_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# ── Emotion encoding ──
EMOTION_ENCODING = {
    'Happy': 4, 'Surprised': 3, 'Neutral': 2,
    'Tired / Sad': 1, 'Frustrated': 0, 'Scanning...': 2,
}

# ── Productivity thresholds ──
HIGH_THRESHOLD = 65
LOW_THRESHOLD = 35


def load_all_sessions():
    """Load and combine all session CSVs."""
    pattern = os.path.join(LOG_DIR, "session_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("ERROR: No session CSV files found in logs/")
        print("Run the tracker first to generate data.")
        return [], []

    all_rows = []
    session_dates = []

    for fpath in files:
        # Extract date from filename: session_YYYY-MM-DD_HH-MM-SS.csv
        fname = os.path.basename(fpath)
        try:
            date_part = fname.replace("session_", "").replace(".csv", "")
            session_date = date_part[:10]  # YYYY-MM-DD
        except Exception:
            session_date = "unknown"

        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                row['_session_file'] = fname
                row['_session_date'] = session_date
            all_rows.extend(rows)
            if rows:
                session_dates.append(session_date)

    return all_rows, files


def engineer_features(rows):
    """Convert raw CSV rows into feature matrix + labels."""
    X = []
    y = []
    valid_rows = []

    for row in rows:
        try:
            # Parse numeric features
            engagement = float(row.get('engagement', 50))
            focus_dur = float(row.get('focus_duration_sec', 0))
            distraction_cnt = int(row.get('distraction_count', 0))
            drowsy_cnt = int(row.get('drowsy_count', 0))
            emotion_stab = float(row.get('emotion_stability', 0.5))
            session_elapsed = float(row.get('session_elapsed_min', 0))

            # Boolean features
            is_looking = 1 if row.get('is_looking', 'True') == 'True' else 0
            is_drowsy = 1 if row.get('is_drowsy', 'False') == 'True' else 0

            # Emotion encoding
            emotion = row.get('emotion', 'Neutral')
            emotion_score = EMOTION_ENCODING.get(emotion, 2)

            # Feature vector (9 features)
            features = [
                emotion_score,       # 0: emotion (encoded)
                is_looking,          # 1: attention flag
                is_drowsy,           # 2: drowsiness flag
                focus_dur,           # 3: focus streak duration
                distraction_cnt,     # 4: cumulative distractions
                drowsy_cnt,          # 5: cumulative drowsy events
                emotion_stab,        # 6: emotion stability
                session_elapsed,     # 7: session time
                engagement,          # 8: engagement score (used for label too)
            ]

            # Label: productivity level from engagement
            if engagement >= HIGH_THRESHOLD:
                label = 2  # HIGH
            elif engagement >= LOW_THRESHOLD:
                label = 1  # MEDIUM
            else:
                label = 0  # LOW

            X.append(features)
            y.append(label)
            valid_rows.append(row)

        except (ValueError, KeyError):
            continue

    return np.array(X), np.array(y), valid_rows


def train_model():
    """Main training pipeline."""
    print("=" * 55)
    print("  PRODUCTIVITY MODEL TRAINER")
    print("=" * 55)

    # Load data
    print("\n[1/5] Loading session data...")
    rows, files = load_all_sessions()
    if not rows:
        return

    print(f"  Found {len(files)} session file(s), {len(rows)} total data points")

    # Engineer features
    print("\n[2/5] Engineering features...")
    X, y, valid_rows = engineer_features(rows)

    if len(X) < 10:
        print(f"  ERROR: Only {len(X)} valid data points. Need at least 10.")
        print("  Run more tracking sessions to collect data.")
        return

    # Feature names for reporting
    feature_names = [
        'emotion_score', 'is_looking', 'is_drowsy',
        'focus_duration', 'distraction_count', 'drowsy_count',
        'emotion_stability', 'session_elapsed', 'engagement'
    ]

    label_names = ['LOW', 'MEDIUM', 'HIGH']
    class_counts = Counter(y)
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}")
    for label_idx, name in enumerate(label_names):
        print(f"    {name}: {class_counts.get(label_idx, 0)} samples")

    # Split data
    print("\n[3/5] Training Random Forest...")
    if len(X) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
            if min(Counter(y).values()) >= 2 else None
        )
    else:
        # Too few samples to split, train on all
        X_train, X_test, y_train, y_test = X, X, y, y
        print("  (Training on all data — too few samples to split)")

    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n[4/5] Evaluation Results:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.1%}")
    print()

    # Only show classes that exist in test set
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [label_names[i] for i in present_labels]
    print(classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    # Feature importance
    print("  Feature Importance (what drives productivity):")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx):
        bar = "█" * int(importances[idx] * 40) + "░" * (40 - int(importances[idx] * 40))
        print(f"    {rank+1}. {feature_names[idx]:<20} {bar} {importances[idx]:.1%}")

    # Save model
    print(f"\n[5/5] Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'num_sessions': len(files),
        'num_samples': len(X),
        'accuracy': round(accuracy, 4),
        'feature_names': feature_names,
        'label_names': label_names,
        'feature_importance': {
            feature_names[i]: round(float(importances[i]), 4)
            for i in sorted_idx
        },
        'class_distribution': {
            label_names[k]: int(v) for k, v in class_counts.items()
        },
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Metadata saved to: {METADATA_PATH}")
    print("\n" + "=" * 55)
    print("  ✅ Model ready! It will be loaded automatically")
    print("     by main.py next time you run the tracker.")
    print("=" * 55)


if __name__ == '__main__':
    train_model()
