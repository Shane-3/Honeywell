"""
Training Orchestrator
======================
End-to-end pipeline that:
1. Loads synthetic_logs.csv
2. Performs time-based train/test split
3. Fits baseline profiler
4. Fits sequence model
5. Computes attack-specific features
6. Fits classifier
7. Scores test set
8. Generates explanations
9. Writes alerts.csv, entity_profiles.csv, scored_events.csv
10. Persists trained models

Usage:
    python models/train.py
"""

import json
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

from models.baseline_profile import BaselineProfiler
from models.sequence_model import SequenceModel
from models.classifier import AnomalyClassifier, compute_attack_features
from explain.attribution import generate_explanations_batch


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_logs.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")

# Time-based split ratios (by day offset from start)
TRAIN_DAYS = 30   # First 30 days for training
# Remaining days for testing (~15 days)

# Combined risk score weights
BASELINE_WEIGHT = 0.55
SEQUENCE_WEIGHT = 0.45

# Alert threshold: events above this combined score become alerts
ALERT_THRESHOLD = 0.18


def main():
    start_time = time.time()

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print(f"[!] Data file not found: {DATA_PATH}")
        print("    Run: python data/generate_data.py --seed 42")
        sys.exit(1)

    print(f"\n[1/9] Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    labels = df[["label"]].copy()
    print(f"      {len(df)} events, {df['entity_id'].nunique()} entities")

    # ── Step 2: Time-based split ──────────────────────────────────────────
    print(f"\n[2/9] Time-based train/test split...")
    df["_ts"] = pd.to_datetime(df["timestamp"])
    min_date = df["_ts"].min()
    cutoff = min_date + timedelta(days=TRAIN_DAYS)

    train_mask = df["_ts"] < cutoff
    test_mask = df["_ts"] >= cutoff

    train_df = df[train_mask].copy().sort_values("_ts").reset_index(drop=True)
    test_df = df[test_mask].copy().sort_values("_ts").reset_index(drop=True)
    train_labels = labels[train_mask].copy().reset_index(drop=True)
    test_labels = labels[test_mask].copy().reset_index(drop=True)

    # Drop temp column
    train_df = train_df.drop(columns=["_ts"])
    test_df = test_df.drop(columns=["_ts"])
    df = df.drop(columns=["_ts"])

    print(f"      Train: {len(train_df)} events (first {TRAIN_DAYS} days)")
    print(f"      Test:  {len(test_df)} events (remaining days)")
    print(f"      Train attacks: {(train_labels['label'] != 'normal').sum()}")
    print(f"      Test attacks:  {(test_labels['label'] != 'normal').sum()}")

    # ── Step 3: Fit baseline profiler ─────────────────────────────────────
    print(f"\n[3/9] Fitting baseline profiler...")
    profiler = BaselineProfiler()
    profiler.fit(train_df, train_labels)

    # Score test set fully; for train, only score anomaly events (classifier only trains on anomalies)
    print(f"\n[4/9] Scoring with baseline profiler...")
    test_baseline_scores = profiler.score_dataframe(test_df)

    # Score only anomaly train events for classifier training
    anomaly_train_mask = train_labels["label"] != "normal"
    anomaly_train_df = train_df[anomaly_train_mask].copy().reset_index(drop=True)
    anomaly_train_labels = train_labels[anomaly_train_mask].copy().reset_index(drop=True)
    print(f"    Scoring {len(anomaly_train_df)} anomaly train events for classifier...")
    train_baseline_scores = profiler.score_dataframe(anomaly_train_df)

    # ── Step 4: Fit sequence model ────────────────────────────────────────
    print(f"\n[5/9] Fitting sequence model...")
    seq_model = SequenceModel()
    seq_model.fit(train_df, train_labels)

    # Score test set
    print(f"\n[6/9] Scoring with sequence model...")
    seq_model.entity_recent_resources.clear()
    seq_model.low_slow_detector.entity_offhours_history.clear()
    test_seq_scores = seq_model.score_dataframe(test_df)

    # Score anomaly train events
    seq_model.entity_recent_resources.clear()
    seq_model.low_slow_detector.entity_offhours_history.clear()
    train_seq_scores = seq_model.score_dataframe(anomaly_train_df)

    # ── Step 5: Compute attack features ───────────────────────────────────
    print(f"\n[7/9] Computing attack-specific features...")
    train_attack_features = compute_attack_features(
        anomaly_train_df, train_baseline_scores, train_seq_scores
    )
    test_attack_features = compute_attack_features(
        test_df, test_baseline_scores, test_seq_scores
    )

    # ── Step 6: Train classifier ──────────────────────────────────────────
    print(f"\n[8/9] Training classifier...")
    classifier = AnomalyClassifier()
    classifier.fit(train_baseline_scores, train_seq_scores,
                   train_attack_features, anomaly_train_labels)

    # ── Step 7: Classify test set ─────────────────────────────────────────
    print(f"\n[9/9] Classifying and explaining test set...")
    classification = classifier.predict(
        test_baseline_scores, test_seq_scores, test_attack_features, test_df
    )

    # ── Step 8: Generate explanations ─────────────────────────────────────
    explanations = generate_explanations_batch(
        test_df, classification, test_baseline_scores, test_attack_features
    )

    # ── Step 9: Compute combined risk score ───────────────────────────────
    risk_scores = (
        BASELINE_WEIGHT * test_baseline_scores["baseline_score"].values +
        SEQUENCE_WEIGHT * test_seq_scores["sequence_score"].values
    )
    risk_scores = np.clip(risk_scores, 0, 1)

    # ── Step 10: Build alerts table ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("BUILDING OUTPUT TABLES")
    print("=" * 70)

    # Alerts: only events above threshold and classified as non-normal
    alert_mask = (
        (risk_scores > ALERT_THRESHOLD) |
        (classification["predicted_type"] != "normal")
    )

    alerts_df = pd.DataFrame({
        "session_id": test_df.loc[alert_mask, "session_id"].values,
        "entity_id": test_df.loc[alert_mask, "entity_id"].values,
        "entity_type": test_df.loc[alert_mask, "entity_type"].values,
        "timestamp": test_df.loc[alert_mask, "timestamp"].values,
        "source_ip": test_df.loc[alert_mask, "source_ip"].values,
        "geo_location": test_df.loc[alert_mask, "geo_location"].values,
        "resource_accessed": test_df.loc[alert_mask, "resource_accessed"].values,
        "auth_method": test_df.loc[alert_mask, "auth_method"].values,
        "auth_result": test_df.loc[alert_mask, "auth_result"].values,
        "session_duration": test_df.loc[alert_mask, "session_duration"].values,
        "device_fingerprint": test_df.loc[alert_mask, "device_fingerprint"].values,
        "risk_score": np.round(risk_scores[alert_mask], 4),
        "predicted_type": classification.loc[alert_mask, "predicted_type"].values,
        "predicted_type_confidence": classification.loc[alert_mask, "predicted_type_confidence"].values,
        "top_features": classification.loc[alert_mask, "top_features"].values,
        "explanation_text": explanations[alert_mask].values,
        "is_cold_start": test_baseline_scores.loc[alert_mask, "is_cold_start"].values,
        "is_drift_flagged": False,  # Will be populated by drift check
    })

    # Check drift for entities in alerts
    test_dates = pd.to_datetime(test_df["timestamp"])
    for idx in alerts_df.index:
        entity_id = alerts_df.loc[idx, "entity_id"]
        # Simple drift flag: check if entity's recent resource set changed
        alerts_df.loc[idx, "is_drift_flagged"] = profiler.check_drift(
            entity_id,
            test_dates.max(),
            df, labels
        )

    # Sort by risk score descending
    alerts_df = alerts_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    print(f"\n[*] Alerts generated: {len(alerts_df)}")
    print(f"    Predicted type distribution:")
    for ptype, count in alerts_df["predicted_type"].value_counts().items():
        print(f"      {ptype:25s}: {count}")

    # ── Step 11: Build scored_events table (all test events with scores) ──
    scored_events_df = test_df.copy()
    scored_events_df["risk_score"] = np.round(risk_scores, 4)
    scored_events_df["baseline_score"] = test_baseline_scores["baseline_score"].values
    scored_events_df["sequence_score"] = test_seq_scores["sequence_score"].values
    scored_events_df["predicted_type"] = classification["predicted_type"].values
    scored_events_df["predicted_type_confidence"] = classification["predicted_type_confidence"].values
    scored_events_df["explanation_text"] = explanations.values
    scored_events_df["is_cold_start"] = test_baseline_scores["is_cold_start"].values
    scored_events_df["ground_truth_label"] = test_labels["label"].values

    # Add baseline sub-scores
    for col in ["hour_zscore", "geo_novelty", "resource_novelty",
                "duration_zscore", "auth_method_novelty", "device_novelty",
                "geo_velocity_kmh"]:
        if col in test_baseline_scores.columns:
            scored_events_df[col] = test_baseline_scores[col].values

    # Add attack features
    for col in test_attack_features.columns:
        scored_events_df[col] = test_attack_features[col].values

    # ── Step 12: Save everything ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Create output dirs
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save alerts
    alerts_path = os.path.join(OUTPUT_DIR, "alerts.csv")
    alerts_df.to_csv(alerts_path, index=False)
    print(f"[OK] Alerts saved: {alerts_path} ({len(alerts_df)} rows)")

    # Save scored events
    scored_path = os.path.join(OUTPUT_DIR, "scored_events.csv")
    scored_events_df.to_csv(scored_path, index=False)
    print(f"[OK] Scored events saved: {scored_path} ({len(scored_events_df)} rows)")

    # Save entity profiles
    profiles_df = profiler.get_profiles_df()
    profiles_path = os.path.join(OUTPUT_DIR, "entity_profiles.csv")
    profiles_df.to_csv(profiles_path, index=False)
    print(f"[OK] Entity profiles saved: {profiles_path} ({len(profiles_df)} rows)")

    # Save trained models
    classifier.save(os.path.join(MODELS_DIR, "classifier.joblib"))
    joblib.dump(profiler, os.path.join(MODELS_DIR, "profiler.joblib"))
    joblib.dump(seq_model, os.path.join(MODELS_DIR, "sequence_model.joblib"))
    print(f"[OK] Models saved to {MODELS_DIR}")

    # Save train/test labels for evaluation
    test_labels.to_csv(os.path.join(OUTPUT_DIR, "test_labels.csv"), index=False)
    print(f"[OK] Test labels saved for evaluation")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"  Total events processed: {len(df)}")
    print(f"  Train set: {len(train_df)} | Test set: {len(test_df)}")
    print(f"  Alerts generated: {len(alerts_df)}")
    print(f"  Entity profiles: {len(profiles_df)}")
    print(f"\n  Output files:")
    print(f"    {alerts_path}")
    print(f"    {scored_path}")
    print(f"    {profiles_path}")

    # Print a few sample explanations
    print(f"\n  Sample explanations (top 5 alerts):")
    for i, row in alerts_df.head(5).iterrows():
        print(f"    [{row['predicted_type']:20s}] (score={row['risk_score']:.3f}) "
              f"{row['explanation_text'][:80]}")


if __name__ == "__main__":
    main()
