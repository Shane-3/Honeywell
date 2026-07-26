"""
Anomaly Classifier
==================
Hybrid classification system combining:
1. Rule-based overrides for unambiguous attack patterns (brute force,
   impossible travel, credential stuffing, device spoofing)
2. RandomForest multi-class classifier for ambiguous cases (lateral movement,
   low-and-slow exfiltration, insider drift)

Uses 7 attack-specific engineered features + baseline/sequence model scores
as input. Trained on labeled synthetic data.
"""

import json
import os
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# Attack-Specific Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_LABELS = [
    "brute_force", "impossible_travel", "credential_stuffing",
    "lateral_movement", "device_spoofing", "low_and_slow_exfil",
    "insider_drift",
]

# Rule thresholds for deterministic classification
RULE_THRESHOLDS = {
    "brute_force_failed_auth_count": 4,          # >=4 failed auths in 5min window
    "credential_stuffing_distinct_entities": 3,   # >=3 distinct entities from same IP in 5min
    "impossible_travel_velocity_kmh": 1000,       # >1000 km/h implied travel
    "device_spoofing_mismatch": True,             # fingerprint not in known set
}


def compute_attack_features(events_df, baseline_scores_df, sequence_scores_df):
    """
    Compute 7 attack-specific engineered features for all events.
    These features require looking at context around each event (temporal
    windows, cross-entity patterns).

    Args:
        events_df: Full events DataFrame (sorted by timestamp)
        baseline_scores_df: DataFrame with baseline sub-scores
        sequence_scores_df: DataFrame with sequence model scores

    Returns:
        DataFrame with 7 attack-specific features per event
    """
    print("[*] Computing attack-specific engineered features...")

    n = len(events_df)
    features = {
        "failed_auth_count_5min": np.zeros(n),
        "distinct_entities_from_ip_5min": np.zeros(n),
        "geo_velocity_kmh": np.zeros(n),
        "resource_breadth_new_1hr": np.zeros(n),
        "fingerprint_mismatch": np.zeros(n),
        "offhours_access_trend_7d": np.zeros(n),
        "privilege_footprint_growth_30d": np.zeros(n),
    }

    # Precompute timestamps
    timestamps = pd.to_datetime(events_df["timestamp"])
    events_df = events_df.copy()
    events_df["_ts"] = timestamps
    events_df["_hour"] = timestamps.dt.hour

    # Copy geo velocity from baseline scores if available
    if "geo_velocity_kmh" in baseline_scores_df.columns:
        features["geo_velocity_kmh"] = baseline_scores_df["geo_velocity_kmh"].values

    # Copy fingerprint mismatch from baseline scores
    if "device_novelty" in baseline_scores_df.columns:
        features["fingerprint_mismatch"] = baseline_scores_df["device_novelty"].values

    # Copy low-slow score from sequence model
    if "low_slow_score" in sequence_scores_df.columns:
        features["offhours_access_trend_7d"] = sequence_scores_df["low_slow_score"].values

    # ── Failed auth count in 5-minute window (brute force) ─────────────
    # Group by entity and compute rolling failed auth counts
    auth_results = events_df["auth_result"].values
    entity_ids = events_df["entity_id"].values
    ts_values = timestamps.values

    # Build index for efficient lookups
    entity_event_indices = defaultdict(list)
    for i in range(n):
        entity_event_indices[entity_ids[i]].append(i)

    for entity_id, indices in entity_event_indices.items():
        for idx in indices:
            if auth_results[idx] != "fail":
                continue
            # Count failed auths in 5-min window centered on this event
            window_start = ts_values[idx] - np.timedelta64(5, "m")
            window_end = ts_values[idx] + np.timedelta64(5, "m")
            count = 0
            for j in indices:
                if (ts_values[j] >= window_start and ts_values[j] <= window_end
                        and auth_results[j] == "fail"):
                    count += 1
            features["failed_auth_count_5min"][idx] = count

    # ── Distinct entities from same IP in 5-minute window (credential stuffing) ──
    ip_event_indices = defaultdict(list)
    source_ips = events_df["source_ip"].values
    for i in range(n):
        ip_event_indices[source_ips[i]].append(i)

    for source_ip, indices in ip_event_indices.items():
        if len(indices) < 3:
            continue
        for idx in indices:
            window_start = ts_values[idx] - np.timedelta64(5, "m")
            window_end = ts_values[idx] + np.timedelta64(5, "m")
            entities_in_window = set()
            for j in indices:
                if ts_values[j] >= window_start and ts_values[j] <= window_end:
                    entities_in_window.add(entity_ids[j])
            features["distinct_entities_from_ip_5min"][idx] = len(entities_in_window)

    # ── Resource breadth (new resources in 1hr) for lateral movement ──────
    resources = events_df["resource_accessed"].values

    # Build per-entity historical resource set (resources seen before each event)
    for entity_id, indices in entity_event_indices.items():
        historical_resources = set()
        for idx in indices:
            # Count new resources in 1hr window
            window_start = ts_values[idx] - np.timedelta64(1, "h")
            new_in_window = 0
            for j in indices:
                if ts_values[j] >= window_start and ts_values[j] <= ts_values[idx]:
                    if resources[j] not in historical_resources:
                        new_in_window += 1
            features["resource_breadth_new_1hr"][idx] = new_in_window
            historical_resources.add(resources[idx])

    # ── Privilege footprint growth (insider drift) ────────────────────────
    for entity_id, indices in entity_event_indices.items():
        resource_set_7d_ago = set()
        resource_set_now = set()
        for idx in indices:
            ts_now = ts_values[idx]
            ts_7d_ago = ts_now - np.timedelta64(30, "D")
            if ts_values[idx] <= ts_7d_ago + np.timedelta64(7, "D"):
                resource_set_7d_ago.add(resources[idx])
            resource_set_now.add(resources[idx])

        if len(resource_set_7d_ago) > 0:
            growth = len(resource_set_now - resource_set_7d_ago) / max(len(resource_set_7d_ago), 1)
        else:
            growth = 0

        for idx in indices:
            features["privilege_footprint_growth_30d"][idx] = min(growth, 5.0)

    result_df = pd.DataFrame(features, index=events_df.index)
    print(f"    Computed features for {n} events")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Override Layer
# ─────────────────────────────────────────────────────────────────────────────

def apply_rule_overrides(attack_features, baseline_scores, events_df):
    """
    Apply deterministic rule-based classification for unambiguous patterns.

    Returns:
        Series of predicted labels (or None where rules don't fire)
        Series of confidence scores
        Series of rule names that fired (or None)
    """
    n = len(attack_features)
    predictions = pd.Series([None] * n, index=attack_features.index)
    confidences = pd.Series([0.0] * n, index=attack_features.index)
    rule_fired = pd.Series([None] * n, index=attack_features.index)

    # Rule 1: Brute Force
    mask = (attack_features["failed_auth_count_5min"] >=
            RULE_THRESHOLDS["brute_force_failed_auth_count"])
    predictions[mask] = "brute_force"
    confidences[mask] = 0.95
    rule_fired[mask] = "failed_auth_burst"

    # Rule 2: Impossible Travel
    mask = (attack_features["geo_velocity_kmh"] >
            RULE_THRESHOLDS["impossible_travel_velocity_kmh"])
    predictions[mask] = "impossible_travel"
    confidences[mask] = 0.92
    rule_fired[mask] = "geo_velocity_exceeded"

    # Rule 3: Credential Stuffing
    mask = (attack_features["distinct_entities_from_ip_5min"] >=
            RULE_THRESHOLDS["credential_stuffing_distinct_entities"])
    # Don't override if already classified as brute force
    mask = mask & predictions.isna()
    predictions[mask] = "credential_stuffing"
    confidences[mask] = 0.90
    rule_fired[mask] = "multi_entity_same_ip"

    # Rule 4: Device Spoofing
    mask = (attack_features["fingerprint_mismatch"] > 0.5)
    mask = mask & predictions.isna()
    # Only flag if other signals also indicate anomaly
    if "baseline_score" in baseline_scores.columns:
        mask = mask & (baseline_scores["baseline_score"] > 0.3)
    predictions[mask] = "device_spoofing"
    confidences[mask] = 0.88
    rule_fired[mask] = "fingerprint_mismatch"

    return predictions, confidences, rule_fired


# ─────────────────────────────────────────────────────────────────────────────
# ML Classifier
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyClassifier:
    """
    Hybrid classifier: rule overrides + RandomForest for ambiguous cases.
    """

    def __init__(self):
        self.rf_model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.feature_importances = {}

    def _build_feature_matrix(self, baseline_scores, sequence_scores, attack_features):
        """Combine all features into a single feature matrix for the RF."""
        feature_dfs = []

        # Baseline sub-scores
        baseline_cols = ["baseline_score", "hour_zscore", "geo_novelty",
                         "resource_novelty", "duration_zscore",
                         "auth_method_novelty", "device_novelty"]
        for col in baseline_cols:
            if col in baseline_scores.columns:
                feature_dfs.append(baseline_scores[[col]])

        # Sequence scores
        seq_cols = ["sequence_score", "markov_score", "low_slow_score"]
        for col in seq_cols:
            if col in sequence_scores.columns:
                feature_dfs.append(sequence_scores[[col]])

        # Attack-specific features
        feature_dfs.append(attack_features)

        X = pd.concat(feature_dfs, axis=1)
        # Fill any NaN
        X = X.fillna(0.0)
        self.feature_columns = list(X.columns)
        return X

    def fit(self, baseline_scores, sequence_scores, attack_features, labels):
        """
        Train the RandomForest classifier on anomalous events only.
        """
        X = self._build_feature_matrix(baseline_scores, sequence_scores, attack_features)

        # Only train on labeled anomaly events (not normal)
        anomaly_mask = labels["label"] != "normal"
        X_train = X[anomaly_mask]
        y_train = labels[anomaly_mask]["label"]

        if len(X_train) < 10:
            print("[!] Not enough anomaly events to train classifier")
            return

        print(f"[*] Training RandomForest classifier on {len(X_train)} anomaly events...")
        print(f"    Features: {len(self.feature_columns)}")
        print(f"    Classes: {sorted(y_train.unique())}")

        # Encode labels
        self.label_encoder.fit(ATTACK_LABELS + ["insider_drift"])
        y_encoded = self.label_encoder.transform(y_train)

        # Train RandomForest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",  # Handle imbalance within attack types
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_encoded)

        # Store feature importances
        self.feature_importances = dict(
            zip(self.feature_columns, self.rf_model.feature_importances_)
        )

        print(f"    Training accuracy: {self.rf_model.score(X_train, y_encoded):.4f}")
        print(f"    Top 5 features:")
        sorted_fi = sorted(self.feature_importances.items(), key=lambda x: -x[1])
        for name, imp in sorted_fi[:5]:
            print(f"      {name:35s}: {imp:.4f}")

    def predict(self, baseline_scores, sequence_scores, attack_features, events_df):
        """
        Classify anomalous events. Applies rule overrides first, then ML.

        Returns:
            DataFrame with: predicted_type, predicted_type_confidence,
                           rule_fired, top_features (JSON)
        """
        X = self._build_feature_matrix(baseline_scores, sequence_scores, attack_features)

        n = len(X)
        result = pd.DataFrame({
            "predicted_type": ["normal"] * n,
            "predicted_type_confidence": [1.0] * n,
            "rule_fired": [None] * n,
        }, index=X.index)

        # Step 1: Determine which events are flagged as anomalous
        # (above a combined score threshold)
        combined_score = np.zeros(n)
        if "baseline_score" in baseline_scores.columns:
            combined_score += 0.6 * baseline_scores["baseline_score"].values
        if "sequence_score" in sequence_scores.columns:
            combined_score += 0.4 * sequence_scores["sequence_score"].values

        anomaly_threshold = 0.15  # Flag anything above this for classification
        flagged_mask = combined_score > anomaly_threshold

        # Step 2: Apply rule overrides
        rule_preds, rule_confs, rule_names = apply_rule_overrides(
            attack_features, baseline_scores, events_df
        )

        # Apply rules to flagged events
        rule_mask = rule_preds.notna() & flagged_mask
        result.loc[rule_mask, "predicted_type"] = rule_preds[rule_mask]
        result.loc[rule_mask, "predicted_type_confidence"] = rule_confs[rule_mask]
        result.loc[rule_mask, "rule_fired"] = rule_names[rule_mask]

        # Step 3: ML classification for flagged events without rule match
        ml_mask = flagged_mask & rule_preds.isna()
        if self.rf_model is not None and ml_mask.sum() > 0:
            X_ml = X[ml_mask]
            y_pred_encoded = self.rf_model.predict(X_ml)
            y_pred_proba = self.rf_model.predict_proba(X_ml)

            y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)
            y_pred_conf = y_pred_proba.max(axis=1)

            result.loc[ml_mask, "predicted_type"] = y_pred_labels
            result.loc[ml_mask, "predicted_type_confidence"] = np.round(y_pred_conf, 4)

        # Step 4: Compute top contributing features per event
        top_features_list = []
        for idx in X.index:
            if result.loc[idx, "predicted_type"] == "normal":
                top_features_list.append(json.dumps([]))
                continue

            # Get this event's feature values
            event_features = X.loc[idx]
            # Rank by feature_importance * feature_value (contribution)
            contributions = {}
            for col in self.feature_columns:
                imp = self.feature_importances.get(col, 0.05)
                val = event_features[col]
                contributions[col] = imp * val

            # Top 5 contributing features
            sorted_contribs = sorted(contributions.items(), key=lambda x: -abs(x[1]))[:5]
            top_features = [(name, round(val, 4)) for name, val in sorted_contribs if abs(val) > 0.001]
            top_features_list.append(json.dumps(top_features))

        result["top_features"] = top_features_list

        print(f"[*] Classification results:")
        print(f"    Flagged events: {flagged_mask.sum()}")
        print(f"    Rule-classified: {rule_mask.sum()}")
        print(f"    ML-classified: {ml_mask.sum()}")
        pred_counts = result["predicted_type"].value_counts()
        for ptype, count in pred_counts.items():
            if ptype != "normal":
                print(f"    {ptype:25s}: {count}")

        return result

    def save(self, path):
        """Save the trained model."""
        joblib.dump({
            "rf_model": self.rf_model,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "feature_importances": self.feature_importances,
        }, path)
        print(f"[OK] Classifier saved to {path}")

    def load(self, path):
        """Load a trained model."""
        data = joblib.load(path)
        self.rf_model = data["rf_model"]
        self.label_encoder = data["label_encoder"]
        self.feature_columns = data["feature_columns"]
        self.feature_importances = data["feature_importances"]
        print(f"[OK] Classifier loaded from {path}")
