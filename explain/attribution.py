"""
Explainability / Attribution Layer
====================================
Generates human-readable explanations for each alert based on the actual
features that drove the anomaly score and classification.

Two modes:
1. Rule-classified alerts: explanation directly from which rule fired
2. ML-classified alerts: top contributing features + templated explanation

Output: (top_features_list, explanation_text) per alert
"""

import json
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Feature Human Names — maps engineered feature names to analyst-readable phrases
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_HUMAN_NAMES = {
    # Baseline sub-scores
    "baseline_score": "overall behavioral anomaly score",
    "hour_zscore": "unusual login time",
    "geo_novelty": "access from unfamiliar location",
    "resource_novelty": "access to unusual resource",
    "duration_zscore": "unusual session length",
    "auth_method_novelty": "unusual authentication method",
    "device_novelty": "unrecognized device",

    # Sequence model scores
    "sequence_score": "unusual access sequence pattern",
    "markov_score": "unusual resource transition pattern",
    "low_slow_score": "elevated off-hours access trend",

    # Attack-specific features
    "failed_auth_count_5min": "failed authentication attempts in 5-min window",
    "distinct_entities_from_ip_5min": "multiple accounts accessed from same IP",
    "geo_velocity_kmh": "implied travel speed between logins",
    "resource_breadth_new_1hr": "new resources accessed in 1-hour window",
    "fingerprint_mismatch": "device fingerprint mismatch",
    "offhours_access_trend_7d": "off-hours access trend over 7 days",
    "privilege_footprint_growth_30d": "privilege/resource footprint expansion",
}

# Attack type descriptions for context
ATTACK_DESCRIPTIONS = {
    "brute_force": "Rapid repeated failed authentication attempts suggesting a brute force attack",
    "impossible_travel": "Login from geographically distant location in an implausibly short time",
    "credential_stuffing": "Multiple user accounts targeted from the same IP addresses",
    "lateral_movement": "Unusual breadth of resource access suggesting lateral movement",
    "device_spoofing": "Device fingerprint does not match known devices for this entity",
    "low_and_slow_exfil": "Gradual off-hours access pattern suggesting slow data exfiltration",
    "insider_drift": "Gradual expansion of resource access footprint beyond typical scope",
}

# Rule-fired explanation templates
RULE_EXPLANATIONS = {
    "failed_auth_burst": "{count} failed authentication attempts from {ip} in {window}",
    "geo_velocity_exceeded": "Login from {geo1} then {geo2} implies {velocity:.0f} km/h travel ({distance:.0f} km in {minutes:.0f} min)",
    "multi_entity_same_ip": "{count} different accounts accessed from {ip} in 5 minutes",
    "fingerprint_mismatch": "Device fingerprint does not match any of {n_known} known devices for {entity}",
}


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(event, classification_result, baseline_scores_row,
                         attack_features_row):
    """
    Generate a human-readable explanation for a single alert.

    Args:
        event: dict/Series of the original event data
        classification_result: dict with predicted_type, rule_fired, top_features
        baseline_scores_row: dict with baseline sub-scores
        attack_features_row: dict with attack-specific features

    Returns:
        str: human-readable explanation text
    """
    predicted_type = classification_result.get("predicted_type", "unknown")
    rule_fired = classification_result.get("rule_fired")
    top_features_json = classification_result.get("top_features", "[]")

    # Parse top features
    if isinstance(top_features_json, str):
        try:
            top_features = json.loads(top_features_json)
        except (json.JSONDecodeError, TypeError):
            top_features = []
    else:
        top_features = top_features_json if top_features_json else []

    # ── Rule-based explanation ────────────────────────────────────────────
    if rule_fired:
        return _generate_rule_explanation(
            rule_fired, event, baseline_scores_row, attack_features_row
        )

    # ── ML-based explanation ──────────────────────────────────────────────
    return _generate_ml_explanation(
        predicted_type, top_features, event, baseline_scores_row, attack_features_row
    )


def _generate_rule_explanation(rule_fired, event, baseline_scores, attack_features):
    """Generate explanation from a fired rule with real feature values."""

    if rule_fired == "failed_auth_burst":
        count = int(attack_features.get("failed_auth_count_5min", 0))
        ip = event.get("source_ip", "unknown")
        return (f"{count} failed authentication attempts from {ip} "
                f"in a 5-minute window (brute force pattern)")

    elif rule_fired == "geo_velocity_exceeded":
        velocity = float(baseline_scores.get("geo_velocity_kmh", 0))
        geo = event.get("geo_location", "unknown location")
        return (f"Implied travel speed of {velocity:.0f} km/h to reach {geo} "
                f"(impossible travel pattern)")

    elif rule_fired == "multi_entity_same_ip":
        count = int(attack_features.get("distinct_entities_from_ip_5min", 0))
        ip = event.get("source_ip", "unknown")
        return (f"{count} different user accounts accessed from {ip} "
                f"in 5 minutes (credential stuffing pattern)")

    elif rule_fired == "fingerprint_mismatch":
        entity = event.get("entity_id", "unknown")
        fp = event.get("device_fingerprint", "unknown")
        return (f"Device fingerprint '{fp[:30]}...' does not match known "
                f"devices for {entity} (device spoofing)")

    return f"Rule '{rule_fired}' triggered for this event"


def _generate_ml_explanation(predicted_type, top_features, event,
                              baseline_scores, attack_features):
    """Generate explanation from ML classifier's top contributing features."""

    # Build the main reason string from top features
    if not top_features:
        # Fallback: use the attack type description
        desc = ATTACK_DESCRIPTIONS.get(predicted_type, f"Classified as {predicted_type}")
        return desc

    # Convert feature names to human-readable phrases
    reasons = []
    details = []

    for feat_name, feat_value in top_features[:3]:  # Top 3 features
        human_name = FEATURE_HUMAN_NAMES.get(feat_name, feat_name.replace("_", " "))
        reasons.append(human_name)

        # Add specific value context where useful
        if feat_name == "geo_velocity_kmh":
            details.append(f"{human_name} ({feat_value:.0f} km/h)")
        elif feat_name == "failed_auth_count_5min":
            details.append(f"{int(feat_value)} {human_name}")
        elif feat_name == "distinct_entities_from_ip_5min":
            details.append(f"{int(feat_value)} {human_name}")
        elif feat_name == "resource_breadth_new_1hr":
            details.append(f"{int(feat_value)} {human_name}")
        elif feat_name in ("hour_zscore", "duration_zscore"):
            details.append(f"{human_name} (z={feat_value:.1f})")
        else:
            details.append(human_name)

    # Construct explanation
    attack_desc = ATTACK_DESCRIPTIONS.get(predicted_type, "")
    if details:
        reason_str = " + ".join(details[:3])
        explanation = f"Flagged due to {reason_str}"
        if attack_desc:
            explanation += f". {attack_desc}"
    else:
        explanation = attack_desc or f"Classified as {predicted_type}"

    return explanation


def generate_explanations_batch(events_df, classification_df,
                                 baseline_scores_df, attack_features_df):
    """
    Generate explanations for all classified events.

    Returns:
        Series of explanation strings
    """
    print(f"[*] Generating explanations for {len(events_df)} events...")

    explanations = []
    for idx in events_df.index:
        predicted_type = classification_df.loc[idx, "predicted_type"]

        if predicted_type == "normal":
            explanations.append("")
            continue

        event = events_df.loc[idx].to_dict()
        classification = classification_df.loc[idx].to_dict()
        baseline = baseline_scores_df.loc[idx].to_dict() if idx in baseline_scores_df.index else {}
        attack_feats = attack_features_df.loc[idx].to_dict() if idx in attack_features_df.index else {}

        explanation = generate_explanation(event, classification, baseline, attack_feats)
        explanations.append(explanation)

    n_explained = sum(1 for e in explanations if e)
    print(f"    Generated {n_explained} explanations")

    return pd.Series(explanations, index=events_df.index)
