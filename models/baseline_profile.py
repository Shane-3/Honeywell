"""
Baseline Profiling Model
========================
Builds per-entity behavioral profiles from historical normal events and scores
new events against those profiles using 6 interpretable sub-scores.

Handles:
- Cold-start entities (< N observed events) via entity-type aggregate fallback
- Concept drift via rolling-window profile recomputation (14-day window)

Output: (score, sub_scores_dict) per event, where score is 0-1 normalized.
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

COLD_START_THRESHOLD = 5  # min events before entity-specific profile is trusted
ROLLING_WINDOW_DAYS = 14  # drift handling: profile built from trailing N days
SCORE_WEIGHTS = {
    "hour_zscore": 0.15,
    "geo_novelty": 0.25,
    "resource_novelty": 0.15,
    "duration_zscore": 0.10,
    "auth_method_novelty": 0.15,
    "device_novelty": 0.20,
}

# Geo coordinates for distance calculation (city, country -> lat, lon)
GEO_COORDS = {
    "New York, US": (40.7128, -74.0060),
    "San Francisco, US": (37.7749, -122.4194),
    "Chicago, US": (41.8781, -87.6298),
    "London, UK": (51.5074, -0.1278),
    "Berlin, DE": (52.5200, 13.4050),
    "Paris, FR": (48.8566, 2.3522),
    "Tokyo, JP": (35.6762, 139.6503),
    "Sydney, AU": (-33.8688, 151.2093),
    "Mumbai, IN": (19.0760, 72.8777),
    "Singapore, SG": (1.3521, 103.8198),
    "Toronto, CA": (43.6532, -79.3832),
    "Dubai, AE": (25.2048, 55.2708),
    "Sao Paulo, BR": (-23.5505, -46.6333),
    "Seoul, KR": (37.5665, 126.9780),
    "Amsterdam, NL": (52.3676, 4.9041),
    "Stockholm, SE": (59.3293, 18.0686),
    "Moscow, RU": (55.7558, 37.6173),
    "Cape Town, ZA": (-33.9249, 18.4241),
    "Mexico City, MX": (19.4326, -99.1332),
    "Bangkok, TH": (13.7563, 100.5018),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def get_geo_coords(geo_name):
    """Look up coordinates for a geo location name, with fuzzy matching."""
    if geo_name in GEO_COORDS:
        return GEO_COORDS[geo_name]
    # Fuzzy match: try matching city name
    city = geo_name.split(",")[0].strip()
    for key, coords in GEO_COORDS.items():
        if city.lower() in key.lower():
            return coords
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Entity Profile
# ─────────────────────────────────────────────────────────────────────────────

class EntityProfile:
    """Statistical behavioral profile for a single entity."""

    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type

        # Computed from normal events
        self.hour_histogram = np.zeros(24)  # 24 buckets
        self.hour_mean = 12.0
        self.hour_std = 6.0
        self.home_geo_set = set()
        self.home_ip_prefixes = set()
        self.resource_frequency = defaultdict(int)
        self.total_resource_accesses = 0
        self.session_duration_mean = 300.0
        self.session_duration_std = 150.0
        self.auth_method_counts = defaultdict(int)
        self.primary_auth_method = "password"
        self.known_device_fingerprints = set()
        self.first_seen = None
        self.last_seen = None
        self.n_observed_events = 0

        # For geo-velocity calculation
        self.last_event_time = None
        self.last_event_geo = None

    def update_from_events(self, events_df):
        """Build/update profile from a DataFrame of normal events for this entity."""
        if len(events_df) == 0:
            return

        self.n_observed_events = len(events_df)

        # Parse timestamps
        timestamps = pd.to_datetime(events_df["timestamp"])
        hours = timestamps.dt.hour

        # Hour histogram
        self.hour_histogram = np.zeros(24)
        for h in hours:
            self.hour_histogram[h] += 1
        if self.hour_histogram.sum() > 0:
            self.hour_histogram /= self.hour_histogram.sum()

        # Hour mean/std (circular mean would be better, but linear is simpler)
        hour_values = hours.values.astype(float)
        if len(hour_values) > 1:
            self.hour_mean = np.mean(hour_values)
            self.hour_std = max(1.0, np.std(hour_values))
        else:
            self.hour_mean = hour_values[0] if len(hour_values) > 0 else 12.0
            self.hour_std = 6.0

        # Geo set
        self.home_geo_set = set(events_df["geo_location"].unique())

        # IP prefixes (/24)
        self.home_ip_prefixes = set()
        for ip in events_df["source_ip"].unique():
            parts = ip.split(".")
            if len(parts) == 4:
                self.home_ip_prefixes.add(".".join(parts[:3]))

        # Resource frequency
        self.resource_frequency = defaultdict(int)
        for res in events_df["resource_accessed"]:
            self.resource_frequency[res] += 1
        self.total_resource_accesses = sum(self.resource_frequency.values())

        # Session duration stats
        durations = events_df["session_duration"].astype(float)
        if len(durations) > 1:
            self.session_duration_mean = durations.mean()
            self.session_duration_std = max(1.0, durations.std())
        elif len(durations) == 1:
            self.session_duration_mean = durations.iloc[0]
            self.session_duration_std = max(1.0, durations.iloc[0] * 0.3)

        # Auth method
        self.auth_method_counts = defaultdict(int)
        for m in events_df["auth_method"]:
            self.auth_method_counts[m] += 1
        if self.auth_method_counts:
            self.primary_auth_method = max(self.auth_method_counts,
                                            key=self.auth_method_counts.get)

        # Device fingerprints
        self.known_device_fingerprints = set(events_df["device_fingerprint"].unique())

        # Time range
        self.first_seen = timestamps.min()
        self.last_seen = timestamps.max()

    def to_dict(self):
        """Serialize profile to a dictionary for CSV/JSON storage."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "hour_mean": round(self.hour_mean, 2),
            "hour_std": round(self.hour_std, 2),
            "home_geo_set": json.dumps(sorted(self.home_geo_set)),
            "home_ip_prefixes": json.dumps(sorted(self.home_ip_prefixes)),
            "typical_resources": json.dumps(dict(self.resource_frequency)),
            "session_duration_mean": round(self.session_duration_mean, 2),
            "session_duration_std": round(self.session_duration_std, 2),
            "primary_auth_method": self.primary_auth_method,
            "known_device_fingerprints": json.dumps(sorted(self.known_device_fingerprints)),
            "first_seen": str(self.first_seen) if self.first_seen else "",
            "last_seen": str(self.last_seen) if self.last_seen else "",
            "n_observed_events": self.n_observed_events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_hour_zscore(event_hour, profile):
    """Z-score of event hour vs. entity's typical hour distribution."""
    if profile.hour_std < 0.5:
        return 0.0
    z = abs(event_hour - profile.hour_mean) / profile.hour_std
    # Handle wrap-around (e.g., 23 and 1 are close)
    z_wrap = abs(24 - abs(event_hour - profile.hour_mean)) / profile.hour_std
    return min(z, z_wrap)


def compute_geo_novelty(event_geo, profile):
    """1 if geo_location not in home_geo_set, else 0."""
    if not profile.home_geo_set:
        return 0.0  # No profile data, don't flag
    return 0.0 if event_geo in profile.home_geo_set else 1.0


def compute_geo_velocity(event_geo, event_time, profile):
    """
    Compute implied travel velocity (km/h) between this event and the
    entity's last known event. Returns (velocity_kmh, is_impossible).
    """
    if profile.last_event_time is None or profile.last_event_geo is None:
        return 0.0, False

    coords_current = get_geo_coords(event_geo)
    coords_last = get_geo_coords(profile.last_event_geo)

    if coords_current is None or coords_last is None:
        return 0.0, False

    dist_km = haversine_km(coords_last[0], coords_last[1],
                            coords_current[0], coords_current[1])

    time_diff = (event_time - profile.last_event_time).total_seconds()
    if time_diff <= 0:
        return 0.0, False

    velocity_kmh = dist_km / (time_diff / 3600)

    # > 1000 km/h is suspicious (commercial flight is ~900 km/h)
    is_impossible = velocity_kmh > 1000 and dist_km > 500

    return velocity_kmh, is_impossible


def compute_resource_novelty(event_resource, profile):
    """1 - frequency_ratio of resource in entity's typical resources."""
    if profile.total_resource_accesses == 0:
        return 0.0  # No profile data
    freq = profile.resource_frequency.get(event_resource, 0)
    ratio = freq / profile.total_resource_accesses
    return 1.0 - ratio  # unseen resource -> 1.0, most common -> low


def compute_duration_zscore(event_duration, profile):
    """Z-score of session_duration vs. entity's typical distribution."""
    if profile.session_duration_std < 1.0:
        return 0.0
    return abs(event_duration - profile.session_duration_mean) / profile.session_duration_std


def compute_auth_method_novelty(event_auth, profile):
    """1 if auth method doesn't match primary, else 0."""
    if not profile.primary_auth_method:
        return 0.0
    return 0.0 if event_auth == profile.primary_auth_method else 1.0


def compute_device_novelty(event_fingerprint, profile):
    """1 if device fingerprint not in known set, else 0."""
    if not profile.known_device_fingerprints:
        return 0.0  # No profile data
    return 0.0 if event_fingerprint in profile.known_device_fingerprints else 1.0


def compute_baseline_score(event, profile, is_cold_start=False):
    """
    Compute all 6 sub-scores and the combined baseline anomaly score.

    Returns:
        (combined_score, sub_scores_dict, geo_velocity_kmh)
    """
    event_hour = pd.to_datetime(event["timestamp"]).hour
    event_duration = float(event["session_duration"])
    event_time = pd.to_datetime(event["timestamp"])

    sub_scores = {}

    # 1. Hour z-score (capped at 4 for normalization)
    raw_hz = compute_hour_zscore(event_hour, profile)
    sub_scores["hour_zscore"] = min(raw_hz / 4.0, 1.0)

    # 2. Geo novelty
    sub_scores["geo_novelty"] = compute_geo_novelty(event["geo_location"], profile)

    # 3. Geo velocity (computed here, stored for classifier use)
    geo_velocity, is_impossible = compute_geo_velocity(
        event["geo_location"], event_time, profile
    )
    if is_impossible:
        sub_scores["geo_novelty"] = max(sub_scores["geo_novelty"], 0.9)

    # 4. Resource novelty
    sub_scores["resource_novelty"] = compute_resource_novelty(
        event["resource_accessed"], profile
    )

    # 5. Duration z-score (capped at 4 for normalization)
    raw_dz = compute_duration_zscore(event_duration, profile)
    sub_scores["duration_zscore"] = min(raw_dz / 4.0, 1.0)

    # 6. Auth method novelty
    sub_scores["auth_method_novelty"] = compute_auth_method_novelty(
        event["auth_method"], profile
    )

    # 7. Device novelty
    sub_scores["device_novelty"] = compute_device_novelty(
        event["device_fingerprint"], profile
    )

    # Cold-start adjustment: widen uncertainty (reduce confidence in scores)
    if is_cold_start:
        # Dampen novelty scores for cold-start to avoid false positives
        for key in ["geo_novelty", "resource_novelty", "device_novelty"]:
            sub_scores[key] *= 0.5

    # Combined weighted score
    combined = sum(
        SCORE_WEIGHTS[key] * sub_scores[key]
        for key in SCORE_WEIGHTS
    )
    # Normalize to 0-1 (weights sum to 1.0, each sub-score is 0-1)
    combined = min(max(combined, 0.0), 1.0)

    return combined, sub_scores, geo_velocity


# ─────────────────────────────────────────────────────────────────────────────
# Profile Builder
# ─────────────────────────────────────────────────────────────────────────────

class BaselineProfiler:
    """Builds and manages entity profiles, handles cold-start and drift."""

    def __init__(self, cold_start_threshold=COLD_START_THRESHOLD,
                 rolling_window_days=ROLLING_WINDOW_DAYS):
        self.cold_start_threshold = cold_start_threshold
        self.rolling_window_days = rolling_window_days

        # entity_id -> EntityProfile
        self.entity_profiles = {}
        # entity_type -> EntityProfile (aggregate for cold-start fallback)
        self.type_profiles = {}

    def fit(self, events_df, labels_df=None):
        """
        Build entity profiles from normal-labeled events.

        Args:
            events_df: DataFrame of events
            labels_df: Optional DataFrame with 'label' column. If provided,
                       only 'normal' events are used for profiling.
        """
        if labels_df is not None:
            normal_mask = labels_df["label"] == "normal"
            profile_data = events_df[normal_mask].copy()
        else:
            profile_data = events_df.copy()

        print(f"[*] Building profiles from {len(profile_data)} normal events...")

        # Build per-entity profiles
        for entity_id, group in profile_data.groupby("entity_id"):
            entity_type = group["entity_type"].iloc[0]
            profile = EntityProfile(entity_id, entity_type)
            profile.update_from_events(group)
            self.entity_profiles[entity_id] = profile

        # Build entity-type aggregate profiles (for cold-start fallback)
        for entity_type, group in profile_data.groupby("entity_type"):
            profile = EntityProfile(f"__type_{entity_type}", entity_type)
            profile.update_from_events(group)
            self.type_profiles[entity_type] = profile

        n_cold = sum(1 for p in self.entity_profiles.values()
                     if p.n_observed_events < self.cold_start_threshold)
        print(f"    Built {len(self.entity_profiles)} entity profiles")
        print(f"    Built {len(self.type_profiles)} entity-type aggregate profiles")
        print(f"    Cold-start entities (<{self.cold_start_threshold} events): {n_cold}")

    def fit_rolling(self, events_df, labels_df, reference_date):
        """
        Build profiles using only events within the rolling window ending
        at reference_date. Used for drift handling.

        Args:
            events_df: Full DataFrame of events
            labels_df: DataFrame with 'label' column
            reference_date: End of the rolling window (datetime)
        """
        window_start = reference_date - timedelta(days=self.rolling_window_days)

        timestamps = pd.to_datetime(events_df["timestamp"])
        window_mask = (timestamps >= window_start) & (timestamps < reference_date)

        windowed_events = events_df[window_mask]
        windowed_labels = labels_df[window_mask]

        self.fit(windowed_events, windowed_labels)

    def score_event(self, event):
        """
        Score a single event against its entity's profile.

        Returns:
            dict with: baseline_score, sub_scores, is_cold_start, geo_velocity_kmh
        """
        entity_id = event["entity_id"]
        entity_type = event["entity_type"]

        # Determine which profile to use
        is_cold_start = False
        if entity_id in self.entity_profiles:
            profile = self.entity_profiles[entity_id]
            if profile.n_observed_events < self.cold_start_threshold:
                # Use type-level aggregate instead
                is_cold_start = True
                profile = self.type_profiles.get(entity_type, profile)
        else:
            # Completely unseen entity
            is_cold_start = True
            if entity_type in self.type_profiles:
                profile = self.type_profiles[entity_type]
            else:
                # No profile at all — return neutral score
                return {
                    "baseline_score": 0.3,
                    "sub_scores": {k: 0.0 for k in SCORE_WEIGHTS},
                    "is_cold_start": True,
                    "geo_velocity_kmh": 0.0,
                }

        # Compute score
        score, sub_scores, geo_velocity = compute_baseline_score(
            event, profile, is_cold_start
        )

        # Update last event tracking for geo-velocity (for next event)
        if entity_id in self.entity_profiles:
            actual_profile = self.entity_profiles[entity_id]
            actual_profile.last_event_time = pd.to_datetime(event["timestamp"])
            actual_profile.last_event_geo = event["geo_location"]

        return {
            "baseline_score": round(score, 4),
            "sub_scores": {k: round(v, 4) for k, v in sub_scores.items()},
            "is_cold_start": is_cold_start,
            "geo_velocity_kmh": round(geo_velocity, 2),
        }

    def score_dataframe(self, events_df):
        """
        Score all events in a DataFrame.

        Returns:
            DataFrame with added columns: baseline_score, hour_zscore, geo_novelty,
            resource_novelty, duration_zscore, auth_method_novelty, device_novelty,
            is_cold_start, geo_velocity_kmh
        """
        print(f"[*] Scoring {len(events_df)} events against baseline profiles...")

        n = len(events_df)
        # Pre-allocate output arrays
        baseline_scores = np.zeros(n)
        hour_zscores = np.zeros(n)
        geo_novelties = np.zeros(n)
        resource_novelties = np.zeros(n)
        duration_zscores = np.zeros(n)
        auth_novelties = np.zeros(n)
        device_novelties = np.zeros(n)
        cold_starts = np.zeros(n, dtype=bool)
        geo_velocities = np.zeros(n)

        # Extract columns as arrays for fast access
        entity_ids = events_df["entity_id"].values
        entity_types = events_df["entity_type"].values
        timestamps = pd.to_datetime(events_df["timestamp"])
        hours = timestamps.dt.hour.values.astype(float)
        geos = events_df["geo_location"].values
        resources = events_df["resource_accessed"].values
        durations = events_df["session_duration"].values.astype(float)
        auth_methods = events_df["auth_method"].values
        fingerprints = events_df["device_fingerprint"].values

        # Build entity_id -> profile lookup (with cold-start fallback)
        for i in range(n):
            eid = entity_ids[i]
            etype = entity_types[i]

            # Determine profile
            is_cold = False
            if eid in self.entity_profiles:
                profile = self.entity_profiles[eid]
                if profile.n_observed_events < self.cold_start_threshold:
                    is_cold = True
                    profile = self.type_profiles.get(etype, profile)
            elif etype in self.type_profiles:
                is_cold = True
                profile = self.type_profiles[etype]
            else:
                # No profile at all
                baseline_scores[i] = 0.3
                cold_starts[i] = True
                continue

            cold_starts[i] = is_cold

            # 1. Hour z-score
            if profile.hour_std >= 0.5:
                z = abs(hours[i] - profile.hour_mean) / profile.hour_std
                z_wrap = abs(24 - abs(hours[i] - profile.hour_mean)) / profile.hour_std
                raw_hz = min(z, z_wrap)
            else:
                raw_hz = 0.0
            hour_zscores[i] = min(raw_hz / 4.0, 1.0)

            # 2. Geo novelty
            geo_novelties[i] = 0.0 if (not profile.home_geo_set or geos[i] in profile.home_geo_set) else 1.0

            # 3. Resource novelty
            if profile.total_resource_accesses > 0:
                freq = profile.resource_frequency.get(resources[i], 0)
                resource_novelties[i] = 1.0 - freq / profile.total_resource_accesses
            
            # 4. Duration z-score
            if profile.session_duration_std >= 1.0:
                raw_dz = abs(durations[i] - profile.session_duration_mean) / profile.session_duration_std
                duration_zscores[i] = min(raw_dz / 4.0, 1.0)

            # 5. Auth method novelty
            if profile.primary_auth_method:
                auth_novelties[i] = 0.0 if auth_methods[i] == profile.primary_auth_method else 1.0

            # 6. Device novelty
            if profile.known_device_fingerprints:
                device_novelties[i] = 0.0 if fingerprints[i] in profile.known_device_fingerprints else 1.0

            # Cold-start dampening
            if is_cold:
                geo_novelties[i] *= 0.5
                resource_novelties[i] *= 0.5
                device_novelties[i] *= 0.5

            # Combined score
            combined = (
                SCORE_WEIGHTS["hour_zscore"] * hour_zscores[i] +
                SCORE_WEIGHTS["geo_novelty"] * geo_novelties[i] +
                SCORE_WEIGHTS["resource_novelty"] * resource_novelties[i] +
                SCORE_WEIGHTS["duration_zscore"] * duration_zscores[i] +
                SCORE_WEIGHTS["auth_method_novelty"] * auth_novelties[i] +
                SCORE_WEIGHTS["device_novelty"] * device_novelties[i]
            )
            baseline_scores[i] = min(max(combined, 0.0), 1.0)

        scores_df = pd.DataFrame({
            "baseline_score": np.round(baseline_scores, 4),
            "hour_zscore": np.round(hour_zscores, 4),
            "geo_novelty": np.round(geo_novelties, 4),
            "resource_novelty": np.round(resource_novelties, 4),
            "duration_zscore": np.round(duration_zscores, 4),
            "auth_method_novelty": np.round(auth_novelties, 4),
            "device_novelty": np.round(device_novelties, 4),
            "is_cold_start": cold_starts,
            "geo_velocity_kmh": np.round(geo_velocities, 2),
        }, index=events_df.index)

        print(f"    Scored {len(scores_df)} events")
        print(f"    Score distribution: mean={scores_df['baseline_score'].mean():.4f}, "
              f"std={scores_df['baseline_score'].std():.4f}, "
              f"max={scores_df['baseline_score'].max():.4f}")

        return scores_df

    def get_profiles_df(self):
        """Export all entity profiles as a DataFrame."""
        rows = [p.to_dict() for p in self.entity_profiles.values()]
        return pd.DataFrame(rows)

    def check_drift(self, entity_id, current_date, events_df, labels_df):
        """
        Check if an entity's profile has significantly changed between the
        full-history profile and a recent-window profile.

        Returns:
            is_drift_flagged (bool): True if the entity's recent behavior
            differs significantly from its historical profile.
        """
        if entity_id not in self.entity_profiles:
            return False

        full_profile = self.entity_profiles[entity_id]

        # Build a recent-window profile
        timestamps = pd.to_datetime(events_df["timestamp"])
        window_start = current_date - timedelta(days=self.rolling_window_days)
        entity_mask = events_df["entity_id"] == entity_id
        normal_mask = labels_df["label"] == "normal"
        window_mask = (timestamps >= window_start) & (timestamps < current_date)

        recent_events = events_df[entity_mask & normal_mask & window_mask]

        if len(recent_events) < 3:
            return False

        recent_profile = EntityProfile(entity_id, full_profile.entity_type)
        recent_profile.update_from_events(recent_events)

        # Compare: check if resource set, geo set, or auth method have changed
        old_resources = set(full_profile.resource_frequency.keys())
        new_resources = set(recent_profile.resource_frequency.keys())
        resource_change = len(new_resources - old_resources) / max(len(old_resources), 1)

        geo_change = len(recent_profile.home_geo_set - full_profile.home_geo_set)

        auth_changed = (recent_profile.primary_auth_method !=
                        full_profile.primary_auth_method)

        # Flag drift if significant changes detected
        return resource_change > 0.3 or geo_change > 0 or auth_changed


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check: load data, build profiles, score, compare
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "synthetic_logs.csv")

    if not os.path.exists(data_path):
        print(f"[!] Data file not found: {data_path}")
        print("    Run data/generate_data.py first.")
        exit(1)

    print(f"[*] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    labels = df[["label"]].copy()

    # Use first 30 days for profiling, rest for scoring
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].min() + timedelta(days=30)
    train_mask = df["timestamp"] < cutoff
    test_mask = df["timestamp"] >= cutoff

    train_df = df[train_mask].copy()
    train_labels = labels[train_mask].copy()
    test_df = df[test_mask].copy()
    test_labels = labels[test_mask].copy()

    # Reset timestamp to string for scoring
    train_df["timestamp"] = train_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    test_df["timestamp"] = test_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Build profiles
    profiler = BaselineProfiler()
    profiler.fit(train_df, train_labels)

    # Score test set
    scores = profiler.score_dataframe(test_df)

    # Compare scores: normal vs attacks
    test_with_scores = test_df.copy()
    test_with_scores["baseline_score"] = scores["baseline_score"].values

    normal_scores = test_with_scores[test_labels["label"] == "normal"]["baseline_score"]
    attack_scores = test_with_scores[test_labels["label"] != "normal"]["baseline_score"]

    print(f"\n[*] Score comparison (test set):")
    print(f"    Normal  - mean: {normal_scores.mean():.4f}, median: {normal_scores.median():.4f}")
    if len(attack_scores) > 0:
        print(f"    Attacks - mean: {attack_scores.mean():.4f}, median: {attack_scores.median():.4f}")
        print(f"    Separation ratio: {attack_scores.mean() / max(normal_scores.mean(), 0.001):.2f}x")
    else:
        print("    No attack events in test set")

    # Per-attack-type breakdown
    print(f"\n[*] Per-attack-type scores:")
    for label in test_labels["label"].unique():
        if label == "normal":
            continue
        mask = test_labels["label"] == label
        if mask.sum() > 0:
            label_scores = test_with_scores[mask]["baseline_score"]
            print(f"    {label:25s}: mean={label_scores.mean():.4f}, "
                  f"count={len(label_scores)}")

    # Save profiles
    profiles_df = profiler.get_profiles_df()
    profiles_path = os.path.join(script_dir, "..", "data", "entity_profiles.csv")
    profiles_df.to_csv(profiles_path, index=False)
    print(f"\n[OK] Profiles saved to {profiles_path}")
