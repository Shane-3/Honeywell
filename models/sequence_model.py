"""
Sequence-Aware Detection Model (Markov MVP)
=============================================
Builds per-entity-type transition probability matrices over resource access
sequences. Scores events by negative log-likelihood of the observed transition
under the learned model.

Additionally implements a multi-day rolling window detector for low-and-slow
exfiltration patterns that span across sessions.

This is genuinely sequence-aware (satisfies FR-3.1) without deep-learning
training risk. The model is trained on normal-labeled sequences only (one-class
framing, FR-3.3).
"""

import json
import math
import os
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Smoothing constant for unseen transitions (Laplace smoothing)
LAPLACE_ALPHA = 1e-4

# Sliding window for session-level sequences (how many recent events per entity)
SESSION_WINDOW_SIZE = 10

# Multi-day window for low-and-slow detection
LOW_SLOW_WINDOW_DAYS = 7
LOW_SLOW_OFF_HOURS = set([22, 23, 0, 1, 2, 3, 4, 5])

# Score normalization: max NLL before clipping (controls sensitivity)
MAX_NLL = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Markov Transition Model
# ─────────────────────────────────────────────────────────────────────────────

def _def_dict_float():
    return defaultdict(float)


class MarkovTransitionModel:
    """
    First-order Markov chain over resource access sequences.
    Learns P(resource_t | resource_{t-1}) from normal event sequences.
    """

    def __init__(self, laplace_alpha=LAPLACE_ALPHA):
        self.laplace_alpha = laplace_alpha
        # transition_counts[from_resource][to_resource] = count
        self.transition_counts = defaultdict(_def_dict_float)
        # Total transitions from each resource
        self.transition_totals = defaultdict(float)
        # Set of all seen resources
        self.vocabulary = set()
        # Initial state distribution
        self.initial_counts = defaultdict(float)
        self.total_sequences = 0

    def fit(self, sequences):
        """
        Learn transition probabilities from a list of resource sequences.

        Args:
            sequences: list of lists, where each inner list is an ordered
                       sequence of resource names accessed by an entity.
        """
        self.transition_counts = defaultdict(_def_dict_float)
        self.transition_totals = defaultdict(float)
        self.initial_counts = defaultdict(float)
        self.vocabulary = set()
        self.total_sequences = 0

        for seq in sequences:
            if len(seq) == 0:
                continue
            self.total_sequences += 1
            self.vocabulary.update(seq)

            # Count initial state
            self.initial_counts[seq[0]] += 1

            # Count transitions
            for i in range(len(seq) - 1):
                from_res = seq[i]
                to_res = seq[i + 1]
                self.transition_counts[from_res][to_res] += 1
                self.transition_totals[from_res] += 1

    def transition_probability(self, from_resource, to_resource):
        """
        Get P(to_resource | from_resource) with Laplace smoothing.
        """
        vocab_size = max(len(self.vocabulary), 1)
        count = self.transition_counts[from_resource][to_resource]
        total = self.transition_totals[from_resource]

        # Laplace smoothed probability
        prob = (count + self.laplace_alpha) / (total + self.laplace_alpha * vocab_size)
        return prob

    def initial_probability(self, resource):
        """Get P(resource as first in sequence)."""
        if self.total_sequences == 0:
            return 1.0 / max(len(self.vocabulary), 1)
        vocab_size = max(len(self.vocabulary), 1)
        count = self.initial_counts[resource]
        prob = (count + self.laplace_alpha) / (self.total_sequences + self.laplace_alpha * vocab_size)
        return prob

    def precompute_log_probs(self):
        """Precompute log transition probabilities for O(1) lookups."""
        self._log_trans = {}
        self._log_init = {}
        vocab_size = max(len(self.vocabulary), 1)

        for from_res in self.vocabulary:
            total = self.transition_totals[from_res]
            for to_res in self.vocabulary:
                count = self.transition_counts[from_res][to_res]
                prob = (count + self.laplace_alpha) / (total + self.laplace_alpha * vocab_size)
                self._log_trans[(from_res, to_res)] = math.log(max(prob, 1e-10))

        # Default log prob for unseen transitions
        self._default_log_prob = math.log(self.laplace_alpha / (self.laplace_alpha * vocab_size))

        for res in self.vocabulary:
            count = self.initial_counts[res]
            prob = (count + self.laplace_alpha) / (self.total_sequences + self.laplace_alpha * vocab_size)
            self._log_init[res] = math.log(max(prob, 1e-10))

        self._default_log_init = math.log(1.0 / vocab_size) if vocab_size > 0 else 0.0

    def sequence_nll(self, sequence):
        """
        Compute negative log-likelihood of a resource sequence under the model.
        Higher NLL = more anomalous sequence. Uses precomputed log probs.
        """
        if len(sequence) == 0:
            return 0.0

        nll = 0.0

        # Initial state probability
        nll -= self._log_init.get(sequence[0], self._default_log_init)

        # Transition probabilities
        for i in range(len(sequence) - 1):
            nll -= self._log_trans.get((sequence[i], sequence[i + 1]), self._default_log_prob)

        # Normalize by sequence length
        nll /= max(len(sequence), 1)

        return nll


# ─────────────────────────────────────────────────────────────────────────────
# Low-and-Slow Detector
# ─────────────────────────────────────────────────────────────────────────────

class LowAndSlowDetector:
    """
    Detects gradual, off-hours resource access patterns building up over days.
    Maintains a per-entity rolling count of off-hours accesses and scores
    based on upward trend.
    """

    def __init__(self, window_days=LOW_SLOW_WINDOW_DAYS,
                 off_hours=LOW_SLOW_OFF_HOURS):
        self.window_days = window_days
        self.off_hours = off_hours
        # entity_id -> list of (timestamp, resource) for off-hours accesses
        self.entity_offhours_history = defaultdict(list)
        # entity_id -> baseline off-hours rate (from training)
        self.entity_baseline_rate = {}

    def fit(self, events_df, labels_df):
        """
        Learn baseline off-hours access rates per entity from normal events.
        """
        normal_mask = labels_df["label"] == "normal"
        normal_events = events_df[normal_mask].copy()
        normal_events["_hour"] = pd.to_datetime(normal_events["timestamp"]).dt.hour

        # Calculate per-entity off-hours access rate
        for entity_id, group in normal_events.groupby("entity_id"):
            total = len(group)
            offhours = group["_hour"].isin(self.off_hours).sum()
            self.entity_baseline_rate[entity_id] = offhours / max(total, 1)

    def score_event(self, event, event_time):
        """
        Score an event for low-and-slow exfiltration pattern.

        Returns:
            float: 0.0 (normal) to 1.0 (highly suspicious off-hours trend)
        """
        entity_id = event["entity_id"]
        hour = event_time.hour

        # Only track off-hours events
        if hour not in self.off_hours:
            return 0.0

        # Add to history
        self.entity_offhours_history[entity_id].append(
            (event_time, event["resource_accessed"])
        )

        # Prune old entries outside window
        window_start = event_time - timedelta(days=self.window_days)
        self.entity_offhours_history[entity_id] = [
            (t, r) for t, r in self.entity_offhours_history[entity_id]
            if t >= window_start
        ]

        recent_count = len(self.entity_offhours_history[entity_id])
        baseline_rate = self.entity_baseline_rate.get(entity_id, 0.1)

        # Expected off-hours events in the window
        # (rough estimate based on entity's typical daily event count)
        expected_per_window = max(1, baseline_rate * 50 * self.window_days)

        # Score: ratio of observed to expected, capped
        if expected_per_window > 0:
            ratio = recent_count / expected_per_window
            score = min(max((ratio - 1.0) / 3.0, 0.0), 1.0)
        else:
            score = min(recent_count / 10.0, 1.0)

        return score


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Model (combines Markov + Low-and-Slow)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceModel:
    """
    Full sequence-aware detection model combining:
    1. Markov transition model for resource access sequences
    2. Low-and-slow multi-day rolling window detector
    """

    def __init__(self):
        # Per entity-type Markov models
        self.markov_models = {}
        # Per-entity recent resource sequence (sliding window)
        self.entity_recent_resources = defaultdict(list)
        # Low-and-slow detector
        self.low_slow_detector = LowAndSlowDetector()

    def fit(self, events_df, labels_df):
        """
        Train on normal-labeled event sequences.
        Builds per-entity-type Markov models and baseline off-hours rates.
        """
        normal_mask = labels_df["label"] == "normal"
        normal_events = events_df[normal_mask].copy()
        normal_events["_ts"] = pd.to_datetime(normal_events["timestamp"])
        normal_events = normal_events.sort_values("_ts")

        print(f"[*] Training sequence model on {len(normal_events)} normal events...")

        # Build per-entity resource sequences, then group by entity type
        entity_sequences = defaultdict(list)
        type_sequences = defaultdict(list)

        for entity_id, group in normal_events.groupby("entity_id"):
            entity_type = group["entity_type"].iloc[0]
            resources = group["resource_accessed"].tolist()

            # Split into session-sized chunks
            for i in range(0, len(resources), SESSION_WINDOW_SIZE):
                chunk = resources[i:i + SESSION_WINDOW_SIZE]
                if len(chunk) >= 2:
                    entity_sequences[entity_id].append(chunk)
                    type_sequences[entity_type].append(chunk)

        # Train per-entity-type Markov models
        for entity_type, sequences in type_sequences.items():
            model = MarkovTransitionModel()
            model.fit(sequences)
            model.precompute_log_probs()
            self.markov_models[entity_type] = model
            print(f"    {entity_type}: {len(sequences)} sequences, "
                  f"{len(model.vocabulary)} resources in vocabulary")

        # Train low-and-slow detector
        self.low_slow_detector.fit(events_df, labels_df)
        print(f"    Low-and-slow baseline rates computed for "
              f"{len(self.low_slow_detector.entity_baseline_rate)} entities")

    def score_event(self, event):
        """
        Score a single event using both the Markov model and low-and-slow detector.

        Returns:
            dict with: sequence_score, markov_nll, low_slow_score
        """
        entity_id = event["entity_id"]
        entity_type = event["entity_type"]
        resource = event["resource_accessed"]
        event_time = pd.to_datetime(event["timestamp"])

        # Update entity's recent resource sequence
        self.entity_recent_resources[entity_id].append(resource)
        # Keep only recent window
        if len(self.entity_recent_resources[entity_id]) > SESSION_WINDOW_SIZE:
            self.entity_recent_resources[entity_id] = \
                self.entity_recent_resources[entity_id][-SESSION_WINDOW_SIZE:]

        # 1. Markov transition score
        markov_nll = 0.0
        if entity_type in self.markov_models:
            recent_seq = self.entity_recent_resources[entity_id]
            if len(recent_seq) >= 2:
                markov_nll = self.markov_models[entity_type].sequence_nll(recent_seq)

        # Normalize NLL to 0-1
        markov_score = min(markov_nll / MAX_NLL, 1.0)

        # 2. Low-and-slow score
        low_slow_score = self.low_slow_detector.score_event(event, event_time)

        # Combined sequence score (weighted)
        sequence_score = 0.7 * markov_score + 0.3 * low_slow_score

        return {
            "sequence_score": round(sequence_score, 4),
            "markov_nll": round(markov_nll, 4),
            "markov_score": round(markov_score, 4),
            "low_slow_score": round(low_slow_score, 4),
        }

    def score_dataframe(self, events_df):
        """
        Score all events in a DataFrame. Events MUST be sorted by timestamp.

        Returns:
            DataFrame with: sequence_score, markov_nll, markov_score, low_slow_score
        """
        print(f"[*] Scoring {len(events_df)} events with sequence model...")

        # Reset state for clean scoring
        self.entity_recent_resources = defaultdict(list)
        self.low_slow_detector.entity_offhours_history = defaultdict(list)

        n = len(events_df)
        seq_scores = np.zeros(n)
        markov_nlls = np.zeros(n)
        markov_scores_arr = np.zeros(n)
        low_slow_scores = np.zeros(n)

        entity_ids = events_df["entity_id"].values
        entity_types = events_df["entity_type"].values
        resources = events_df["resource_accessed"].values
        hours = pd.to_datetime(events_df["timestamp"]).dt.hour.values

        off_hours = self.low_slow_detector.off_hours
        _offhours_counts = {}
        _last_resource = {}  # entity_id -> last resource accessed

        # Precompute model lookups
        _models = self.markov_models

        for i in range(n):
            eid = entity_ids[i]
            etype = entity_types[i]
            res = resources[i]

            # Markov score: only last transition (O(1) per event)
            markov_nll = 0.0
            if etype in _models:
                prev_res = _last_resource.get(eid)
                if prev_res is not None:
                    log_p = _models[etype]._log_trans.get((prev_res, res), _models[etype]._default_log_prob)
                    markov_nll = -log_p
            _last_resource[eid] = res
            markov_nlls[i] = markov_nll
            m_score = min(markov_nll / MAX_NLL, 1.0)
            markov_scores_arr[i] = m_score

            # Low-slow score (counter-based)
            hour = hours[i]
            ls_score = 0.0
            if hour in off_hours:
                _offhours_counts[eid] = _offhours_counts.get(eid, 0) + 1
                recent_count = _offhours_counts[eid]
                baseline_rate = self.low_slow_detector.entity_baseline_rate.get(eid, 0.1)
                expected = max(1, baseline_rate * 50 * self.low_slow_detector.window_days)
                ratio = recent_count / expected
                ls_score = min(max((ratio - 1.0) / 3.0, 0.0), 1.0)
            low_slow_scores[i] = ls_score

            seq_scores[i] = 0.7 * m_score + 0.3 * ls_score

        scores_df = pd.DataFrame({
            "sequence_score": np.round(seq_scores, 4),
            "markov_nll": np.round(markov_nlls, 4),
            "markov_score": np.round(markov_scores_arr, 4),
            "low_slow_score": np.round(low_slow_scores, 4),
        }, index=events_df.index)

        print(f"    Scored {len(scores_df)} events")
        print(f"    Sequence score distribution: "
              f"mean={scores_df['sequence_score'].mean():.4f}, "
              f"std={scores_df['sequence_score'].std():.4f}, "
              f"max={scores_df['sequence_score'].max():.4f}")

        return scores_df


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "synthetic_logs.csv")

    if not os.path.exists(data_path):
        print(f"[!] Data file not found: {data_path}")
        print("    Run data/generate_data.py first.")
        exit(1)

    print(f"[*] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    labels = df[["label"]].copy()

    # Time-based split: first 30 days for training, rest for testing
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].min() + timedelta(days=30)
    train_mask = df["timestamp"] < cutoff
    test_mask = df["timestamp"] >= cutoff

    train_df = df[train_mask].copy()
    train_labels = labels[train_mask].copy()
    test_df = df[test_mask].copy().sort_values("timestamp")
    test_labels = labels[test_mask].copy()

    # Convert timestamps back to strings
    train_df["timestamp"] = train_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    test_df["timestamp"] = test_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Train
    model = SequenceModel()
    model.fit(train_df, train_labels)

    # Score test set
    scores = model.score_dataframe(test_df)

    # Compare scores: normal vs attacks
    test_with_scores = test_df.copy()
    test_with_scores["sequence_score"] = scores["sequence_score"].values

    normal_scores = test_with_scores[test_labels["label"] == "normal"]["sequence_score"]
    attack_scores = test_with_scores[test_labels["label"] != "normal"]["sequence_score"]

    print(f"\n[*] Score comparison (test set):")
    print(f"    Normal  - mean: {normal_scores.mean():.4f}, median: {normal_scores.median():.4f}")
    if len(attack_scores) > 0:
        print(f"    Attacks - mean: {attack_scores.mean():.4f}, median: {attack_scores.median():.4f}")
        print(f"    Separation ratio: {attack_scores.mean() / max(normal_scores.mean(), 0.001):.2f}x")
    else:
        print("    No attack events in test set")

    # Per-attack-type breakdown
    print(f"\n[*] Per-attack-type sequence scores:")
    for label in sorted(test_labels["label"].unique()):
        if label == "normal":
            continue
        mask = test_labels["label"] == label
        if mask.sum() > 0:
            label_scores = test_with_scores.loc[mask.values, "sequence_score"]
            print(f"    {label:25s}: mean={label_scores.mean():.4f}, count={len(label_scores)}")
