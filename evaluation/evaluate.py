"""
Evaluation & Metrics Pipeline
=============================
Computes and visualizes evaluation metrics per Honeywell's evaluation criteria:
1. Binary Anomaly Detection: Precision, Recall, F1, PR-AUC & PR Curve plot
2. Multi-class Classification: Confusion Matrix heatmap across all attack categories
3. Alert Budget Precision: Precision@top-0.5%, @top-1%, @top-3%
4. Cold-Start Slice: Metrics for entities with <5 events vs. rest
5. Concept Drift Slice: Demonstration of rolling window adaptation
6. Generates metrics_summary.json for the dashboard.
"""

import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
METRICS_JSON_PATH = os.path.join(DATA_DIR, "metrics_summary.json")

# Ensure figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_system():
    print("=" * 70)
    print("SYSTEM EVALUATION & METRICS GENERATION")
    print("=" * 70)

    scored_path = os.path.join(DATA_DIR, "scored_events.csv")
    alerts_path = os.path.join(DATA_DIR, "alerts.csv")

    if not os.path.exists(scored_path):
        print(f"[!] Scored events file not found: {scored_path}")
        print("    Run: python models/train.py first.")
        sys.exit(1)

    df_scored = pd.read_csv(scored_path)
    df_alerts = pd.read_csv(alerts_path) if os.path.exists(alerts_path) else pd.DataFrame()

    print(f"[*] Loaded {len(df_scored)} scored test events.")

    # 1. Binary anomaly ground truth and continuous risk score
    y_true_binary = (df_scored["ground_truth_label"] != "normal").astype(int)
    y_scores = df_scored["risk_score"].values

    # ── Metric 1: Precision, Recall, PR-AUC & PR Curve ──────────────────────
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#3b82f6', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Binary Anomaly Detection)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    pr_curve_path = os.path.join(FIGURES_DIR, "pr_curve.png")
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()
    print(f"[OK] Saved PR Curve -> {pr_curve_path} (PR-AUC: {pr_auc:.4f})")

    # ── Metric 2: Alert Budget Precision ────────────────────────────────────
    n_total = len(df_scored)
    budget_percentages = [0.005, 0.01, 0.03]
    alert_budget_results = {}

    print("\n[*] Alert Budget Precision:")
    df_sorted = df_scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    df_sorted["is_attack"] = (df_sorted["ground_truth_label"] != "normal").astype(int)

    for pct in budget_percentages:
        k = max(1, int(n_total * pct))
        top_k = df_sorted.head(k)
        prec_at_k = top_k["is_attack"].sum() / k
        rec_at_k = top_k["is_attack"].sum() / max(1, y_true_binary.sum())
        key = f"top_{pct*100:.1f}%"
        alert_budget_results[key] = {
            "top_k_count": k,
            "precision": round(float(prec_at_k), 4),
            "recall": round(float(rec_at_k), 4)
        }
        print(f"    Precision@{key:8s} ({k:5d} events): {prec_at_k*100:6.2f}% (Recall: {rec_at_k*100:6.2f}%)")

    # Bar chart for alert budget precision
    plt.figure(figsize=(7, 4.5))
    keys = list(alert_budget_results.keys())
    precs = [alert_budget_results[k]["precision"] * 100 for k in keys]
    bars = plt.bar(keys, precs, color=['#10b981', '#3b82f6', '#8b5cf6'], width=0.5)
    plt.ylabel('Precision (%)', fontsize=12)
    plt.title('Precision at Realistic Analyst Alert Budgets', fontsize=13, fontweight='bold')
    plt.ylim(0, 105)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    budget_chart_path = os.path.join(FIGURES_DIR, "alert_budget_table.png")
    plt.savefig(budget_chart_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Alert Budget Chart -> {budget_chart_path}")

    # ── Metric 3: Multi-class Confusion Matrix ──────────────────────────────
    classes = sorted(df_scored["ground_truth_label"].unique())
    y_true_cls = df_scored["ground_truth_label"].values
    y_pred_cls = df_scored["predicted_type"].values

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
    plt.ylabel('True Category (Ground Truth)', fontsize=12, fontweight='bold')
    plt.title('Multi-Class Anomaly Classification Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Confusion Matrix -> {cm_path}")

    # ── Metric 4: Cold-Start Slice ──────────────────────────────────────────
    cold_mask = df_scored["is_cold_start"].astype(bool)
    n_cold = cold_mask.sum()
    n_warm = len(df_scored) - n_cold

    cold_slice_results = {
        "cold_start_events_count": int(n_cold),
        "warm_events_count": int(n_warm),
    }

    if n_cold > 0:
        cold_prec = precision_score(y_true_binary[cold_mask], (y_scores[cold_mask] > 0.2).astype(int), zero_division=0)
        cold_rec = recall_score(y_true_binary[cold_mask], (y_scores[cold_mask] > 0.2).astype(int), zero_division=0)
        cold_slice_results["cold_start_precision"] = round(float(cold_prec), 4)
        cold_slice_results["cold_start_recall"] = round(float(cold_rec), 4)

    warm_prec = precision_score(y_true_binary[~cold_mask], (y_scores[~cold_mask] > 0.2).astype(int), zero_division=0)
    warm_rec = recall_score(y_true_binary[~cold_mask], (y_scores[~cold_mask] > 0.2).astype(int), zero_division=0)
    cold_slice_results["warm_precision"] = round(float(warm_prec), 4)
    cold_slice_results["warm_recall"] = round(float(warm_rec), 4)

    print(f"[*] Cold-Start Slice: Warm events precision={warm_prec*100:.2f}%, recall={warm_rec*100:.2f}%")

    # ── Metric 5: Concept Drift Demonstration Plot ──────────────────────────
    drift_entities = df_scored[df_scored["ground_truth_label"] == "insider_drift"]["entity_id"].unique()
    target_entity = drift_entities[0] if len(drift_entities) > 0 else df_scored["entity_id"].iloc[0]

    df_entity = df_scored[df_scored["entity_id"] == target_entity].copy()
    df_entity["_ts"] = pd.to_datetime(df_entity["timestamp"])
    df_entity = df_entity.sort_values("_ts")

    plt.figure(figsize=(10, 4.5))
    plt.plot(df_entity["_ts"], df_entity["risk_score"], label='Risk Score', color='#ef4444', lw=2)
    plt.axhline(0.2, color='#6b7280', linestyle='--', label='Alert Threshold (0.20)')
    plt.title(f'Concept Drift & Risk Score Adaptation for Entity: {target_entity}', fontsize=13, fontweight='bold')
    plt.xlabel('Timestamp', fontsize=11)
    plt.ylabel('Risk Score', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    drift_chart_path = os.path.join(FIGURES_DIR, "drift_demo.png")
    plt.savefig(drift_chart_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Concept Drift Plot -> {drift_chart_path}")

    # ── Summary JSON ────────────────────────────────────────────────────────
    summary_metrics = {
        "eval_timestamp": datetime.now().isoformat(),
        "total_test_events": int(len(df_scored)),
        "total_alerts_generated": int(len(df_alerts)),
        "pr_auc": round(float(pr_auc), 4),
        "alert_budget_precision": alert_budget_results,
        "cold_start_slice": cold_slice_results,
        "attack_distribution_predicted": df_scored["predicted_type"].value_counts().to_dict(),
        "attack_distribution_actual": df_scored["ground_truth_label"].value_counts().to_dict(),
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\n[OK] Evaluation complete! Metrics saved -> {METRICS_JSON_PATH}")


if __name__ == "__main__":
    evaluate_system()
