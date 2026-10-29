import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def plot_kaplan_meier_curves(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Dict[str, Any]:
    """
    Stratifies patients into High-Risk vs Low-Risk groups using median predicted risk,
    computes the log-rank test p-value, and plots Kaplan-Meier survival curves.
    """
    median_risk = np.median(risk_scores)
    high_risk = risk_scores > median_risk
    low_risk = ~high_risk

    # Log-rank test
    lr_results = logrank_test(
        durations_A=times[high_risk],
        durations_B=times[low_risk],
        event_observed_A=events[high_risk],
        event_observed_B=events[low_risk]
    )
    p_value = float(lr_results.p_value)
    test_stat = float(lr_results.test_statistic)

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    kmf_high.fit(times[high_risk], events[high_risk], label=f"High Risk (N={np.sum(high_risk)}, E={int(np.sum(events[high_risk]))})")
    kmf_low.fit(times[low_risk], events[low_risk], label=f"Low Risk (N={np.sum(low_risk)}, E={int(np.sum(events[low_risk]))})")

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        standalone = True

    kmf_low.plot_survival_function(ax=ax, color="#1f77b4", linewidth=2.0, ci_show=True, ci_alpha=0.15)
    kmf_high.plot_survival_function(ax=ax, color="#d62728", linewidth=2.0, ci_show=True, ci_alpha=0.15)

    p_str = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
    ax.set_title(f"{model_name}\nLog-Rank {p_str} (stat={test_stat:.2f})", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Time (Days)", fontsize=10)
    ax.set_ylabel("Overall Survival Probability", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    if standalone and save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "model": model_name,
        "p_value": p_value,
        "test_statistic": test_stat,
        "high_risk_n": int(np.sum(high_risk)),
        "high_risk_events": int(np.sum(events[high_risk])),
        "low_risk_n": int(np.sum(low_risk)),
        "low_risk_events": int(np.sum(events[low_risk]))
    }

def plot_all_models_km_comparison(
    models_oof_dict: Dict[str, np.ndarray],
    times: np.ndarray,
    events: np.ndarray,
    save_path: str
) -> Dict[str, Any]:
    """
    Creates a 2x2 multi-panel figure comparing Kaplan-Meier survival curves
    for all evaluated models on the out-of-fold patient cohort.
    """
    model_keys = list(models_oof_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    stats_summary = {}

    for idx, key in enumerate(model_keys):
        ax = axes[idx]
        risks = models_oof_dict[key]
        stats = plot_kaplan_meier_curves(
            risk_scores=risks,
            times=times,
            events=events,
            model_name=key,
            ax=ax
        )
        stats_summary[key] = stats

    plt.suptitle("Out-of-Fold Kaplan-Meier Survival Risk Stratification (TCGA-BRCA 298-Patient Cohort)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return stats_summary
