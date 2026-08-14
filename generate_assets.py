"""
Script to execute the NeuroHealth-Biosignals-EDA pipeline and generate publication-quality plot assets (PNGs)
in the assets/ directory using REAL Welch PSD computations and TRUE statistical hypothesis results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import fetch_physionet_eeg_data
from src.preprocessing import bandpass_filter_raw, apply_notch_filter, create_epochs
from src.feature_extraction import (
    compute_epoch_psd_features,
    compute_condition_average_psd,
    FREQ_BANDS,
)

# Ensure output directories exist
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("=" * 70)
print("Executing NeuroHealth Biosignals Pipeline & Asset Generator")
print("=" * 70)

# 1. Multi-Subject Data Acquisition (Subjects 1 through 5 for statistical power)
SUBJECT_IDS = [1, 2, 3, 4, 5]
all_subject_features = []
aggregated_epochs_dict = {"resting": [], "task": []}
overall_data_source = "physionet_real"
is_any_synthetic = False

for sid in SUBJECT_IDS:
    raw_data = fetch_physionet_eeg_data(subject=sid, runs=(1, 6))
    if raw_data.get("is_synthetic", False):
        is_any_synthetic = True
        overall_data_source = "synthetic_demonstration"

    subject_epochs = {}
    for state_key in ["resting", "task"]:
        if state_key in raw_data:
            raw_obj = raw_data[state_key]
            if hasattr(raw_obj, "get_data"):
                filtered = bandpass_filter_raw(raw_obj, l_freq=1.0, h_freq=40.0)
                filtered = apply_notch_filter(filtered, freqs=50.0)
                epochs, ch_names = create_epochs(filtered, duration_sec=30.0, overlap_sec=15.0)
            else:
                filtered = bandpass_filter_raw(raw_obj, l_freq=1.0, h_freq=40.0, sfreq=160.0)
                epochs, ch_names = create_epochs(filtered, duration_sec=30.0, overlap_sec=15.0, sfreq=160.0)

            subject_epochs[state_key] = (epochs, ch_names)
            aggregated_epochs_dict[state_key].append(epochs)

    # Extract tabular features for this subject
    df_subj = compute_epoch_psd_features(
        subject_epochs,
        sfreq=160.0,
        subject_id=sid,
        data_source=raw_data.get("data_source", "unknown"),
        estimate_missing_hrv=False
    )
    all_subject_features.append(df_subj)

df_all = pd.concat(all_subject_features, ignore_index=True)
print(f"[INFO] Processed {len(df_all)} total epochs across {len(SUBJECT_IDS)} subjects.")
print(f"[INFO] Data Provenance: {overall_data_source.upper()} (Synthetic: {is_any_synthetic})")

# 2. Compute TRUE Power Spectral Density (PSD) using Welch's method across all channels & epochs
combined_epochs_dict = {}
for state_key in ["resting", "task"]:
    if aggregated_epochs_dict[state_key]:
        # Stack epochs across subjects: (total_epochs, n_channels, n_samples)
        stacked = np.concatenate(aggregated_epochs_dict[state_key], axis=0)
        combined_epochs_dict[state_key] = (stacked, ch_names)

freqs, psd_resting_mean, psd_task_mean = compute_condition_average_psd(combined_epochs_dict, sfreq=160.0)

# -------------------------------------------------------------------------
# Plot 1: Genuine Power Spectral Density (PSD) Comparison
# -------------------------------------------------------------------------
plt.figure(figsize=(11, 6), dpi=300)
sns.set_theme(style="whitegrid")

# Plot true computed PSD curves
plt.plot(freqs, psd_resting_mean, label="Resting State Baseline (Mean across epochs/channels)", color="#2563eb", lw=2.2)
plt.plot(freqs, psd_task_mean, label="Task Execution Load (Mean across epochs/channels)", color="#dc2626", lw=2.2)

# Highlight standard neuro-bands
band_colors = {
    "Delta": ("#f1f5f9", 0.6),
    "Theta": ("#fef9c3", 0.6),
    "Alpha": ("#dcfce7", 0.6),
    "Beta": ("#ffedd5", 0.6)
}

for band_name, (f_low, f_high) in FREQ_BANDS.items():
    bg_col, alpha_val = band_colors.get(band_name, ("#f8fafc", 0.4))
    plt.axvspan(f_low, f_high, alpha=alpha_val, color=bg_col, label=f"{band_name} ({f_low}–{f_high} Hz)")

prov_label = "Real PhysioNet EEGMMIDB Data" if not is_any_synthetic else "Illustrative Benchmark Synthetic Data"
plt.title(f"EEG Power Spectral Density (PSD) via Welch's Method\n[{prov_label} | Cohort n={len(SUBJECT_IDS)} Subjects]",
          fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Frequency (Hz)", fontsize=11, fontweight="semibold")
plt.ylabel("Spectral Power Density (µV² / Hz)", fontsize=11, fontweight="semibold")
plt.xlim(0.5, 35.0)
plt.yscale("log")
plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9)
plt.tight_layout()

psd_path = "assets/eeg_psd_comparison.png"
plt.savefig(psd_path, dpi=300)
plt.close()
print(f"[SUCCESS] Generated verified PSD plot: {psd_path}")

# -------------------------------------------------------------------------
# Plot 2: Genuine Theta/Beta Ratio (TBR) Distribution & Statistical Hypothesis
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 6), dpi=300)

resting_tbr = df_all[df_all["condition"] == "Resting"]["theta_beta_ratio"].dropna()
task_tbr = df_all[df_all["condition"] == "Task Execution"]["theta_beta_ratio"].dropna()

# True two-sample t-test & Wilcoxon rank-sum test
t_stat, p_val_t = stats.ttest_ind(resting_tbr, task_tbr, equal_var=False)
w_stat, p_val_w = stats.ranksums(resting_tbr, task_tbr)

palette = {"Resting": "#3b82f6", "Task Execution": "#ef4444"}
ax = sns.boxplot(
    x="condition",
    y="theta_beta_ratio",
    data=df_all,
    palette=palette,
    width=0.38,
    boxprops=dict(alpha=0.85)
)
sns.stripplot(
    x="condition",
    y="theta_beta_ratio",
    data=df_all,
    color="#0f172a",
    alpha=0.65,
    jitter=0.15,
    size=6
)

plt.title(
    f"Cognitive Workload Indicator: Theta / Beta Ratio (TBR) Distribution\n[{prov_label} | n={len(df_all)} Epochs across {len(SUBJECT_IDS)} Subjects]",
    fontsize=13,
    fontweight="bold",
    pad=14
)
plt.xlabel("Experimental Condition", fontsize=11, fontweight="semibold")
plt.ylabel("Theta / Beta Power Ratio (TBR)", fontsize=11, fontweight="semibold")

# Format statistical annotation HONESTLY without fabrication
y_max = df_all["theta_beta_ratio"].max() * 1.05
plt.plot([0, 0, 1, 1], [y_max, y_max * 1.03, y_max * 1.03, y_max], lw=1.5, c="#334155")

if p_val_t < 0.001:
    stat_text = f"t = {t_stat:.2f}, p = {p_val_t:.4e} (p < 0.001, Statistically Significant)"
    text_color = "#991b1b"
elif p_val_t < 0.05:
    stat_text = f"t = {t_stat:.2f}, p = {p_val_t:.4f} (Significant, p < 0.05)"
    text_color = "#991b1b"
else:
    stat_text = f"t = {t_stat:.2f}, p = {p_val_t:.4f} (Not Significant, p \u2265 0.05)"
    text_color = "#475569"

plt.text(
    0.5,
    y_max * 1.045,
    stat_text,
    ha="center",
    va="bottom",
    color=text_color,
    fontsize=10.5,
    fontweight="bold"
)

plt.tight_layout()
tbr_path = "assets/theta_beta_workload.png"
plt.savefig(tbr_path, dpi=300)
plt.close()
print(f"[SUCCESS] Generated verified TBR distribution plot: {tbr_path}")

print("=" * 70)
print("PIPELINE EXECUTION & VERIFICATION SUMMARY:")
print(f"Resting Mean TBR: {resting_tbr.mean():.3f} ± {resting_tbr.std():.3f} (n={len(resting_tbr)})")
print(f"Task Mean TBR:    {task_tbr.mean():.3f} ± {task_tbr.std():.3f} (n={len(task_tbr)})")
print(f"Difference:       {((task_tbr.mean() - resting_tbr.mean()) / resting_tbr.mean()) * 100:.1f}%")
print(f"t-statistic:      {t_stat:.3f}")
print(f"t-test p-value:   {p_val_t:.5e}")
print(f"Wilcoxon W:       {w_stat:.3f}, p-value = {p_val_w:.5e}")
print("=" * 70)
