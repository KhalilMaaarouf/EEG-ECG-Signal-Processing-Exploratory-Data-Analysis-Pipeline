"""
Feature Extraction Module for EEG & ECG Biosignals.

Computes Power Spectral Density (PSD) using Welch's method, extracts standard EEG frequency bands
(Delta, Theta, Alpha, Beta), calculates clinical neuro-markers (Theta/Beta Ratio, Alpha/Beta Ratio),
and computes Heart Rate Variability (HRV) metrics with transparent metadata and simulation flagging.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from scipy.signal import welch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard EEG Frequency Band Definitions (Hz)
FREQ_BANDS = {
    "Delta": (0.5, 4.0),   # Deep sleep, slow-wave activity (0.5 - 4 Hz)
    "Theta": (4.0, 8.0),   # Drowsiness, cognitive fatigue, working memory allocation (4 - 8 Hz)
    "Alpha": (8.0, 12.0),  # Relaxed alert baseline, posterior sensory idling (8 - 12 Hz)
    "Beta": (13.0, 30.0),  # Active mental focus, cognitive workload, stress (13 - 30 Hz)
}


def compute_welch_psd(
    signal_epoch: np.ndarray,
    sfreq: float = 160.0,
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density (PSD) using Welch's method.

    Parameters:
    -----------
    signal_epoch : np.ndarray
        1D array (single channel time series) or 2D array (n_channels, n_samples).
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch estimate (default 2 * sfreq for 0.5 Hz resolution).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        - freqs: Array of sample frequencies (Hz).
        - psd: Power spectral density values (V^2/Hz or uV^2/Hz).
    """
    if nperseg is None:
        nperseg = int(min(2 * sfreq, signal_epoch.shape[-1]))

    freqs, psd = welch(signal_epoch, fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd


def extract_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float]
) -> float:
    """
    Calculate absolute band power via trapezoidal numerical integration over the specified frequency range.
    """
    f_low, f_high = band
    idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
    if not np.any(idx_band):
        return 0.0

    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    
    # Numpy / Scipy trapezoid compatibility
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_func is not None:
        band_power = trapz_func(psd[..., idx_band], dx=freq_res, axis=-1)
    else:
        band_power = np.sum(psd[..., idx_band], axis=-1) * freq_res

    return float(np.mean(band_power))


def calculate_hrv_metrics(ecg_signal: np.ndarray, sfreq: float = 160.0) -> Dict[str, Any]:
    """
    Extract Heart Rate Variability (HRV) metrics from actual ECG time series.
    Calculates Heart Rate (BPM), SDNN (ms), and RMSSD (ms).
    """
    from scipy.signal import find_peaks

    # Normalize ECG signal
    std_val = np.std(ecg_signal)
    if std_val < 1e-8:
        return {"heart_rate_bpm": np.nan, "sdnn_ms": np.nan, "rmssd_ms": np.nan, "hrv_valid": False}

    ecg_norm = (ecg_signal - np.mean(ecg_signal)) / std_val
    
    # Peak detection for R-peaks (min 350ms between beats = max 170 BPM)
    min_dist = int(0.35 * sfreq)
    peaks, _ = find_peaks(ecg_norm, height=1.0, distance=min_dist)

    if len(peaks) < 4:
        return {"heart_rate_bpm": np.nan, "sdnn_ms": np.nan, "rmssd_ms": np.nan, "hrv_valid": False}

    # RR intervals in milliseconds
    rr_intervals_ms = (np.diff(peaks) / sfreq) * 1000.0

    mean_rr = np.mean(rr_intervals_ms)
    heart_rate_bpm = 60000.0 / mean_rr if mean_rr > 0 else np.nan
    sdnn = np.std(rr_intervals_ms, ddof=1) if len(rr_intervals_ms) > 1 else np.nan
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals_ms)))) if len(rr_intervals_ms) > 1 else np.nan

    return {
        "heart_rate_bpm": round(float(heart_rate_bpm), 2) if not np.isnan(heart_rate_bpm) else np.nan,
        "sdnn_ms": round(float(sdnn), 2) if not np.isnan(sdnn) else np.nan,
        "rmssd_ms": round(float(rmssd), 2) if not np.isnan(rmssd) else np.nan,
        "hrv_valid": True,
    }


def compute_epoch_psd_features(
    epochs_dict: Dict[str, Tuple[np.ndarray, List[str]]],
    sfreq: float = 160.0,
    estimate_missing_hrv: bool = False,
    subject_id: int = 1,
    data_source: str = "unknown"
) -> pd.DataFrame:
    """
    Extract tabular features across all epochs and experimental conditions from real signal arrays.

    Parameters:
    -----------
    epochs_dict : Dict[str, Tuple[np.ndarray, List[str]]]
        Dictionary mapping condition ('resting', 'task') to (epochs_array, channel_names).
    sfreq : float
        Sampling frequency in Hz.
    estimate_missing_hrv : bool
        If True, estimates surrogate HRV when ECG lead is missing, explicitly flagging `is_hrv_simulated=True`.
        If False (default), missing ECG sets HRV fields to np.nan with `is_hrv_simulated=False`.
    subject_id : int
        ID of the subject being processed.
    data_source : str
        Provenance label ('physionet_real', 'synthetic_demonstration', etc.).

    Returns:
    --------
    pd.DataFrame
        Structured tabular feature dataset with exact PSD powers, TBR, and HRV metadata.
    """
    logger.info("Extracting spectral band powers and medical ratios across conditions...")
    feature_rows = []

    for condition_key, (epochs, ch_names) in epochs_dict.items():
        n_epochs, n_channels, n_samples = epochs.shape
        condition_label = "Resting" if "rest" in condition_key.lower() else "Task Execution"

        # Identify EEG vs ECG channels
        eeg_indices = [i for i, name in enumerate(ch_names) if "ecg" not in name.lower()]
        ecg_indices = [i for i, name in enumerate(ch_names) if "ecg" in name.lower()]
        has_ecg = len(ecg_indices) > 0

        for epoch_idx in range(n_epochs):
            row_dict = {
                "subject_id": subject_id,
                "epoch_id": epoch_idx + 1,
                "condition": condition_label,
                "data_source": data_source,
                "has_ecg_lead": has_ecg,
            }

            # Compute EEG band powers averaged over EEG channels
            channel_band_powers = {band: [] for band in FREQ_BANDS.keys()}

            for ch_idx in eeg_indices:
                ch_signal = epochs[epoch_idx, ch_idx, :]
                freqs, psd = compute_welch_psd(ch_signal, sfreq=sfreq)

                for band_name, band_range in FREQ_BANDS.items():
                    power = extract_band_power(freqs, psd, band_range)
                    # Convert to uV^2 (if input was in Volts < 1e-4)
                    power_uV2 = power * 1e12 if np.max(psd) < 1e-4 else power
                    channel_band_powers[band_name].append(power_uV2)

            # Average across EEG channels
            mean_delta = np.mean(channel_band_powers["Delta"]) if channel_band_powers["Delta"] else np.nan
            mean_theta = np.mean(channel_band_powers["Theta"]) if channel_band_powers["Theta"] else np.nan
            mean_alpha = np.mean(channel_band_powers["Alpha"]) if channel_band_powers["Alpha"] else np.nan
            mean_beta = np.mean(channel_band_powers["Beta"]) if channel_band_powers["Beta"] else np.nan

            total_power = (mean_delta if not np.isnan(mean_delta) else 0) + \
                          (mean_theta if not np.isnan(mean_theta) else 0) + \
                          (mean_alpha if not np.isnan(mean_alpha) else 0) + \
                          (mean_beta if not np.isnan(mean_beta) else 0) + 1e-8

            tbr = float(mean_theta / (mean_beta + 1e-8)) if (not np.isnan(mean_theta) and not np.isnan(mean_beta)) else np.nan
            abr = float(mean_alpha / (mean_beta + 1e-8)) if (not np.isnan(mean_alpha) and not np.isnan(mean_beta)) else np.nan

            row_dict.update({
                "delta_power": round(float(mean_delta), 4) if not np.isnan(mean_delta) else np.nan,
                "theta_power": round(float(mean_theta), 4) if not np.isnan(mean_theta) else np.nan,
                "alpha_power": round(float(mean_alpha), 4) if not np.isnan(mean_alpha) else np.nan,
                "beta_power": round(float(mean_beta), 4) if not np.isnan(mean_beta) else np.nan,
                "relative_theta": round(float(mean_theta / total_power), 4) if not np.isnan(mean_theta) else np.nan,
                "relative_beta": round(float(mean_beta / total_power), 4) if not np.isnan(mean_beta) else np.nan,
                # Primary Clinical Ratios
                "theta_beta_ratio": round(tbr, 4) if not np.isnan(tbr) else np.nan,
                "alpha_beta_ratio": round(abr, 4) if not np.isnan(abr) else np.nan,
            })

            # HRV Extraction with strict transparency
            if has_ecg:
                ecg_sig = epochs[epoch_idx, ecg_indices[0], :]
                hrv_res = calculate_hrv_metrics(ecg_sig, sfreq=sfreq)
                row_dict.update({
                    "heart_rate_bpm": hrv_res["heart_rate_bpm"],
                    "sdnn_ms": hrv_res["sdnn_ms"],
                    "rmssd_ms": hrv_res["rmssd_ms"],
                    "is_hrv_simulated": False,
                })
            else:
                if estimate_missing_hrv:
                    # Explicitly flagged surrogate
                    is_task = condition_label == "Task Execution"
                    row_dict.update({
                        "heart_rate_bpm": round(82.0 + np.random.normal(0, 3) if is_task else 68.0 + np.random.normal(0, 2), 2),
                        "sdnn_ms": round(38.0 + np.random.normal(0, 4) if is_task else 52.0 + np.random.normal(0, 5), 2),
                        "rmssd_ms": round(26.0 + np.random.normal(0, 3) if is_task else 41.0 + np.random.normal(0, 4), 2),
                        "is_hrv_simulated": True,
                    })
                else:
                    # Honest unsimulated representation
                    row_dict.update({
                        "heart_rate_bpm": np.nan,
                        "sdnn_ms": np.nan,
                        "rmssd_ms": np.nan,
                        "is_hrv_simulated": False,
                    })

            feature_rows.append(row_dict)

    df_features = pd.DataFrame(feature_rows)
    logger.info(f"Feature dataframe generated successfully with shape {df_features.shape}.")
    return df_features


def compute_condition_average_psd(
    epochs_dict: Dict[str, Tuple[np.ndarray, List[str]]],
    sfreq: float = 160.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes true averaged Power Spectral Density (PSD) curves across all EEG channels and epochs
    for both Resting and Task Execution conditions using Welch's method.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - freqs: 1D array of frequency bins (Hz)
        - psd_resting_mean: 1D array of true mean PSD values for Resting condition (uV^2/Hz)
        - psd_task_mean: 1D array of true mean PSD values for Task condition (uV^2/Hz)
    """
    condition_psds = {}

    for condition_key, (epochs, ch_names) in epochs_dict.items():
        n_epochs, n_channels, n_samples = epochs.shape
        eeg_indices = [i for i, name in enumerate(ch_names) if "ecg" not in name.lower()]

        all_psds = []
        freq_bins = None

        for ep_idx in range(n_epochs):
            for ch_idx in eeg_indices:
                ch_signal = epochs[ep_idx, ch_idx, :]
                freqs, psd = compute_welch_psd(ch_signal, sfreq=sfreq)
                freq_bins = freqs
                
                # Scale to uV^2/Hz if in Volts
                psd_uV2 = psd * 1e12 if np.max(psd) < 1e-4 else psd
                all_psds.append(psd_uV2)

        mean_condition_psd = np.mean(all_psds, axis=0) if all_psds else np.zeros_like(freq_bins)
        label = "resting" if "rest" in condition_key.lower() else "task"
        condition_psds[label] = (freq_bins, mean_condition_psd)

    freqs_rest, psd_rest = condition_psds.get("resting", (np.linspace(0, 40, 100), np.zeros(100)))
    _, psd_task = condition_psds.get("task", (freqs_rest, np.zeros(len(freqs_rest))))

    return freqs_rest, psd_rest, psd_task


if __name__ == "__main__":
    from data_loader import load_subject_data
    from preprocessing import bandpass_filter_raw, create_epochs

    data = load_subject_data(1)
    epochs_dict = {}

    for state, raw in data.items():
        if state in ("resting", "task"):
            if hasattr(raw, "get_data"):
                filtered = bandpass_filter_raw(raw)
                epochs, ch_names = create_epochs(filtered, duration_sec=30.0)
                epochs_dict[state] = (epochs, ch_names)

    if epochs_dict:
        df = compute_epoch_psd_features(epochs_dict, sfreq=160.0, data_source=data.get("data_source", "unknown"))
        print("Feature extraction test complete. Head:\n", df.head())
