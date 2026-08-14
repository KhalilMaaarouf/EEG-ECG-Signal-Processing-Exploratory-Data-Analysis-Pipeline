"""
Preprocessing & Artifact Removal Module for Biomedical Signals.

Applies Zero-Phase Bandpass Filtering (1.0 - 40.0 Hz), Powerline Notch Filtering (50/60 Hz),
and Epoch Segmentation (30-second sliding windows) to continuous EEG/ECG signals.
"""

import logging
from typing import List, Tuple, Dict, Union, Optional, Any
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


def bandpass_filter_raw(
    raw: Any,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    sfreq: float = 160.0
) -> Any:
    """
    Apply zero-phase FIR/IIR Bandpass Filter (1.0 - 40.0 Hz) to attenuate DC drift and high-freq noise.

    Parameters:
    -----------
    raw : mne.io.Raw or np.ndarray
        Continuous signal object or multi-channel data matrix.
    l_freq : float
        Lower cutoff frequency (Hz). Default is 1.0 Hz.
    h_freq : float
        Upper cutoff frequency (Hz). Default is 40.0 Hz.
    sfreq : float
        Sampling rate in Hz (used if raw is np.ndarray).

    Returns:
    --------
    Filtered raw object or np.ndarray data matrix.
    """
    logger.info(f"Applying zero-phase bandpass filter: {l_freq} Hz - {h_freq} Hz...")

    if HAS_MNE and isinstance(raw, mne.io.BaseRaw):
        raw_filtered = raw.copy()
        # Filter EEG channels using FIR filter with zero-phase
        eeg_picks = mne.pick_types(raw_filtered.info, eeg=True, ecg=False, eog=False, meg=False)
        if len(eeg_picks) > 0:
            raw_filtered.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks=eeg_picks,
                filter_length="auto",
                l_trans_bandwidth="auto",
                h_trans_bandwidth="auto",
                method="fir",
                phase="zero",
                verbose=False,
            )
        return raw_filtered

    # Numpy Array Fallback using Scipy Butterworth Bandpass Filter
    nyquist = 0.5 * sfreq
    low = max(0.001, l_freq / nyquist)
    high = min(0.99, h_freq / nyquist)
    b, a = butter(N=4, Wn=[low, high], btype="bandpass")

    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"].copy()
    else:
        data = np.copy(raw)

    if data.ndim == 1:
        data = filtfilt(b, a, data)
    elif data.ndim == 2:
        for ch_idx in range(data.shape[0]):
            data[ch_idx, :] = filtfilt(b, a, data[ch_idx, :])

    return data


def apply_notch_filter(
    raw: Any,
    freqs: Union[float, List[float]] = 50.0,
    sfreq: float = 160.0
) -> Any:
    """
    Apply Notch filter to suppress 50Hz / 60Hz powerline interference.
    """
    if isinstance(freqs, (int, float)):
        freqs = [float(freqs)]

    logger.info(f"Applying powerline notch filter at {freqs} Hz...")

    if HAS_MNE and isinstance(raw, mne.io.BaseRaw):
        raw_notch = raw.copy()
        eeg_picks = mne.pick_types(raw_notch.info, eeg=True)
        if len(eeg_picks) > 0:
            raw_notch.notch_filter(freqs=freqs, picks=eeg_picks, verbose=False)
        return raw_notch

    data = np.copy(raw)
    nyquist = 0.5 * sfreq
    for f in freqs:
        w0 = f / nyquist
        if 0 < w0 < 1:
            b, a = iirnotch(w0=w0, Q=30.0)
            if data.ndim == 1:
                data = filtfilt(b, a, data)
            elif data.ndim == 2:
                for ch_idx in range(data.shape[0]):
                    data[ch_idx, :] = filtfilt(b, a, data[ch_idx, :])

    return data


def create_epochs(
    raw: Any,
    duration_sec: float = 30.0,
    overlap_sec: float = 15.0,
    sfreq: float = 160.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Segments continuous multi-channel signals into uniform sliding time windows (Epochs).

    Parameters:
    -----------
    raw : mne.io.Raw or np.ndarray
        Continuous signal input.
    duration_sec : float
        Epoch duration in seconds (Default 30s).
    overlap_sec : float
        Sliding window overlap in seconds (Default 15s).
    sfreq : float
        Sampling frequency in Hz.

    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        - epochs_array: 3D array of shape (n_epochs, n_channels, n_samples)
        - channel_names: List of channel names.
    """
    logger.info(f"Segmenting signals into {duration_sec}s epochs with {overlap_sec}s overlap...")

    if HAS_MNE and isinstance(raw, mne.io.BaseRaw):
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data()
        ch_names = list(raw.ch_names)
    elif isinstance(raw, dict):
        data = raw.get("data")
        ch_names = raw.get("ch_names", [f"Ch_{i}" for i in range(data.shape[0])])
        sfreq = float(raw.get("sfreq", sfreq))
    else:
        data = raw
        ch_names = [f"Ch_{i}" for i in range(data.shape[0])]

    n_channels, total_samples = data.shape
    samples_per_epoch = int(duration_sec * sfreq)
    step_samples = int((duration_sec - overlap_sec) * sfreq)

    if step_samples <= 0:
        step_samples = samples_per_epoch

    epochs_list = []
    start_sample = 0

    while start_sample + samples_per_epoch <= total_samples:
        epoch = data[:, start_sample : start_sample + samples_per_epoch]
        epochs_list.append(epoch)
        start_sample += step_samples

    if len(epochs_list) == 0:
        logger.warning("Signal duration shorter than 1 epoch length. Using full signal as single epoch.")
        epochs_list.append(data[:, :samples_per_epoch])

    epochs_array = np.array(epochs_list)  # Shape: (n_epochs, n_channels, n_samples)
    logger.info(f"Epoch segmentation complete. Generated {epochs_array.shape[0]} epochs of shape {epochs_array.shape[1:]}.")

    return epochs_array, ch_names


if __name__ == "__main__":
    from data_loader import load_subject_data
    raw_data = load_subject_data(1)
    resting_filtered = bandpass_filter_raw(raw_data["resting"], l_freq=1.0, h_freq=40.0)
    epochs, ch_names = create_epochs(resting_filtered, duration_sec=30.0, overlap_sec=15.0)
    print("Preprocessing test complete. Epochs shape:", epochs.shape)
