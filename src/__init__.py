"""
NeuroHealth-Biosignals-EDA package initialization.
Provides modular tools for EEG/ECG biosignal loading, preprocessing, filtering,
and spectral feature extraction for mental workload and cognitive fatigue evaluation.
"""

from .data_loader import fetch_physionet_eeg_data, load_subject_data
from .preprocessing import bandpass_filter_raw, apply_notch_filter, create_epochs
from .feature_extraction import compute_epoch_psd_features, calculate_hrv_metrics

__version__ = "1.0.0"
__all__ = [
    "fetch_physionet_eeg_data",
    "load_subject_data",
    "bandpass_filter_raw",
    "apply_notch_filter",
    "create_epochs",
    "compute_epoch_psd_features",
    "calculate_hrv_metrics",
]
