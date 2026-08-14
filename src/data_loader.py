"""
Data Loader Module for PhysioNet EEG/ECG Signals.

Fetches motor movement/execution EEG datasets from PhysioNet via MNE-Python,
or provides transparent synthetic biosignal benchmarks when offline/in demonstration mode.
Includes explicit data provenance tracking and multi-subject cohort loading.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import mne
    from mne.datasets import eegbci
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    logger.warning("MNE package not installed. Fallback synthetic mode enabled.")


def fetch_physionet_eeg_data(
    subject: int = 1,
    runs: Tuple[int, int] = (1, 6),
    data_path: Optional[str] = "./data",
    force_synthetic: bool = False
) -> Dict[str, Any]:
    """
    Fetch EEG data from the PhysioNet EEG Motor Movement/Execution Dataset (EEGMMIDB).

    Parameters:
    -----------
    subject : int
        Subject ID (1-109). Default is 1.
    runs : Tuple[int, int]
        Run numbers to load. Default is (1, 6):
        - Run 1: Baseline resting state (Eyes Open)
        - Run 6: Motor execution / mental task state (hands/feet execution)
    data_path : str, optional
        Local directory path to cache EDF files.
    force_synthetic : bool, optional
        If True, skips network call and generates benchmark data with explicit provenance flag.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'resting': MNE Raw or ndarray object
        - 'task': MNE Raw or ndarray object
        - 'data_source': 'physionet_real' | 'synthetic_demonstration'
        - 'is_synthetic': bool
        - 'subject_id': int
        - 'sfreq': float
    """
    os.makedirs(data_path, exist_ok=True)
    logger.info(f"Loading EEG data for Subject {subject}, Runs {runs}...")

    raw_dict = {}
    run_labels = {runs[0]: "resting", runs[1]: "task"}

    if HAS_MNE and not force_synthetic:
        try:
            # PhysioNet EEG BCI Dataset download via MNE
            edf_paths = eegbci.load_data(subject, list(runs), path=data_path, update_path=True, verbose=False)
            
            for path, run_num in zip(edf_paths, runs):
                label = run_labels[run_num]
                # Preload raw EDF data
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                # Standardize channel names (e.g., 'Fc1.' -> 'FC1')
                eegbci.standardize_names(raw)
                # Set 10-20 montage
                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage, on_missing="ignore", verbose=False)
                
                raw_dict[label] = raw
                logger.info(
                    f"Successfully loaded real PhysioNet record for Subject {subject} '{label}' (Run {run_num}): "
                    f"{len(raw.ch_names)} channels, {raw.n_times} samples @ {raw.info['sfreq']} Hz."
                )
            
            raw_dict["data_source"] = "physionet_real"
            raw_dict["is_synthetic"] = False
            raw_dict["subject_id"] = subject
            raw_dict["sfreq"] = float(raw_dict["resting"].info["sfreq"])
            return raw_dict
        except Exception as e:
            logger.warning(
                f"PhysioNet online fetch unavailable or failed ({e}). "
                "Switching to illustrative synthetic benchmark dataset."
            )

    # Fallback synthetic generator
    logger.warning(
        "\n"
        "================================================================================\n"
        "[PROVENANCE WARNING] Pipeline is running on illustrative synthetic benchmark data.\n"
        "These results are for demonstration/pipeline validation only, and NOT real PhysioNet records.\n"
        "================================================================================\n"
    )
    synth_data = generate_synthetic_biosignals(sfreq=160.0, duration_sec=120.0, seed=42 + subject)
    synth_data["subject_id"] = subject
    return synth_data


def generate_synthetic_biosignals(
    sfreq: float = 160.0,
    duration_sec: float = 120.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates multi-channel EEG + ECG biosignals for resting vs mental task states
    with stochastic variation, 1/f pink noise, and physiological variance.

    Explicitly returns metadata flagging that data is synthetic.
    """
    n_samples = int(sfreq * duration_sec)
    time = np.linspace(0, duration_sec, n_samples)
    
    ch_names = ["Fz", "Cz", "Pz", "O1", "O2", "F3", "F4", "C3", "C4", "P3", "P4", "ECG"]
    ch_types = ["eeg"] * 11 + ["ecg"]
    
    np.random.seed(seed)
    
    def synthesize_state(is_task: bool) -> np.ndarray:
        signals = []
        for i, ch in enumerate(ch_names[:-1]):
            # Base 1/f pink noise approximation
            white = np.random.normal(0, 1.0, n_samples)
            pink = np.cumsum(white)
            pink = pink - np.mean(pink)
            pink = pink / (np.std(pink) + 1e-8) * 1.5
            
            # Slow cortical baseline oscillation (Delta: 0.5-4 Hz)
            delta_freq = 2.0 + np.random.uniform(-0.5, 0.5)
            delta = 2.2 * np.sin(2 * np.pi * delta_freq * time + np.random.uniform(0, 2*np.pi))
            
            if not is_task:
                # Baseline Resting State: Natural Alpha dominance with individual variability
                alpha_freq = 10.2 + np.random.uniform(-0.8, 0.8)
                theta_freq = 5.8 + np.random.uniform(-0.6, 0.6)
                beta_freq = 19.5 + np.random.uniform(-1.5, 1.5)
                
                theta_amp = 3.2 + np.random.uniform(-0.6, 0.6)
                alpha_amp = 5.8 + np.random.uniform(-1.0, 1.0)
                beta_amp = 1.6 + np.random.uniform(-0.4, 0.4)
            else:
                # Cognitive Task State: Task-induced Alpha desynchronization and Beta power surge
                alpha_freq = 10.0 + np.random.uniform(-0.8, 0.8)
                theta_freq = 6.4 + np.random.uniform(-0.6, 0.6)
                beta_freq = 22.0 + np.random.uniform(-1.5, 1.5)
                
                theta_amp = 2.4 + np.random.uniform(-0.5, 0.5)
                alpha_amp = 2.1 + np.random.uniform(-0.5, 0.5)
                beta_amp = 5.2 + np.random.uniform(-0.8, 0.8)

            theta = theta_amp * np.sin(2 * np.pi * theta_freq * time + np.random.uniform(0, 2*np.pi))
            alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * time + np.random.uniform(0, 2*np.pi))
            beta = beta_amp * np.sin(2 * np.pi * beta_freq * time + np.random.uniform(0, 2*np.pi))
            
            # Combine components with realistic microvolt scaling
            ch_signal = (delta + theta + alpha + beta + pink) * 1e-6  # Volts
            signals.append(ch_signal)
            
        # Synthetic ECG signal generation (R-peak intervals)
        bpm = 82.0 + np.random.uniform(-4, 4) if is_task else 68.0 + np.random.uniform(-3, 3)
        beat_interval = 60.0 / bpm
        ecg_signal = np.zeros(n_samples)
        for t_beat in np.arange(0.5, duration_sec - 0.2, beat_interval):
            # Add subtle HRV jitter
            jitter = np.random.uniform(-0.02, 0.02)
            idx = int((t_beat + jitter) * sfreq)
            if 0 <= idx < n_samples:
                # QRS wave spike
                qrs_profile = np.array([-0.15, 1.2, -0.35, 0.1, -0.05])
                qrs_len = len(qrs_profile)
                start_i = max(0, idx - 2)
                end_i = min(n_samples, start_i + qrs_len)
                ecg_signal[start_i:end_i] += qrs_profile[:end_i - start_i]
                
        ecg_signal += np.random.normal(0, 0.04, n_samples)
        ecg_signal *= 1e-3  # Volts
        signals.append(ecg_signal)
        
        return np.array(signals)

    data_resting = synthesize_state(is_task=False)
    data_task = synthesize_state(is_task=True)

    if HAS_MNE:
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw_resting = mne.io.RawArray(data_resting, info, verbose=False)
        raw_task = mne.io.RawArray(data_task, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_resting.set_montage(montage, on_missing="ignore", verbose=False)
        raw_task.set_montage(montage, on_missing="ignore", verbose=False)
        return {
            "resting": raw_resting,
            "task": raw_task,
            "data_source": "synthetic_demonstration",
            "is_synthetic": True,
            "sfreq": sfreq
        }
    else:
        return {
            "resting": data_resting,
            "task": data_task,
            "ch_names": ch_names,
            "sfreq": sfreq,
            "data_source": "synthetic_demonstration",
            "is_synthetic": True
        }


def load_subject_data(subject_id: int = 1) -> Dict[str, Any]:
    """
    Convenience function to load resting and cognitive workload data for a subject.
    """
    return fetch_physionet_eeg_data(subject=subject_id, runs=(1, 6))


def load_cohort_data(subject_ids: List[int] = [1, 2, 3, 4, 5]) -> List[Dict[str, Any]]:
    """
    Load data across a multi-subject cohort (e.g. n=5 or n=10 subjects)
    to enable group-level statistical testing across individuals.
    """
    logger.info(f"Loading cohort dataset for subjects: {subject_ids}")
    cohort = []
    for sid in subject_ids:
        subject_dict = fetch_physionet_eeg_data(subject=sid, runs=(1, 6))
        cohort.append(subject_dict)
    return cohort


if __name__ == "__main__":
    data = load_subject_data(1)
    print(f"Data loader test complete. Source: {data.get('data_source')}, Is Synthetic: {data.get('is_synthetic')}")
