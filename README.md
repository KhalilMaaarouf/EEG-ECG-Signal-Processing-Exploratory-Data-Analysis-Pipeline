# NeuroHealth-Biosignals-EDA

[!\[Bio-AI Pipeline](https://img.shields.io/badge/Bio--AI-Signal%20Processing-blue?style=for-the-badge\&logo=python)](https://github.com)
[!\[MNE-Python](https://img.shields.io/badge/MNE--Python-1.5%2B-green?style=for-the-badge)](https://mne.tools/)
[!\[Data: Synthetic Demo](https://img.shields.io/badge/Data-Synthetic%20Demo-yellow?style=for-the-badge)](#-data-provenance--scientific-methodology)
[!\[PhysioNet-Compatible](https://img.shields.io/badge/PhysioNet--Compatible-EEGMMIDB-orange?style=for-the-badge)](https://physionet.org/content/eegmmidb/1.0.0/)
[!\[License: MIT](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](https://opensource.org/licenses/MIT)

> \*\*Biomedical Signal Processing \& Mental Workload / Cognitive Fatigue Exploratory Data Analysis Pipeline\*\*

\---





> ⚠️ \\\*\\\*Results shown below were generated in Synthetic Demonstration Mode\\\*\\\*
> (`is\\\_synthetic=True`, n=5 synthetic subjects, 70 epochs). The code automatically fetches
> real PhysioNet EEGMMIDB data when network access allows, but no network access was
> available in the environment that produced these specific numbers and plots. Treat all
> statistics and figures below as an illustrative pipeline validation, not as a finding
> from real human EEG/ECG data. See "Data Provenance \\\& Scientific Methodology" above for
> details.







## 📌 Project Overview

**NeuroHealth-Biosignals-EDA** is an open-source exploratory data analysis (EDA) and feature engineering framework designed for multi-channel electroencephalogram (EEG) and electrocardiogram (ECG) signals.

Cognitive fatigue and elevated mental workload are key areas of study in cognitive neuroscience, ergonomics, and human-computer interaction. By leveraging Power Spectral Density (PSD) decomposition via Welch's method and extracting standard neuro-markers (such as the **Theta / Beta Ratio** and **Heart Rate Variability**), this pipeline automates the transformation of continuous raw voltage time series into structured, interpretable tabular datasets for biomedical exploratory data analysis.

\---

## 🔍 Data Provenance \& Scientific Methodology

To ensure reproducibility and scientific transparency:

* **Real Data Mode:** When network access is available, the pipeline automatically streams and caches real 64-channel EDF recordings from the **PhysioNet EEG Motor Movement/Execution Dataset (EEGMMIDB)**.
* **Demonstration / Benchmark Mode:** When executed in an offline sandbox without internet access or MNE binaries, the pipeline transparently runs on a synthetic biosignal generator with explicit provenance logging (`is\_synthetic=True`).
* **No Masked Results:** Missing ECG leads are left as unsimulated (`NaN` / `is\_hrv\_simulated=False`) by default, and all statistical hypothesis testing reports exact two-sided $p$-values without hardcoded significance overrides.

\---

## 📊 Pipeline Validation — Synthetic Demonstration

### 1\. Synthetic EEG Spectral Demonstration



The synthetic benchmark reproduces an expected spectral pattern in which alpha-band power decreases while beta-band power increases during the simulated task condition.



Baseline Resting State: The synthetic signal exhibits prominent Alpha power (\~10 Hz).



Simulated Task Condition: The synthetic signal exhibits reduced Alpha power and increased Beta power (\~22 Hz).



These patterns are generated for pipeline validation and demonstrate the ability of the feature-extraction and visualization modules to detect controlled spectral differences. They should not be interpreted as empirical physiological findings.

!\[EEG PSD Comparison](assets/eeg\_psd\_comparison.png)

\---

### 2\. Synthetic TBR Feature Demonstration



The **Theta / Beta Ratio (TBR)** is a standard biomarker in neuro-ergonomics and attentional research (e.g., Barry et al., 2003):

* **Resting Baseline Mean TBR:** $3.751 \\pm 0.152$ (higher relative slow-wave power during relaxed baseline)
* **Task Execution Mean TBR:** $0.211 \\pm 0.017$ (significant reduction driven by elevated Beta power)
* **Statistical Significance:** Independent sample $t$-test yields $t = 136.857$, $p = 3.1576 \\times 10^{-49}$ ($p < 0.001$, Cohen's $d = 32.72$).

!\[Theta Beta Ratio Workload](assets/theta\_beta\_workload.png)

\---

## 🔑 Quantitative Results Table

The metrics below are computed from $n=70$ epochs across 5 subjects evaluated in the accompanying notebook (`01\_EEG\_ECG\_Workload\_Analysis.ipynb`):

|Biomarker / Feature|Resting State Baseline|Task Execution State|Shift / % Change|Statistical Metric|
|-|-|-|-|-|
|**Theta/Beta Ratio (TBR)**|$3.751 \\pm 0.152$|$0.211 \\pm 0.017$|$-94.4%$ reduction|$t = 136.857, p = 3.1576 \\times 10^{-49}$|
|**Beta Power (13–30 Hz)**|$1.330 \\pm 0.094\\ \\mu\\text{V}^2$|$14.059 \\pm 0.838\\ \\mu\\text{V}^2$|$+956.9%$ surge|$t = -89.314, p = 9.3707 \\times 10^{-43}$|
|**Alpha Power (8–12 Hz)**|$16.722 \\pm 0.695\\ \\mu\\text{V}^2$|$2.496 \\pm 0.104\\ \\mu\\text{V}^2$|$-85.1%$ suppression|$t = 119.796, p = 6.6499 \\times 10^{-48}$|
|**Theta Power (4–8 Hz)**|$4.985 \\pm 0.335\\ \\mu\\text{V}^2$|$2.951 \\pm 0.151\\ \\mu\\text{V}^2$|$-40.8%$ reduction|$t = 32.741, p = 3.9812 \\times 10^{-34}$|
|**Heart Rate (ECG)**|$71.6 \\pm 2.8\\ \\text{BPM}$|$81.9 \\pm 2.1\\ \\text{BPM}$|$+10.3\\ \\text{BPM}$ increase|$t = -17.506, p = 5.7076 \\times 10^{-26}$|
|**RMSSD (HRV)**|$108.8 \\pm 55.8\\ \\text{ms}$|$41.5 \\pm 26.6\\ \\text{ms}$|$-61.9%$ reduction|$t = 6.441, p = 5.0211 \\times 10^{-08}$|

\---

## 📂 Repository Architecture

```
NeuroHealth-Biosignals-EDA/
├── assets/                                 # High-resolution output plot images (PNGs)
│   ├── eeg\_psd\_comparison.png              # Welch PSD spectrum overlay (Resting vs Task)
│   └── theta\_beta\_workload.png             # Boxplot of Theta/Beta Ratio with p-value annotation
├── data/                                   # Downloaded/cached PhysioNet EDF recordings
├── src/                                    # Modular Python source modules
│   ├── \_\_init\_\_.py                         # Package initialization
│   ├── data\_loader.py                      # PhysioNet data loader \& synthetic benchmark generator
│   ├── preprocessing.py                   # Zero-phase bandpass filtering \& epoch segmentation
│   └── feature\_extraction.py              # Welch PSD, band power integration, TBR, \& HRV
├── 01\_EEG\_ECG\_Workload\_Analysis.ipynb      # Reproducible Jupyter notebook with all cell outputs
├── generate\_assets.py                      # Batch asset generation script using real Welch PSD
├── requirements.txt                        # Strict dependency specifications
├── .gitignore                             # Python build/data ignore rules
└── README.md                               # Complete project documentation
```

\---

## 🚀 Quickstart Guide

### 1\. Clone the Repository \& Navigate

```bash
[git clone https://github.com/your-username/NeuroHealth-Biosignals-EDA.git
cd NeuroHealth-Biosignals-EDA](https://github.com/KhalilMaaarouf/EEG-ECG-Signal-Processing-Exploratory-Data-Analysis-Pipeline.git)
```

### 2\. Set Up a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\\Scripts\\activate
```

### 3\. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4\. Run the Pipeline \& Jupyter Notebook

```bash
# Launch interactive analysis notebook
jupyter notebook 01\_EEG\_ECG\_Workload\_Analysis.ipynb
```

Or execute modular python scripts directly:

```bash
python3 src/data\_loader.py
python3 src/preprocessing.py
python3 src/feature\_extraction.py
python3 generate\_assets.py
```

\---

## 🔬 Neuro-Electrophysiological Frequency Bands

|Brain Wave Band|Frequency Range (Hz)|Physiological State \& Clinical Significance|
|-|-|-|
|**Delta** ($\\delta$)|0.5 – 4.0 Hz|Deep non-REM sleep, slow-wave cortical synchronization. Elevated in waking states during deep fatigue.|
|**Theta** ($\\theta$)|4.0 – 8.0 Hz|Drowsiness, working memory allocation, and cognitive fatigue.|
|**Alpha** ($\\alpha$)|8.0 – 12.0 Hz|Relaxed alert state, posterior sensory idling. Suppressed during visual engagement and mental load.|
|**Beta** ($\\beta$)|13.0 – 30.0 Hz|Active cognitive focus, analytical problem solving, and mental effort.|
|**TBR ($\\theta/\\beta$)**|Ratio|Standard neuro-marker of cognitive workload. Lower ratio indicates increased mental exertion.|

\---

## 📖 References \& Acknowledgments

1. **PhysioNet EEGMMIDB Dataset:** Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation* 101(23):e215-e220, 2000.
2. **MNE-Python:** Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7:267, 2013.
3. **EEG Theta/Beta Ratio in Cognitive Research:** Barry RJ, Clarke AR, Johnstone SJ. A review of electrophysiology in attention-deficit/hyperactivity disorder: I. Qualitative and quantitative EEG. *Clin Neurophysiol*, 114(2):171-183, 2003.

\---

*Developed for Biomedical Signal Processing, Neuro-Ergonomics, and Medical Data Science.*

