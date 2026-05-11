# PHM Model Architecture

## Overall Pipeline

TDMS vibration
+
Operation CSV
↓
Feature extraction
↓
Dual-branch temporal model
↓
RUL prediction

---

# Input Structure

## Vibration Input

Shape:
(batch, seq_len, 4, freq_bins)

Example:
(batch, 32, 4, 129)

Channels:
- CH1
- CH2
- CH3
- CH4

Representation:
- STFT magnitude spectrogram

---

## Operation Input

Shape:
(batch, seq_len, features)

Example:
(batch, 32, 6)

Features:
- torque
- speed
- temp_front
- temp_rear
- delta_t
- engineered features

---

# Vibration Branch

## Step 1 — STFT

Raw waveform
↓
Short-Time Fourier Transform
↓
Frequency-time representation

Parameters:
- nperseg = 256
- noverlap = 128

Output:
(freq_bins = 129)

---

## Step 2 — CNN Feature Extractor

Input:
(batch * seq_len, 4, 129)

Architecture:

Conv1d(4 → 32, kernel=5)
↓
BatchNorm
↓
ReLU
↓
Conv1d(32 → 64, kernel=3)
↓
BatchNorm
↓
ReLU
↓
Pooling
↓
AdaptiveAvgPool1d
↓
Feature vector

Purpose:
- learn local frequency patterns
- extract fault signatures
- identify resonance regions

---

## Step 3 — Temporal Modeling

CNN features
↓
LSTM(hidden=128, layers=2)

Purpose:
- learn degradation progression
- temporal evolution
- fault growth trend

Output:
h_vib

---

# Operation Branch

Input:
(batch, seq_len, operation_features)

Architecture:
LSTM(hidden=32)

Purpose:
- capture slow thermal trends
- torque evolution
- long-term degradation

Output:
h_op

---

# Fusion Branch

concat(h_vib, h_op)
↓
FC(160 → 128)
↓
ReLU
↓
Dropout(0.3)
↓
FC(128 → 64)
↓
ReLU
↓
FC(64 → 1)

Output:
Predicted RUL

---

# Why CNN Before LSTM?

Raw vibration contains:
- extremely high-frequency variation
- local fault patterns
- impulsive components

CNN:
- extracts compact local features
- reduces noise
- compresses spectral information

LSTM:
- models long-term temporal degradation

CNN + LSTM is more efficient than raw LSTM.

---

# Why Multi-Modal?

Vibration:
- sensitive to immediate faults

Temperature/Torque:
- sensitive to accumulated degradation

Combining both:
- improves robustness
- improves generalization
- improves RUL estimation

---

# Potential Upgrades

## Attention Fusion
Dynamic weighting between:
- vibration
- thermal
- torque information

---

## Transformer Encoder
For longer temporal dependency.

Risk:
- heavier GPU usage
- larger dataset requirement

---

## Physics-Informed Constraints
Examples:
- temperature rise consistency
- degradation monotonicity
- torque-growth relation

