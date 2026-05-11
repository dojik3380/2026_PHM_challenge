# KSPHM-KIMM 2026 PHM Challenge Context

## Project Goal
Predict Remaining Useful Life (RUL) of bearing systems until stop condition is reached.

The model must estimate:
- Remaining operating time
- Degradation progression
- Failure proximity

The challenge evaluation is based on:
- RUL prediction accuracy
- Asymmetric penalty score (A_RUL)

Underestimation and overestimation have different penalties.

---

# Dataset Overview

## Sensor Modalities

### Vibration Signals (TDMS)

Sampling Rate:
- 25.6 kHz

Channels:
- CH1: Front Vertical Vibration
- CH2: Front Axial Vibration
- CH3: Rear Vertical Vibration
- CH4: Rear Axial Vibration

Unit:
- g (acceleration)

Each TDMS file contains:
- 4-channel acceleration waveform
- Approximately 1 minute measurement

Measurement cycle:
- 1 minute measurement
- 9 minute rest

---

## Operation CSV Signals

Sampling interval:
- 10 seconds

Columns:
- Torque [Nm]
- Motor Speed [RPM]
- TC SP Front [°C]
- TC SP Rear [°C]

Derived features:
- abs_torque
- torque_delta
- torque_slope
- front_rear_temp_diff
- temp_slope
- delta_t

---

# Important Challenge Notes

## Stop Condition

Experiment stops when:
- Bearing degradation reaches critical condition
- Torque threshold exceeded
- Other failure conditions triggered

Important:
- Actual failure time may NOT exactly match final TDMS timestamp
- Failure can occur during non-measurement interval

---

## Validation/Test Data

Validation/Test sets contain:
- Only partial lifetime data

Actual RUL:
- Time from final provided sample
- Until real stop condition

Therefore:
- Final TDMS file is NOT guaranteed to be near failure

---

# Core Modeling Philosophy

This challenge is NOT simple anomaly detection.

The model must learn:
- Temporal degradation progression
- Multi-modal condition evolution
- Failure trajectory

---

# Signal Interpretation

## Vibration
Fast-changing fault information:
- impacts
- resonance
- fault frequency
- impulsive events

## Temperature / Torque
Slow-changing degradation indicators:
- friction increase
- lubrication degradation
- thermal accumulation

---

# Recommended Feature Engineering

## Vibration Domain
- STFT magnitude
- RMS
- Kurtosis
- Skewness
- Crest factor
- Envelope spectrum (future)

## Operation Domain
- Rolling mean
- Delta features
- Trend features
- Temporal gradients

---

# Current Model Architecture

Dual-branch multi-modal architecture:

1. Vibration branch
- STFT
- CNN
- LSTM

2. Operation branch
- LSTM

3. Fusion branch
- Fully connected regression head

---

# Loss Function

Custom A_RUL asymmetric penalty loss.

Purpose:
- Penalize dangerous prediction errors
- Reflect industrial maintenance risk

Typically:
- Underestimation and overestimation have different costs

---

# Future Upgrade Ideas

- Attention mechanism
- Transformer encoder
- Envelope analysis
- Physics-informed loss
- Sensor reliability weighting
- Noise robustness augmentation
- Multi-scale STFT