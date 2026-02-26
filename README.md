# Cost-Aware Confidence-Gated Two-Stage Network Intrusion Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper: **"Cost-Aware Confidence-Gated Two-Stage Network Intrusion Detection under Cross-Class Distribution Shift"**.

We propose a cost-aware defense-in-depth Network Intrusion Detection System (NIDS) that integrates the high-throughput screening of **LSTMs** (Long Short-Term Memory) with a selectively invoked **Lightweight Verifier** (Random Forest) to recover detection performance against unseen Zero-Day attacks, while strictly bounding computational cost.

---

## 🚀 Overview

Deep Learning models (like LSTMs) are highly sensitive to Distribution Shift. They often fail silently and confidently when test-time traffic deviates significantly from the training distribution (e.g., cross-class shift).

* **The Problem:** An LSTM trained exclusively on **DoS Attacks** exhibits complete generalization failure (Recall = 0.000) when evaluated against unseen **Web Attacks** under a strict cross-file protocol.
* **The Solution:** A confidence-gated "Fast & Slow" pipeline that exposes a direct Accuracy-Cost trade-off.
    * **⚡ Stage 1 (Fast-Path LSTM):** Handles the initial screening of all traffic. High-confidence predictions ($P_{attack} \le 0.05$ or $P_{attack} \ge 0.95$) are accepted immediately, efficiently filtering out ~45% of easily distinguishable traffic at near-zero latency.
    * **🛡️ Stage 2 (Slow-Path Verifier):** Flows falling into the ambiguity interval ($0.05 < P_{attack} < 0.95$) are escalated to a few-shot Random Forest verifier. Operating on a highly compact 3-tuple summary (Port, Duration, Avg Packet Size), this stage successfully recovers the F1 score to ~0.678.
    * **📝 Optional Audit Layer (LLM):** Unlike naive hybrid systems, we position Large Language Models strictly as an off-critical-path audit mechanism. It generates structured JSON rationales for escalated cases, acknowledging that LLM latency (~1519 ms) is completely impractical for inline network decision-making compared to our RF verifier (~3.00 ms, a **506x speedup**).

---

## 📊 Key Results

Our empirical evaluation on the **CIC-IDS-2017** dataset demonstrates the critical advantage of confidence-gated selective invocation.

### 1. Accuracy-Cost Trade-off (Gating Sweep)
By tuning the ambiguity interval, we can explicitly control the Verifier Invocation Rate (EscRate). At a 55% invocation rate, the system recovers nearly all the performance of an "always-on" verifier, demonstrating true cost-awareness.
*(Insert `gating_tradeoff_fix.png` here)*
`![Accuracy-Cost Trade-off](Images/gating_tradeoff_fix.png)`

### 2. Detection Coverage by Subclass
The system maintains a near-perfect True Negative Rate (TNR 99.60%) on benign traffic while successfully recovering detection capabilities for unseen attack variants like XSS and Brute Force.
*(Insert `recall_by_subclass_fix.png` here)*
`![Recall by Subclass](Images/recall_by_subclass_fix.png)`

### 3. Latency & Practical Deployability
The Stage 2 Lightweight Verifier (Random Forest) introduces minimal overhead, with an average inference time of **3.00 ms** and a highly stable P95 latency of **3.05 ms**. In contrast, an LLM audit path takes over 1.5 seconds per sample.
*(Insert `latency_comparison.png` here)*
`![Latency Comparison](Images/latency_comparison.png)`

---

