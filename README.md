Cost-Aware Confidence-Gated Two-Stage Network Intrusion Detection
This repository contains the official implementation of the paper: "Cost-Aware Confidence-Gated Two-Stage Network Intrusion Detection under Cross-Class Distribution Shift".

We propose a cost-aware defense-in-depth Network Intrusion Detection System (NIDS) that integrates the high-throughput screening of LSTMs (Long Short-Term Memory) with a selectively invoked Lightweight Verifier (Random Forest) to recover detection performance against unseen Zero-Day attacks, while strictly bounding computational cost.

🚀 Overview
Deep Learning models (like LSTMs) are highly sensitive to Distribution Shift. They often fail silently and confidently when test-time traffic deviates significantly from the training distribution (e.g., cross-class shift).

The Problem: An LSTM trained exclusively on DoS Attacks exhibits complete generalization failure (Recall = 0.000) when evaluated against unseen Web Attacks under a strict cross-file protocol.

The Solution: A confidence-gated "Fast & Slow" pipeline that exposes a direct Accuracy-Cost trade-off.

⚡ Stage 1 (Fast-Path LSTM): Handles the initial screening of all traffic. High-confidence predictions (P_attack <= 0.05 or P_attack >= 0.95) are accepted immediately, efficiently filtering out ~45% of easily distinguishable traffic at near-zero latency.

🛡️ Stage 2 (Slow-Path Verifier): Flows falling into the ambiguity interval (0.05 < P_attack < 0.95) are escalated to a few-shot Random Forest verifier. Operating on a highly compact 3-tuple summary (Port, Duration, Avg Packet Size), this stage successfully recovers the F1 score to ~0.678.

📝 Optional Audit Layer (LLM): Unlike naive hybrid systems, we position Large Language Models strictly as an off-critical-path audit mechanism. It generates structured JSON rationales for escalated cases, acknowledging that LLM latency (~1519 ms) is completely impractical for inline network decision-making compared to our RF verifier (~3.00 ms, a 506x speedup).

## 📂 Project Structure

```text
├── data/                   # Place CIC-IDS-2017 CSV files here
├── results/                # Generated plots, confusion matrices, and logs
│   ├── zeroday_training_curve.png
│   ├── Zero-Day_Confusion_Matrix.png
│   └── 2_Summary_Comparison.png
├── LSTM.py                 # Stage 1: Train the LSTM baseline (DoS -> Web Attack split)
├── model.py                # Stage 2: Hybrid Inference (LSTM + Confidence Gating + Ollama)
├── requirements.txt        # Python dependencies
└── README.md               # This file
