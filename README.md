# 🚗 Ride Predictor

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

> Predict which driver is most likely to accept a given ride order — enabling smarter, faster, and more efficient ride allocations.

---

## 📘 Overview

**Ride Predictor** is a machine learning system that predicts the probability a particular driver will accept a given incoming order.  
Unlike standard trip-completion models, this project focuses on **driver–order matching**: given an order and a set of candidate drivers, rank or score drivers by acceptance likelihood. The pipeline is modular and production-aware.

Key design choice: a **two-stage pipeline**
1. **Candidate selection (Stage 1):** quickly filter drivers who are plausibly available for an order (proximity, online status, shift, etc.).
2. **Acceptance scoring (Stage 2):** for each candidate driver-order pair, predict acceptance using a binary classifier.

---

## 🧩 Key Features

- **Two-stage ranking architecture** — efficient candidate filtering followed by accurate ranking.
- **Rich feature engineering**: geographical, temporal, behavioral, clustering, and derived features (e.g., `log_acceptance_time`).
- **Multiple model options**: Logistic Regression (baseline), Random Forest, XGBoost (recommended).
- **Hyperparameter tuning** with cross-validation.
- **Reproducible scripts** for ingestion, training, evaluation, and scoring.
- **Sphinx documentation** following NumPy docstring style.
- **Unit and integration tests** using `pytest`.
- Ready for extension to **real-time scoring (FastAPI/Flask)**.



