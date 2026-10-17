# Superstar-Driven Running Mode: A Multiple-Objective Framework in Professional Sports

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Preprint-orange.svg)]()

Official implementation and data repository for the paper:**Superstar-driven running mode: A multiple-objective framework in professional sports**

---

## 📖 Overview

Professional sports franchises operating under strict salary caps must simultaneously maximize **athletic competitiveness** and **financial sustainability**. This repository implements the **"Optimization, Pricing & Hedging"** framework proposed in the paper — a unified, three-stage quantitative pipeline that bridges operations research, revenue management, and financial engineering:

1. **Algorithm I — Multi-Objective Roster Optimization (MOMILP):** Selects an optimal 12-player roster and allocates playing time under a hard salary cap, using the ε-constraint method to trace the Pareto frontier between on-court production (Win Shares) and financial surplus value. Introduces a dual usage-rate mechanism (fatigue penalty + aggregate usage envelope) to prevent tactical cannibalization and produce the **"Barbell Strategy."**

2. **Algorithm II — Endogenous Demand & Adaptive Dynamic Pricing:** Uses **Tobit regression** to recover censored latent attendance demand and **Two-Stage Least Squares (2SLS)** on secondary-market ticket data to identify the true price elasticity. Combines both to build a counterfactual, superstar-aware dynamic pricing engine.

3. **Algorithm III — Monte Carlo Risk Analysis & CVaR Hedging:** Simulates 5,000 seasonal trajectories to translate discrete superstar injury events into bivariate (performance + revenue) financial shocks, and evaluates the impact of **Key Person Insurance** hedging using **Conditional Value-at-Risk (CVaR)**.

Empirically validated on the **Indiana Fever (WNBA)**, the framework:
- Identifies a Pareto-optimal roster achieving **65.90 Win Shares** under the $1.46M salary cap (Barbell Strategy).
- Generates a **+$3.05M (+7.12%)** revenue premium via dynamic pricing over static benchmarks.
- Improves **5% CVaR from $30.31M → $34.06M** through proactive injury hedging.

---

## 📁 Repository Structure

```
├── Home_ticekt/                     # Raw home-game ticketing data (secondary market, by game)
├── 2023_ticket.xlsx               # 2023 season secondary-market ticket listings (price, qty, lead time)
├── 2024_ticket.xlsx               # 2024 season secondary-market ticket listings (price, qty, lead time)
├── 2023_attendance.xlsx               # 2023 season game-level attendance records
├── 2024_attendance.xlsx               # 2024 season game-level attendance records
├── merged_result.xlsx           # Consolidated/cleaned dataset merging all seasons & panels
├── Final.py                     # Main script: implements Algorithms I, II, and III end-to-end
└── README.md                    # This file
```

> **Note on data provenance:** Player pool statistics are compiled from *Basketball-Reference* and *HerHoopStats*; salary data from *Spotrac*; attendance data from *Across the Timeline*; secondary-market ticket data from *SeatData.io*; average ticket price benchmark from *Statista (2024)*.

---

## ⚙️ Installation

### Requirements

- Python ≥ 3.11
- Recommended: virtual environment (`venv` or `conda`)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/superstar-driven-running-mode.git
cd superstar-driven-running-mode

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
pandas
scipy
statsmodels
pulp
matplotlib
seaborn
openpyxl
```

> **Notes on dependencies:**
> - `os`, `glob`, `time`, `warnings` are part of the Python standard library and require no installation.
> - `pulp` provides the Mixed Integer Linear Programming (MILP) solver used in **Algorithm I** (roster optimization). By default it ships with the open-source **CBC** solver; no external license is required.
> - `statsmodels` is used for the **Tobit** latent-demand regression and supporting econometric routines.
> - `scipy` (`scipy.stats.norm`, `scipy.optimize.minimize_scalar`, `scipy.optimize`) powers the Tobit likelihood computation and the bounded scalar optimization used in the **dynamic pricing engine (Algorithm II)**.
> - `openpyxl` is required by `pandas` to read/write the `.xlsx` data files provided in this repository (it is not imported directly in `Final.py`, but must be installed).

---

## 🚀 Usage

### 1. Prepare the data

Ensure the following files remain in the project root (or update file paths at the top of `Final.py` accordingly):

```
2023_ticket.xlsx
2024_ticket.xlsx
2023_attendance.xlsx
2024_attendance.xlsx
merged_result.xlsx
Home_ticket/
```

### 2. Run the full pipeline

```bash
python Final.py
```

This single script sequentially executes:

| Stage | Module | Output |
|---|---|---|
| **Phase 1** | MOMILP roster selection via `pulp` (ε-constraint, Pareto frontier) | Optimal 12-player roster, Pareto frontier plot |
| **Phase 2** | Game-day stochastic dispatch (form variance, usage constraint) | Minute allocation, DNP decisions, volatility plots |
| **Phase 3** | Tobit regression via `statsmodels` / `scipy.stats.norm` (latent demand) | Demand model coefficients, censoring diagnostics |
| **Phase 4** | 2SLS estimation (price elasticity) | First-stage F-stat, structural elasticity estimate |
| **Phase 5** | Dynamic pricing optimization via `scipy.optimize.minimize_scalar` | Game-by-game optimal prices, revenue premium |
| **Phase 6** | Monte Carlo injury simulation + CVaR | VaR/CVaR comparison (hedged vs. unhedged), sensitivity plots |

All generated figures are saved to `Paper_Figures/` (via `matplotlib` / `seaborn`), and tabular results (roster configuration, regression tables, risk metrics) are printed to console / exported as `.xlsx` files.

> ⏱️ Note: `Final.py` includes a Monte Carlo simulation of `S = 5,000` scenarios; runtime is measured internally using the `time` module. Execution typically takes a few minutes depending on hardware.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Optimal roster theoretical Win Shares | **65.90** |
| Salary cap surplus achieved | $869,216 |
| Dynamic pricing revenue premium | **+$3.05M (+7.12%)** |
| 95% bootstrap CI of revenue premium | [$2.155M, $4.042M] |
| Structural price elasticity (2SLS) | −3.618 |
| Unhedged 5% CVaR | $30.31M |
| Hedged 5% CVaR | **$34.06M** |
| CVaR improvement per $1 of premium | 15.0× |

---

## 📄 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{cui2025superstar,
  title   = {Superstar-driven running mode: A multiple-objective framework in professional sports},
  author  = {Cui, Wenhao and Bai, Jiaqi and Sha, Zihan and Duan, Conghao and Gan, Wensheng and Wu, Zujian},
  journal = {Preprint submitted to Elsevier},
  year    = {2025}
}
```

---

## 📬 Contact

For questions regarding the code or data, please contact the corresponding author:

- **Wensheng Gan** — wsgan001@gmail.com
- **Zujian Wu** — zujian.wu@jnu.edu.cn

---

## 📜 License

This project is released under the [MIT License](LICENSE). Data files are provided for academic research and reproducibility purposes only; please respect the original terms of use of third-party data sources (Basketball-Reference, HerHoopStats, Spotrac, SeatData.io, Statista).
