# Battery Energy Storage System (BESS) Arbitrage Optimization

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A techno-economic analysis framework for optimizing Battery Energy Storage System (BESS) arbitrage strategies in the German Day-Ahead electricity market. This project implements a quadratic optimization model to simulate charging/discharging decisions and evaluate achievable revenues under various physical and economic constraints.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Overview

This project analyzes the profitability of large-scale battery storage systems participating in energy arbitrage on the German Day-Ahead market. The model considers:

- **Price Impact**: Large batteries influence market prices when trading (endogenous price effect)
- **Technical Constraints**: Round-trip efficiency, C-rate limitations, state-of-charge bounds
- **Economic Factors**: Degradation costs modeled as hurdle rates

The analysis covers **2019-2024** market data and compares multiple battery capacities (4 MWh to 10,000 MWh).

## Key Features

- **Quadratic Optimization**: Uses CVXPY with OSQP solver for efficient convex optimization
- **Price Impact Modeling**: Implements linear price impact function (α parameter)
- **Multi-Year Simulation**: Continuous simulation across 6 years with SOC carryover
- **Parallel Processing**: Leverages multiprocessing for faster computation
- **Comprehensive Visualization**: Publication-ready plots for academic use
- **Sensitivity Analysis**: Parameter studies for hurdle rate, slope (α), and C-rate

## Project Structure

```
BESS_Optimization/
│
├── 01_data_fetch.py                    # Downloads market data from ENTSO-E API
├── 02_Arbitrage_Optimizer.py           # Core simulation engine (quadratic optimization)
├── 03_analysis_plots.py                # Visualization and analysis scripts
│
├── 04_market_data_2019_2025.csv        # Historical market data
├── 05_simulation_decisions_with_pt.csv # Hourly dispatch decisions
├── 06_arbitrage_summary_with_pt.csv    # Simulation results summary
│
├── plot_*.png                          # Generated figures
├── requirements.txt                    # Python dependencies
├── .env.example                        # API key template
└── LICENSE                             # MIT License
```

## Installation

### Prerequisites

- Python 3.8 or higher
- ENTSO-E Transparency Platform API key ([Register here](https://transparency.entsoe.eu/))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/knoppi10/BESS_Optimization.git
   cd BESS_Optimization
   ```

2. **Create and activate virtual environment**
   ```bash
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your ENTSO-E API key
   ```

## Usage

Execute the scripts in the following order:

### 1. Download Market Data
```bash
python 01_data_fetch.py
```
Downloads Day-Ahead prices from ENTSO-E for 2019-2024. Output: `04_market_data_2019_2025.csv`

### 2. Run Optimization
```bash
python 02_Arbitrage_Optimizer.py
```
Executes the BESS arbitrage simulation for all scenarios. Uses parallel processing. Output: `06_arbitrage_summary_with_pt.csv`, `05_simulation_decisions_with_pt.csv`

### 3. Generate Visualizations
```bash
python 03_analysis_plots.py
```
Creates all analysis plots. Output: `plot_*.png` files

## Methodology

### Optimization Model

The model maximizes arbitrage revenue subject to battery constraints:

$$\max \sum_{t=1}^{T} \left[ p_t \cdot (d_t - c_t) - \alpha \cdot (d_t - c_t)^2 \right] - h \cdot \text{cycles}$$

Where:
- $p_t$: Day-Ahead price at time $t$ (€/MWh)
- $d_t, c_t$: Discharge and charge power (MW)
- $\alpha$: Price impact coefficient (€/MW²h)
- $h$: Hurdle rate / degradation cost (€/cycle)

### Scenarios

| Scenario | Capacity (MWh) | Price Impact (α) |
|----------|----------------|------------------|
| 4        | 4              | 0 (price-taker)  |
| 10       | 10             | 0.01             |
| 100      | 100            | 0.01             |
| 1000     | 1,000          | 0.01             |
| 10000    | 10,000         | 0.01             |
| *_pt     | (same)         | 0 (price-taker)  |

### Key Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Round-trip Efficiency | 85% | η_charge × η_discharge |
| C-Rate | 0.25 | Power/Capacity ratio |
| Hurdle Rate | 5 €/MWh | Minimum spread for trading |
| Slope (α) | 0.01 €/MW²h | Price impact coefficient |

## Results

### Key Findings

- **Price Impact Effect**: Larger batteries (>100 MWh) experience significant revenue reduction due to market price impact
- **Optimal Sizing**: Medium-sized systems (10-100 MWh) achieve best specific revenue (€/MWh capacity)
- **Year Variability**: 2022 showed exceptional arbitrage opportunities due to energy crisis volatility

### Generated Plots

| Plot | Description |
|------|-------------|
| `plot_1_arbitrage_per_mwh.png` | Specific revenue by battery size |
| `plot_2_cycles.png` | Full equivalent cycles by capacity |
| `plot_3_change_rates.png` | Revenue/cycle change rates |
| `plot_5_yearly_comparison.png` | Year-by-year performance |
| `plot_5b_price_impact_cost.png` | Price-taker vs. price-maker comparison |
| `plot_6_sensitivity.png` | Sensitivity analysis |
| `plot_spread_boxplot.png` | Daily price spread evolution |

## Data Sources

- **Day-Ahead Prices**: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- **Market Area**: Germany (DE-LU bidding zone)
- **Time Period**: January 2019 – December 2024
- **Resolution**: Hourly

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ENTSO-E for providing transparent market data access
- CVXPY and OSQP developers for the optimization framework

---

*Developed for academic research in energy economics.*
