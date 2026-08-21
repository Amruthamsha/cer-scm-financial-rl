# Counterfactual Experience Replay via Structural Causal Models for Financial Reinforcement Learning

Final year project implementing **Counterfactual Experience Replay (CER)** — a data-augmentation technique for RL trading agents that uses **Structural Causal Models (SCMs)** to generate counterfactual transitions via Pearl's do-calculus (abduction → action → prediction).

> Optimised for RTX 3090 (24 GB VRAM): AMP mixed precision, `cudnn.benchmark`, `torch.compile`, robust checkpointing, and a full 7-method experiment loop across 30 assets.

## Overview

The project evaluates whether augmenting an RL agent's replay buffer with SCM-generated counterfactual experiences improves sample efficiency and trading performance compared to standard replay strategies. It compares:

- **Dueling DQN** (off-policy baseline)
- **SAC** (off-policy baseline)
- **PPO** (on-policy baseline, no persistent replay buffer)
- **CER-augmented variants** of the above using linear and neural SCMs

## Pipeline

1. **Data & Features** — OHLCV data via `yfinance`, 12 engineered features (returns, volatility, momentum, RSI, Bollinger position, MACD, ATR, OBV, etc.)
2. **Causal Structure Learning** — PC algorithm and NOTEARS-MLP (via `gcastle`) to learn a causal DAG over trading-relevant variables
3. **Structural Causal Model** — linear and neural SEMs implementing Pearl's 3-step do-calculus for counterfactual generation
4. **Trading Environment** — custom Gym-style environment with transaction costs
5. **Agents** — Dueling DQN, SAC, and PPO, each with an optional CER mixin
6. **Ablations** — SCM type (linear vs. neural), fixed vs. rolling SCM, transaction-cost sensitivity, 2×2 factorial design
7. **Statistical Analysis** — journal-grade significance testing, sector-clustered bootstrap tests, bootstrap effect-size CIs, formal power analysis
8. **Results** — per-asset tables, publication-quality figures (PDF + PNG)

## Repository Structure

```
cer-scm-financial-rl/
├── notebooks/
│   └── CER_SCM_RTX3090_v3.ipynb   # main experiment notebook (end-to-end pipeline)
├── results/                        # generated figures, tables, logs (gitignored contents)
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (developed/tuned for RTX 3090, 24 GB VRAM); falls back to CPU automatically

### Installation

```bash
git clone https://github.com/<your-username>/cer-scm-financial-rl.git
cd cer-scm-financial-rl
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running

Open the notebook and run cells top to bottom:

```bash
jupyter notebook notebooks/CER_SCM_RTX3090_v3.ipynb
```

The notebook is organised into 17 numbered sections, from dependency installation through to the final publication-quality figures and paper-claim summary.

## Notes

- The main experiment loop covers **30 assets × 20 trials** across US Tech, US Finance, and Asian markets — this is compute-intensive; a modern GPU is strongly recommended.
- `gcastle==1.0.3` is pinned to avoid a `NotearsNonlinear` API breaking change in later versions.

## License

MIT — see [LICENSE](LICENSE).
