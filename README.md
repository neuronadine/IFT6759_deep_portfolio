# Deep Portfolio

<Deep Portfolio>

Source code available at https://github.com/neuronadine/IFT6759_deep_portfolio

---

## 1  Project overview

This repository contains all code for **Deep Portfolio**, an exploration of deep-reinforcement-learning (DRL) methods for multi-asset portfolio management.  
We implement several DRL agents and benchmark them against a classical mean–variance optimiser (MVO):

Implemented components  

| Category | Main files |
|----------|------------|
| **Actor–Critic baselines** | `RL_A2C.py` · `A2C_LSTM.ipynb` |
| **Recurrent PPO family** | `reccurrentPPO.py` · `RPPO_sentiment.ipynb` |
| **CNN/LSTM price + sentiment agents** | `RL_conv_MVO.ipynb` · `RL_conv_sentiment.ipynb` · `RL_lstm_sentiment.ipynb` |
| **Analysis & utilities** | `MC_draft.ipynb` · `optimal_clusters.ipynb` · `news_papers_sentiment_extraction.ipynb` · `scripts/news_llm.py` |

All experiments use daily close prices for Dow Jones-30 constituents (2010-01-01 → 2023-01-01) and, optionally, news-sentiment signals derived from the **FinSen** dataset.

---

---

## 2  Repository layout
```text
├─ data/                       # (git-ignored) cached datasets
│  └─ stocks/datasets/dj30/raw/*.csv
├─ notebooks/                  
│  └─ A2C_LSTM.ipynb
│  └─ MC_draft.ipynb
│  └─ news_papers_sentiment_extraction.ipynb
│  └─ optimal_clusters.ipynb
│  └─ RecurrentPPO.py
│  └─ RL_A2C.py
│  └─ RL_conv_MVO.ipynb
│  └─ RL_conv_sentiment.ipynb
│  └─ RL_lstm_sentiment.ipynb
│  └─ RPPO_sentiment.ipynb
├─ scripts/
│  └─ news_llm.py              
└─ README.md                   
```

## 3  Environment

* **Python 3.12**
* Scientific stack `numpy ≥1.26` `pandas ≥2.1` `scipy ≥1.11` `matplotlib ≥3.8`
* Deep learning `torch 2.2.*` (+CUDA 12 if available)
* RL `stable-baselines3 2.3.0` `sb3-contrib 2.3.0` `gymnasium 0.29`
* Data `yfinance 0.2.*` `qlib 0.9.*` `statsmodels 0.14.*`
* Sentiment (optional) `google-cloud-aiplatform 2.*` `vertexai 1.*`
* Misc `pydantic 2.*` `wandb 0.17.*`

Generate/update the full lock-file with

```bash
python -m pip freeze > requirements.txt
```

## 4 Quick Start
#### Clone
git clone https://github.com/neuronadine/IFT6759_deep_portfolio
cd IFT6759_deep_portfolio

#### Isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

#### Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


#### Scripts
`RL_A2C.py`                  
Convolutional actor–critic baseline

```bash
python RL_A2C.py --episodes 500
```

`A2C_LSTM.ipynb`              
LSTM variant of the above.
```bash
jupyter nbconvert --execute --to notebook A2C_LSTM.ipynb
```

`reccurrentPPO.py`              
Recurrent PPO pipeline (logs to WandB).
```bash
python reccurrentPPO.py --total_timesteps 1000000
```

`RPPO_sentiment.ipynb`          
Recurrent PPO with news‑sentiment features.
```bash
jupyter nbconvert --execute RPPO_sentiment.ipynb
```

`RL_conv_MVO.ipynb`
CNN policy vs 252‑day MVO benchmark (prices only).
```bash
jupyter nbconvert --execute RL_conv_MVO.ipynb
```

`RL_conv_sentiment.ipynb`       
CNN policy that fuses price + sentiment.
```bash
jupyter nbconvert --execute RL_conv_sentiment.ipynb
```

`RL_lstm_sentiment.ipynb`       
LSTM counterpart of the previous agent.

```bash
jupyter nbconvert --execute RL_lstm_sentiment.ipynb
```

`MC_draft.ipynb`                
Monte‑Carlo simulator & correlation sanity checks.
```bash
jupyter nbconvert --execute MC_draft.ipynb
```

`optimal_clusters.ipynb`        
Elbow/silhouette/gap‑stat diagnostics.
```bash
jupyter nbconvert --execute optimal_clusters.ipynb
```

`news_papers_sentiment_extraction.ipynb`
Extracts sector/stock sentiment from FinSen via Gemini 1.5‑pro
```bash
jupyter nbconvert --execute news_papers_sentiment_extraction.ipynb
```

## 6 Datasets
FinSen news corpusFinSen news corpus

Original repo: https://github.com/EagleAdelaide/FinSen_Dataset

DJ-30 daily prices
Auto-downloaded by any training script via `yfinance`


