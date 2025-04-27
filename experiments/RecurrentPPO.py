import os
import glob
import math
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

import gymnasium as gym
from gymnasium.spaces import Box

import torch
import torch.nn.functional as F

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO

import wandb
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize
from typing import List

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load and preprocess real data

folder_path = os.path.join("/teamspace/studios/this_studio/IFT6759_deep_portfolio/dj30/raw", "*.csv")
csv_files = glob.glob(folder_path)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {folder_path}")

dfs = []
print("--- Starting Data Loading ---")
for file in csv_files:
    symbol = os.path.splitext(os.path.basename(file))[0]
    try:
        df = pd.read_csv(file, parse_dates=["Date"])
        df = (
            df.set_index("Date")
              .sort_index()
              .loc["2010-01-01":"2019-12-31", ["Adj Close"]]
              .rename(columns={"Adj Close": symbol})
        )
        df = df.ffill().bfill()
        if (df[symbol] <= 1e-6).any():
            continue
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {symbol}: {e}")

if not dfs:
    raise ValueError("No valid data loaded.")

merged_df = pd.concat(dfs, axis=1).sort_index().ffill().bfill()

# Splits
TRAIN_START, TRAIN_END = "2010-01-01", "2016-01-01"
# Validation to track progress throughout training
VALID_START, VALID_END = "2016-01-04", "2017-01-03"

train_df = merged_df.loc[TRAIN_START:TRAIN_END]
val_df = merged_df.loc[VALID_START:VALID_END]

print(f"Train: {train_df.shape}, Val: {val_df.shape}")

# Simulate synthetic one-year data immediately after training period

# Compute log-returns on the training data
train_log_returns = np.log(train_df / train_df.shift(1)).dropna()
n_assets = train_log_returns.shape[1]

# Build rolling correlation matrices
window_size = 120
correlation_matrices = []
for start in range(0, len(train_log_returns) - window_size + 1, window_size):
    window = train_log_returns.iloc[start : start + window_size]
    correlation_matrices.append(window.corr().values)

# Cluster the correlation matrices hierarchically
features = np.array([mat[np.triu_indices(n_assets, k=1)] for mat in correlation_matrices])
Z = linkage(features, method="ward")
n_clusters = 4
clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

# Compute representative correlation per cluster
rep_corr_matrices = []
for cid in range(1, n_clusters + 1):
    idxs = np.where(clusters == cid)[0]
    avg_corr = np.mean([correlation_matrices[i] for i in idxs], axis=0)
    rep_corr_matrices.append(avg_corr)

annual_factor = 252
mu = train_log_returns.mean().values * annual_factor
sigmas = train_log_returns.std().values * np.sqrt(annual_factor)
S0s = train_df.iloc[-1].values

# years
T = 1
dt = 1 / 252
n_steps = int(T / dt)

def simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix):
    n = len(S0s)
    steps = int(T / dt)
    prices = np.zeros((steps + 1, n))
    prices[0] = S0s
    L = np.linalg.cholesky(corr_matrix)
    for t in range(1, steps + 1):
        Z = np.random.normal(size=n)
        correlated_Z = L @ Z
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigmas**2)*dt + sigmas*np.sqrt(dt)*correlated_Z)
    return prices

def simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr_matrix, num_paths=1000):
    sims = [simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix) for _ in range(num_paths)]
    return np.array(sims)

# Generate and combine simulations
num_paths = 1000
all_sims = []
for corr in rep_corr_matrices:
    sims = simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr, num_paths=num_paths)
    all_sims.append(sims)

# shape: (n_clusters*num_paths, n_steps+1, n_assets)
combined_sims = np.concatenate(all_sims, axis=0)

# Build a df for env
sim_start = train_df.index[-1] + pd.Timedelta(days=1)
dummy_dates = pd.bdate_range(start=sim_start, periods=combined_sims.shape[1], freq="B")
asset_names = train_df.columns.tolist()

# Env
class PortfolioEnv(gym.Env):
    """
    Gymnasium environment for multi-asset portfolio management.
    Observation: flattened window of log-returns, optional TA, covariance.
    Action: raw logits over assets -> softmax -> weights.
    Reward: log portfolio return net of transaction costs.
    """
    def __init__(
        self,
        price_data_df: pd.DataFrame,
        window_obs: int = 60,
        initial_value: float = 1.0,
        transaction_cost_pct: float = 0.001,
        sma_periods=[10, 20],
        rsi_period=14,
        bollinger_period=20,
        bollinger_dev=2,
        macd_fast=12,
        macd_slow=26,
        macd_sign=9,
        use_technical_indicators: bool = False
    ):
        super().__init__()
        self.price_data = price_data_df.reset_index(drop=True)
        self.date_index = price_data_df.index
        self.window_obs = window_obs
        self.n_assets = self.price_data.shape[1]
        self.initial_value = float(initial_value)
        self.transaction_cost_pct = transaction_cost_pct
        self.total_steps = len(self.price_data)
        self.asset_names = price_data_df.columns.tolist()

        if len(self.price_data) < window_obs + 1:
            raise ValueError(f"Data too short ({len(self.price_data)}) for window_obs={window_obs}")

        # Technical indicators setup
        if use_technical_indicators:
            self.sma_periods = sma_periods
            self.rsi_period = rsi_period
            self.bollinger_period = bollinger_period
            self.bollinger_dev = bollinger_dev
            self.macd_fast = macd_fast
            self.macd_slow = macd_slow
            self.macd_sign = macd_sign
            self.indicator_list = (
                [f"SMA_{p}" for p in sma_periods]
                + [f"RSI_{rsi_period}"]
                + [f"%B_{bollinger_period}"]
                + [f"MACD_{macd_fast}_{macd_slow}", f"MACD_Signal_{macd_sign}"]
            )
        else:
            self.sma_periods = []
            self.rsi_period = 0
            self.bollinger_period = 0
            self.bollinger_dev = 0
            self.macd_fast = 0
            self.macd_slow = 0
            self.macd_sign = 0
            self.indicator_list = []

        self.n_indicators = len(self.indicator_list)
        self.all_log_returns = self._precompute_log_returns()
        self.all_ta_features = (self._precompute_technical_indicators() if use_technical_indicators else pd.DataFrame(index=self.price_data.index))

        # feature dimension
        self.total_features = (
            self.n_assets
            + self.n_assets * self.n_indicators
            + self.n_assets * self.n_assets
        )
        self.state_seq_len = self.window_obs - 1
        flat_dim = self.state_seq_len * self.total_features
        self.observation_space = Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32)
        self.action_space = Box(-10, 10, shape=(self.n_assets,), dtype=np.float32)

        self.precomputed_states = self._precompute_all_states()
        self.current_step = None
        self.done = None
        self.portfolio_value = None
        self.weights = None
        self.history = None

    def _calculate_sma(self, series, period):
        return series.rolling(period, min_periods=1).mean()

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(50.0)

    def _calculate_bollinger_bands(self, series, period, num_dev):
        sma = series.rolling(period, min_periods=1).mean()
        std = series.rolling(period, min_periods=1).std()
        upper = sma + std * num_dev
        lower = sma - std * num_dev
        pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
        return pct_b.fillna(0.5)

    def _calculate_macd(self, series, fast, slow, signal):
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        return macd_line, signal_line

    def _precompute_log_returns(self):
        prices = self.price_data.replace(0, 1e-9)
        prev = prices.shift(1).replace(0, 1e-9)
        lr = np.log(prices / prev)
        return lr.iloc[1:]

    def _precompute_technical_indicators(self):
        df = pd.DataFrame(index=self.price_data.index)
        for asset in self.asset_names:
            series = self.price_data[asset]
            for p in self.sma_periods:
                df[f"{asset}_SMA_{p}"] = self._calculate_sma(series, p)
            df[f"{asset}_RSI_{self.rsi_period}"] = self._calculate_rsi(series, self.rsi_period)
            df[f"{asset}_%B_{self.bollinger_period}"] = self._calculate_bollinger_bands(series, self.bollinger_period, self.bollinger_dev)
            macd, sig = self._calculate_macd(series, self.macd_fast, self.macd_slow, self.macd_sign)
            df[f"{asset}_MACD_{self.macd_fast}_{self.macd_slow}"] = macd
            df[f"{asset}_MACD_Signal_{self.macd_sign}"] = sig
        df = df.ffill().bfill().fillna(0)
        # reorder columns to match indicator_list ordering
        cols = []
        for ind in self.indicator_list:
            for a in self.asset_names:
                cols.append(f"{a}_{ind}")
        return df[cols]

    def _precompute_all_states(self):
        states = []
        seq = self.state_seq_len
        cov_size = self.n_assets * self.n_assets
        for idx in range(len(self.price_data) - self.window_obs):
            step = self.window_obs + idx
            lr_win = self.all_log_returns.iloc[step-seq:step].fillna(0).values
            ta_win = (self.all_ta_features.iloc[step-seq:step].fillna(0).values if self.n_indicators > 0 else np.empty((seq, 0)))
            cov = self.all_log_returns.iloc[step-seq:step].cov().fillna(0).values.flatten()
            if cov.size != cov_size:
                cov = np.eye(self.n_assets, dtype=np.float32).flatten()
            rows = [np.concatenate([lr_win[t], ta_win[t], cov]) for t in range(seq)]
            states.append(np.concatenate(rows).astype(np.float32))
        return states

    def _get_precomputed_state(self):
        idx = self.current_step - self.window_obs
        if 0 <= idx < len(self.precomputed_states):
            return self.precomputed_states[idx]
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = self.window_obs
        self.done = False
        self.portfolio_value = self.initial_value
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        start_date = self.date_index[self.current_step - 1]
        self.history = [(start_date, self.portfolio_value)]
        return self._get_precomputed_state(), {}

    def step(self, raw_action):
        if self.done:
            return self._get_precomputed_state(), 0.0, True, False, {}

        a = np.asarray(raw_action, dtype=np.float32).flatten()
        tensor = torch.from_numpy(a)
        w = F.softmax(tensor, dim=-1).cpu().numpy()
        if np.isnan(w).any():
            w = np.ones_like(w) / self.n_assets
        w /= w.sum()

        prev_idx = self.current_step - 1
        cur_idx = self.current_step
        if cur_idx >= len(self.price_data):
            self.done = True
            info = {
                "weights": self.weights.copy(),
                "portfolio_value": self.portfolio_value,
                "transaction_costs": 0.0
            }
            return self._get_precomputed_state(), 0.0, True, False, info

        prev_prices = self.price_data.iloc[prev_idx].values
        cur_prices = self.price_data.iloc[cur_idx].values
        safe_prev = np.where(prev_prices <= 1e-9, 1e-9, prev_prices)
        asset_rets = cur_prices / safe_prev - 1.0

        gross_ret = np.dot(self.weights, asset_rets)
        tc_ratio = np.sum(np.abs(w - self.weights)) * self.transaction_cost_pct
        factor = (1 + gross_ret) * (1 - tc_ratio)
        self.portfolio_value *= max(factor, 1e-9)
        reward = math.log(max(factor, 1e-9))

        self.weights = w
        self.current_step += 1
        self.done = self.current_step >= self.total_steps

        date = self.date_index[cur_idx]
        self.history.append((date, self.portfolio_value))

        info = {
            "weights": self.weights.copy(),
            "portfolio_value": self.portfolio_value,
            "transaction_costs": tc_ratio,
        }
        return self._get_precomputed_state(), reward, self.done, False, info

    def render(self, mode="human"):
        if len(self.history) < 2:
            print("Insufficient history to render.")
            return
        dates, vals = zip(*self.history)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, vals)
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.yscale("log")
        plt.title("Portfolio Value Over Time")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

class SimulatedPortfolioEnv(PortfolioEnv):
    def __init__(self,
                 all_paths: np.ndarray,
                 dates: pd.DatetimeIndex,
                 asset_names: List[str],
                 window_obs: int = 60,
                 transaction_cost_pct: float = 0.001):
        # dummy DataFrame so super().__init__ still works
        dummy = pd.DataFrame(
            all_paths[0],
            index=dates,
            columns=asset_names
        )
        super().__init__(
            price_data_df=dummy,
            window_obs=window_obs,
            transaction_cost_pct=transaction_cost_pct
        )
        self.all_paths = all_paths
        self.dates = dates
        self.asset_names = asset_names

    def reset(self, seed=None, options=None):
        
        # pick a random path
        idx = np.random.randint(len(self.all_paths))
        path = self.all_paths[idx]
        
        # rebuild price_data and other arrays
        self.price_data = pd.DataFrame(path, index=self.dates, columns=self.asset_names)
        self.all_log_returns = self._precompute_log_returns()
        if self.n_indicators > 0:
            self.all_ta_features = self._precompute_technical_indicators()
        self.precomputed_states = self._precompute_all_states()

        # Also call normal reset
        return super().reset(seed=seed, options=options)


# Metrics / Benchmarks

def calculate_metrics(returns, risk_free_rate=0.0, periods_per_year=252):
    r = np.asarray(returns)
    if r.size < 2:
        return np.nan, np.nan, np.nan
    cum_ret = np.prod(1 + r) - 1
    ann_ret = (1 + cum_ret) ** (periods_per_year / r.size) - 1
    vol = r.std() * np.sqrt(periods_per_year)
    sr = ((r.mean() * periods_per_year - risk_free_rate) / vol if vol > 0 else np.nan)
    return ann_ret, vol, sr

def simulate_benchmark(data_df, weights_strategy="equal", initial_value=1.0, transaction_cost_pct=0.001):
    prices = data_df.values
    dates = data_df.index
    n = data_df.shape[1]

    val = initial_value
    pv_list = [val]
    ret_list = []
    dt_list = [dates[0]]

    if isinstance(weights_strategy, str) and weights_strategy == "equal":
        w = np.ones(n) / n
        rebalance = True
    else:
        w = np.asarray(weights_strategy, dtype=np.float32)
        w /= w.sum()
        rebalance = False

    for i in range(1, len(prices)):
        prev = val
        prev_prices = prices[i - 1]
        cur_prices = prices[i]
        safe_prev = np.where(prev_prices <= 1e-9, 1e-9, prev_prices)
        rets = cur_prices / safe_prev - 1
        gross = np.dot(w, rets)
        tmp = prev * (1 + gross)
        tc = 0.0

        if rebalance:
            by_asset = prev * w
            end_no_tc = by_asset * (1 + rets)
            total_no_tc = end_no_tc.sum()
            drifted = end_no_tc / total_no_tc if total_no_tc > 0 else np.zeros_like(w)
            tc = np.sum(np.abs(drifted - w)) * transaction_cost_pct
            val = tmp * (1 - tc)
            w = np.ones(n) / n
        else:
            val = tmp
            by_asset = prev * w
            end = by_asset * (1 + rets)
            total = end.sum()
            w = end / total if total > 0 else np.zeros_like(w)

        val = max(val, 1e-9)
        ret_list.append(val / prev - 1)
        pv_list.append(val)
        dt_list.append(dates[i])

    return val, ret_list, pv_list, dt_list

def optimize_portfolio_mvo(price_df, periods_per_year=252):
    lr = np.log(price_df / price_df.shift(1)).dropna()
    n = lr.shape[1]
    if lr.shape[0] < n + 2:
        return np.ones(n) / n

    mu = lr.mean().values * periods_per_year
    cov = lr.cov().values * periods_per_year + np.eye(n) * 1e-9

    def neg_sharpe(w):
        ret = w.dot(mu)
        var = w.dot(cov).dot(w)
        return -ret / np.sqrt(var) if var > 0 else 1e9

    bounds = [(0, 1)] * n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    if not res.success:
        return w0
    w = np.clip(res.x, 0, 1)
    return w / w.sum()

# Training

def train_recurrent_ppo(
    real_train_df: pd.DataFrame,
    all_sim_paths: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    asset_names: List[str],
    val_df: pd.DataFrame,
    total_timesteps: int = 200_000,
    window_obs: int = 60,
    transaction_cost_pct: float = 0.001,
    model_save_path: str = "./recurrent_ppo_model",
    num_real_envs: int = 1,
    num_sim_envs: int = 1,
    **ppo_kwargs
):

    def make_real_env():
        env = PortfolioEnv(
            price_data_df=real_train_df,
            window_obs=window_obs,
            transaction_cost_pct=transaction_cost_pct,
        )
        return Monitor(env)

    def make_sim_env(): 
        env = SimulatedPortfolioEnv(
            all_paths=all_sim_paths,
            dates=sim_dates,
            asset_names=asset_names,
            window_obs=window_obs,
            transaction_cost_pct=transaction_cost_pct,
        )

        return Monitor(env)

    def make_val_env():
        env = PortfolioEnv(
            price_data_df=val_df,
            window_obs=window_obs,
            transaction_cost_pct=transaction_cost_pct,
        )
        return Monitor(env)

    # parallel envs: real + synthetic
    train_env_fns = [make_real_env] * num_real_envs + [make_sim_env] * num_sim_envs
    train_env = SubprocVecEnv(train_env_fns)
    val_env = DummyVecEnv([make_val_env])

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=None,
        device='cuda',
        **ppo_kwargs
    )

    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=model_save_path,
        n_eval_episodes=1,
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    print(f"--- Training for {total_timesteps} timesteps with {num_real_envs} real env(s) + {num_sim_envs} synthetic env(s) ---")
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(os.path.join(model_save_path, "recurrent_ppo_final"))
    return model

# Eval

def evaluate_model(model, data_df, window_obs=60, transaction_cost_pct=0.001):
    env = PortfolioEnv(
        price_data_df=data_df,
        window_obs=window_obs,
        transaction_cost_pct=transaction_cost_pct
    )
    obs, _ = env.reset()
    terminated = truncated = False
    lstm_states = None
    episode_start = True

    all_rewards = []
    values = [env.portfolio_value]
    dates = [env.history[-1][0]]

    while not (terminated or truncated):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        all_rewards.append(reward)
        values.append(env.portfolio_value)
        dates.append(env.history[-1][0])
        episode_start = terminated or truncated

    simple_rets = np.exp(all_rewards) - 1.0
    return env.portfolio_value, simple_rets, values, dates


test_periods = [
    ("2016-01-01", "2018-01-01"),
    ("2018-01-01", "2020-01-01"),
    # ("2020-01-01", "2022-01-01"),
]

if __name__ == "__main__":
    
    # for logging
    wandb.init(
        project="IFT6759_SB3",
        name="ppo_lstm_portfolio_sim",
        config={
            "seed": SEED,
            "window_obs": 60,
            "transaction_cost_pct": 0.001,
            "total_timesteps": 1_000_000,
        },
    )

    model_dir = "./recurrent_ppo_saved"
    os.makedirs(model_dir, exist_ok=True)

    # parallelization
    num_real_envs = 4
    num_sim_envs = 4

    # train
    model = train_recurrent_ppo(
        real_train_df=train_df,
        all_sim_paths=combined_sims,
        sim_dates=dummy_dates,
        asset_names=asset_names,
        val_df=val_df,
        total_timesteps=wandb.config.total_timesteps,
        window_obs=wandb.config.window_obs,
        transaction_cost_pct=wandb.config.transaction_cost_pct,
        model_save_path=model_dir,
        num_real_envs=num_real_envs,
        num_sim_envs=num_sim_envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # model = RecurrentPPO.load("./recurrent_ppo_saved/recurrent_ppo_final_1m5")
    # model = RecurrentPPO.load("./recurrent_ppo_saved/best_model_it_10000")

    # evaluation on each test windows
    for start, end in test_periods:
        print(f"Starting Eval for dates = {start}:{end}")
        test_df_i = merged_df.loc[start:end]

        t_val, t_rets, t_vals, t_dates = evaluate_model(
            model,
            test_df_i,
            window_obs=wandb.config.window_obs,
            transaction_cost_pct=wandb.config.transaction_cost_pct,
        )
        t_arn, t_vol, t_sh = calculate_metrics(t_rets)

        ew_val, ew_rets, ew_vals, ew_dates = simulate_benchmark(test_df_i)
        ew_arn, ew_vol, ew_sh = calculate_metrics(ew_rets)

        mvo_w = optimize_portfolio_mvo(test_df_i)
        mvo_val, mvo_rets, mvo_vals, mvo_dates = simulate_benchmark(
            test_df_i, weights_strategy=mvo_w, transaction_cost_pct=0.0
        )
        mvo_arn, mvo_vol, mvo_sh = calculate_metrics(mvo_rets)

        wandb.log({
            f"Test/{start}_{end}/Final": t_val,
            f"Test/{start}_{end}/AR": t_arn,
            f"Test/{start}_{end}/Vol": t_vol,
            f"Test/{start}_{end}/Sharpe": t_sh,
        })

        # Plot
        fig = plt.figure(figsize=(12, 6))
        plt.plot(t_dates, t_vals, label=f"PPO (SR={t_sh:.2f})", linewidth=2)
        plt.plot(ew_dates, ew_vals, label=f"EW (SR={ew_sh:.2f})", linestyle="--")
        plt.plot(mvo_dates, mvo_vals, label=f"MVO (SR={mvo_sh:.2f})", linestyle=":")
        plt.yscale("log")
        plt.title(f"Portfolio Values {start} to {end} (log scale)")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        wandb.log({f"Portfolio Plot {start} to {end}": wandb.Image(fig)})

        plt.close(fig)

    wandb.finish()
