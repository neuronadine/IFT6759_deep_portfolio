#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize
import yfinance as yf
import time
import glob
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# path to folder containing csv files. Each files correspond to an asset (stock)
folder_path = os.path.join("stocks", "datasets","dj30","raw", "*.csv")
csv_files = glob.glob(folder_path)
data_frames = []

for file in csv_files:
    # each file name is the stock symbol, so we simply extract sticker from the file's name
    symbol = os.path.splitext(os.path.basename(file))[0]
    print(f"processing {symbol} from {file}...")
    
    # parse "Date" column and filter the desired timeline (range). To be updated later !!!(problem with yahoo"s API)
    df = pd.read_csv(file, parse_dates=['Date'])
    #df = df.loc[~df.index.duplicated(keep='first')]
    mask = (df['Date'] >= '2010-01-01') & (df['Date'] <= '2018-12-21')
    df = df.loc[mask]
    # we only consider "Date" and "Adj Close" columns
    df = df[['Date', 'Adj Close']].set_index('Date')
    df.rename(columns={'Adj Close': symbol}, inplace=True)
    data_frames.append(df)

merged_df = pd.concat(data_frames, axis=1)
merged_df.sort_index(inplace=True)

# non overlapping windows to compute correlation matrices
log_returns = np.log(merged_df / merged_df.shift(1)).dropna()
window_size = 120
correlation_matrices = []
for start in range(0, len(log_returns) - window_size + 1, window_size):
    window_data = log_returns.iloc[start:start+window_size]
    corr_matrix = window_data.corr().values
    correlation_matrices.append(corr_matrix)
    
# we vectorize the upper triangular part
n_assets = log_returns.shape[1]
features = [mat[np.triu_indices(n_assets, k=1)] for mat in correlation_matrices]
features = np.array(features)

# Ward's method for hierarchical clustering
Z = linkage(features, method='ward')
n_clusters = 4
clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

# representative corr matrix for each cluster
rep_corr_matrices = []
for cluster_id in range(1, n_clusters + 1):
    indices = np.where(clusters == cluster_id)[0]
    avg_corr = np.mean([correlation_matrices[i] for i in indices], axis=0)
    rep_corr_matrices.append(avg_corr)
    
annual_factor = 252  # trading days per year
annual_drift = log_returns.mean() * annual_factor # drift (return) for each asset
annual_vol = log_returns.std() * np.sqrt(annual_factor)  # volatility for each asset
mu = annual_drift.values    
sigmas = annual_vol.values   

# for starting prices, we use the last available prices from our original df
S0s = merged_df.iloc[-1].values

# pre-simulation config
T = 1       # 1 year
dt = 1 / 252  # daily step
n_steps = int(T / dt)

def simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix):
    """
    single price trajectory using the Black-Scholes-Merton model
    """
    n_assets = len(S0s)
    n_steps = int(T / dt)
    prices = np.zeros((n_steps + 1, n_assets))
    prices[0] = S0s
    L = np.linalg.cholesky(corr_matrix)  # we use cholesky for correlated noise
    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, size=n_assets)
        correlated_Z = L @ Z
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * correlated_Z)
    return prices

def simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr_matrix, num_paths=1000):
    """
    here, we generate multiple simulation paths for a given correlation matrix
    """
    sims = []
    for _ in range(num_paths):
        sim = simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix)
        sims.append(sim)
    return np.array(sims)

# generate multiple simulation trajectories for each representative correlation matrix, 
all_simulations = {}
num_paths = 1000  # 1000 trajectories per rep matrix
for idx, corr in enumerate(rep_corr_matrices):
    sims = simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr, num_paths=num_paths)
    all_simulations[idx] = sims
    #print(f"representative Matrix {idx+1}: {sims.shape[0]} trajectories generated")

# we combine all simulations
combined_simulations = np.concatenate(list(all_simulations.values()), axis=0)
dummy_dates = pd.date_range(start="2018-01-01", periods=combined_simulations.shape[1], freq="B")
asset_names = [f"Stock{i+1}" for i in range(combined_simulations.shape[2])]
train_episode_df = pd.DataFrame(combined_simulations[np.random.choice(combined_simulations.shape[0])],
                                index=dummy_dates, columns=asset_names)

class PortfolioEnv:
    def __init__(self, price_data, window_obs=60, window_state=120):
        # price_data: df with one simulation trajectory.
        self.price_data = price_data.reset_index(drop=True)
        self.window_obs = window_obs  # observations are asset returns (log returns) on the window_obs !
        self.window_state = window_state
        self.n_assets = price_data.shape[1]
        self.reset()
    
    def reset(self):
        self.current_step = max(self.window_obs, self.window_state)
        self.done = False
        self.portfolio_value = 1.0
        self.history = [self.portfolio_value]
        self.weights = np.ones(self.n_assets) / self.n_assets  # initial allocation
        return self._get_state()
    
    def step(self, action):
        self.weights = action
        current_prices = self.price_data.iloc[self.current_step].values
        next_prices = self.price_data.iloc[self.current_step+1].values
        asset_returns = (next_prices / current_prices) - 1
        portfolio_return = np.dot(self.weights, asset_returns)
        self.portfolio_value *= (1 + portfolio_return)
        self.history.append(self.portfolio_value)
        self.current_step += 1
        if self.current_step >= len(self.price_data)-1:
            self.done = True
        reward = portfolio_return
        return self._get_state(), reward, self.done, {}
    
    def _get_state(self):
        # obsessed are represented by last window_obs days of log returns
        obs_data = self.price_data.iloc[self.current_step - self.window_obs:self.current_step]
        obs_returns = np.log(obs_data / obs_data.shift(1)).dropna().values.T
        if obs_returns.shape[1] < self.window_obs:
            pad = np.zeros((self.n_assets, self.window_obs - obs_returns.shape[1]))
            obs_returns = np.hstack((obs_returns, pad))
        # state:last window_state log returns
        hist = np.array(self.history)
        if len(hist) < self.window_state+1:
            state_data = np.log(hist[1:] / hist[:-1])
            state_data = np.pad(state_data, (self.window_state - len(state_data), 0), 'constant')
        else:
            window_hist = hist[-(self.window_state+1):]
            state_data = np.log(window_hist[1:] / window_hist[:-1])
        state_data = state_data[np.newaxis, :]
        obs_tensor = torch.tensor(obs_returns, dtype=torch.float32).unsqueeze(0)
        state_tensor = torch.tensor(state_data, dtype=torch.float32).unsqueeze(0)
        return (obs_tensor, state_tensor)
    
    def render(self):
        plt.plot(self.history)
        plt.title("Portfolio Value")
        plt.show()

def generate_actions(n_assets, n_actions=50):
    actions = []
    for _ in range(n_actions):
        w = np.random.rand(n_assets)
        w = w / w.sum()
        actions.append(w)
    return np.array(actions)

actions = generate_actions(train_episode_df.shape[1], n_actions=50)

class RLPortfolioManager(nn.Module):
    def __init__(self, obs_channels, obs_length, state_channels, state_length, num_actions):
        super(RLPortfolioManager, self).__init__()
        self.obs_conv = nn.Sequential(
            nn.Conv1d(in_channels=obs_channels, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_obs_length = obs_length - 4
        self.fc_obs = nn.Linear(32 * conv_obs_length, 128)
        
        self.state_conv = nn.Sequential(
            nn.Conv1d(in_channels=state_channels, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_state_length = state_length - 4
        self.fc_state = nn.Linear(16 * conv_state_length, 128)
        
        self.fc_combined = nn.Linear(128+128, 128)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, observation, state):
        obs_features = self.obs_conv(observation)
        obs_features = obs_features.view(obs_features.size(0), -1)
        obs_features = F.relu(self.fc_obs(obs_features))
        
        state_features = self.state_conv(state)
        state_features = state_features.view(state_features.size(0), -1)
        state_features = F.relu(self.fc_state(state_features))
        
        combined = torch.cat([obs_features, state_features], dim=1)
        combined = F.relu(self.fc_combined(combined))
        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)
        return policy_logits, value

obs_channels = train_episode_df.shape[1]  # 29 assets
obs_length = 60  # log returns window (we consider 60, but it can be adjusted !!)
state_channels = 1
state_length = 120
num_actions = actions.shape[0]

policy_net = RLPortfolioManager(obs_channels, obs_length, state_channels, state_length, num_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

def train_agent(policy_net, optimizer, actions, combined_simulations, num_episodes=500, window_obs=60, window_state=120, gamma=0.99):
    for ep in range(num_episodes):
        # random simulation trajectory for each episode
        episode_idx = np.random.choice(combined_simulations.shape[0])
        dummy_dates = pd.date_range(start="2018-01-01", periods=combined_simulations.shape[1], freq="B")
        asset_names = [f"Stock{i+1}" for i in range(combined_simulations.shape[2])]
        train_episode_df = pd.DataFrame(combined_simulations[episode_idx],
                                        index=dummy_dates, columns=asset_names)
        
        #new env for each episode
        env = PortfolioEnv(train_episode_df, window_obs=window_obs, window_state=window_state)
        state = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        done = False
        #run for each episode
        while not done:
            obs, port_state = state
            policy_logits, value = policy_net(obs, port_state)
            probs = F.softmax(policy_logits, dim=1)
            m = torch.distributions.Categorical(probs)
            action_idx = m.sample()
            log_prob = m.log_prob(action_idx)
            value = value.squeeze(1)
            action = actions[action_idx.item()]
            
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state
        
        #discounted returns.
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Convert collected values to a tensor.
        values = torch.stack(values).squeeze(1)
        log_probs = torch.stack(log_probs)
        
        # Calculate advantage and losses.
        advantage = returns - values.detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Episode {ep+1}/{num_episodes}, Loss: {loss.item():.4f}, Total Reward: {np.sum(rewards):.4f}")
    
    return policy_net

trained_policy = train_agent(policy_net, optimizer, actions, combined_simulations, num_episodes=100)


historical_test_df = merged_df
    
env_test = PortfolioEnv(historical_test_df, window_obs=60, window_state=120)

def test_agent(env, policy_net, actions):
    state = env.reset()
    done = False
    test_rewards = []
    while not done:
        obs, port_state = state
        with torch.no_grad():
            policy_logits, _ = policy_net(obs, port_state)
            probs = F.softmax(policy_logits, dim=1)
        action_idx = torch.argmax(probs, dim=1)
        action = actions[action_idx.item()]
        state, reward, done, _ = env.step(action)
        test_rewards.append(reward)
    total_return = np.prod([1 + r for r in test_rewards])
    return total_return

rl_test_return = test_agent(env_test, trained_policy, actions)
print("RL test return:", rl_test_return)

historical_test_df = merged_df
env_test = PortfolioEnv(historical_test_df, window_obs=60, window_state=120)

def sharpe_ratio(rewards, risk_free_rate=0.04, periods_per_year=252):
    rewards = np.array(rewards)
    mean_return = np.mean(rewards)
    std_return = np.std(rewards)
    if std_return == 0:
        return np.nan
    sharpe_ratio = mean_return / std_return * np.sqrt(periods_per_year)
    return sharpe_ratio

# evaluation on a given environment (episode)
def test_agent_with_rewards(env, policy_net, actions):
    state = env.reset()
    done = False
    rewards = []
    while not done:
        obs, port_state = state
        with torch.no_grad():
            policy_logits, _ = policy_net(obs, port_state)
            probs = F.softmax(policy_logits, dim=1)
        action_idx = torch.argmax(probs, dim=1)
        action = actions[action_idx.item()]
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    total_return = np.prod([1 + r for r in rewards])
    return total_return, rewards

# evaluation on 2 year intervals
def evaluate_fixed_intervals(historical_df, policy_net, actions, window_obs=60, window_state=120):
    
    intervals = [("2010-01-01", "2012-01-01"),
                 ("2012-01-01", "2014-01-01"),
                 ("2014-01-01", "2016-01-01"),
                 ("2016-01-01", "2018-01-01")]
    
    results = {}
    for start, end in intervals:
        # data selection for the intervak
        interval_df = historical_df.loc[start:end].copy()
        if len(interval_df) < (window_obs + 1):
            print(f"Interval {start} to {end} is too short. Skipping.")
            continue
        # env for the interval
        env_interval = PortfolioEnv(interval_df, window_obs=window_obs, window_state=window_state)
        total_return = test_agent(env_interval, policy_net, actions)
        results[(start, end)] = total_return
        print(f"Interval {start} to {end}: Total Portfolio Return = {total_return:.4f}")
    return results

#evaluation
interval_results = evaluate_fixed_intervals(historical_test_df, trained_policy, actions, window_obs=60, window_state=120)

def simulate_benchmark(env, weights):
    state = env.reset()
    done = False
    rewards = []
    while not done:
        state, reward, done, _ = env.step(weights)
        rewards.append(reward)
    total_return = np.prod([1 + r for r in rewards])
    return total_return, rewards

#MVO with rf=0
def optimize_portfolio(mu, cov):
    n = len(mu)
    w0 = np.ones(n) / n
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n)]
    
    def neg_sharpe(w, mu, cov):
        port_return = np.dot(w, mu)
        port_std = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -port_return / port_std if port_std > 0 else 1e6

    result = minimize(neg_sharpe, w0, args=(mu, cov), bounds=bounds, constraints=constraints)
    return result.x if result.success else w0

intervals = [("2010-01-01", "2012-01-01"),
             ("2012-01-01", "2014-01-01"),
             ("2014-01-01", "2016-01-01"),
             ("2016-01-01", "2018-01-01")]

results = {}

for start, end in intervals:
    interval_df = historical_test_df.loc[start:end].copy()
    window_obs = 60
    window_state = 120
    if len(interval_df) < (window_obs + 1):
        print(f"Interval {start} to {end} is too short. Skipping.")
        continue
    env_interval = PortfolioEnv(interval_df, window_obs=window_obs, window_state=window_state)
    
    #RL with returns and rewards
    rl_total_return, rl_rewards = test_agent_with_rewards(env_interval, trained_policy, actions)
    rl_sharpe = sharpe_ratio(rl_rewards)
    
    # MVO test
    returns = np.log(interval_df / interval_df.shift(1)).dropna()
    mu = returns.mean().values
    cov = returns.cov().values
    mvo_weights = optimize_portfolio(mu, cov)
    mvo_total_return, mvo_rewards = simulate_benchmark(env_interval, mvo_weights)
    mvo_sharpe = sharpe_ratio(mvo_rewards)
    
    #equal weights
    n_assets = interval_df.shape[1]
    equal_weights = np.ones(n_assets) / n_assets
    equal_total_return, equal_rewards = simulate_benchmark(env_interval, equal_weights)
    equal_sharpe = sharpe_ratio(equal_rewards)
    
    # results
    results[(start, end)] = {
        "RL": {"Total Return": rl_total_return, "Sharpe": rl_sharpe},
        "MVO": {"Total Return": mvo_total_return, "Sharpe": mvo_sharpe},
        "Equal_weights": {"Total Return": equal_total_return, "Sharpe": equal_sharpe}
    }
    
    print(f"Interval {start} to {end}:")
    print(f"  RL     - Total Return: {rl_total_return:.4f}, Sharpe Ratio: {rl_sharpe:.4f}")
    print(f"  MVO - Total Return: {mvo_total_return:.4f}, Sharpe Ratio: {mvo_sharpe:.4f}")
    print(f"  Equal Weights - Total Return: {equal_total_return:.4f}, Sharpe Ratio: {equal_sharpe:.4f}")
    print("---------------------------------------------------")

