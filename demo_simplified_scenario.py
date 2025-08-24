#====================================================================================
# This code is a part of DeepSatCom Project, which mainly focuses on designing
# deep learning framework for interference management in satellite communications
# Author: Abuzar B. M. Adam
# Date: 15/8/2025
# In this file, we design DSPG+SAC framework for resource allocation in LEO satellite
# networks
#===================================================================================
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env, spaces
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. LEO Satellite Environment
# -------------------------------
class LEOSatelliteEnv(Env):
    def __init__(self, num_users=4, num_subcarriers=6, num_satellites=1, max_power=10.0):
        super(LEOSatelliteEnv, self).__init__()
        self.num_users = num_users
        self.num_subcarriers = num_subcarriers
        self.num_satellites = num_satellites
        self.max_power = max_power
        self.noise_power = 1e-11  # -100 dBm
        self.bandwidth_per_sc = 1e6  # 1 MHz per subcarrier

        # State: [distances (num_users), channel_gains (num_users * num_subcarriers), sat_pos (2)]
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(num_users + num_users * num_subcarriers + 2,), dtype=np.float32
        )

        # Action: [allocation_logits (num_users * num_subcarriers), beam_real (num_users), beam_imag (num_users)]
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(num_users * num_subcarriers + 2 * num_users,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.user_positions = np.random.rand(self.num_users, 2) * 100  # 100km x 100km
        self.satellite_positions = np.random.rand(self.num_satellites, 2) * 100
        self._update_state()
        return self.state.astype(np.float32), {}

    def _compute_channel_gains(self):
        distances = np.linalg.norm(self.user_positions - self.satellite_positions[0], axis=1)
        path_loss = 1 / (distances[:, None] + 1e-3)
        fading = (np.random.randn(self.num_users, self.num_subcarriers) +
                  1j * np.random.randn(self.num_users, self.num_subcarriers)) * 0.1
        h = path_loss * (1 + fading.real)
        return np.abs(h).astype(np.float32)

    def _update_state(self):
        self.distances = np.linalg.norm(self.user_positions - self.satellite_positions[0], axis=1)
        self.channel_gains = self._compute_channel_gains()
        normalized_distances = self.distances / 100.0
        normalized_sat_pos = self.satellite_positions[0] / 100.0
        self.state = np.concatenate([
            normalized_distances,
            self.channel_gains.flatten(),
            normalized_sat_pos
        ])

    def step(self, action):
        # Normalize action to [0,1] range for numerical stability
        action = np.tanh(action)  # Squash to [-1,1]

        # Parse action
        alloc_logits = action[:self.num_users * self.num_subcarriers]
        beam_real = action[self.num_users * self.num_subcarriers:self.num_users * self.num_subcarriers + self.num_users]
        beam_imag = action[self.num_users * self.num_subcarriers + self.num_users:]

        # Subcarrier allocation: one-hot per user
        alloc_logits = alloc_logits.reshape((self.num_users, self.num_subcarriers))
        subcarrier_allocation = np.argmax(alloc_logits, axis=1)

        A = np.zeros((self.num_users, self.num_subcarriers))
        for u in range(self.num_users):
            A[u, subcarrier_allocation[u]] = 1

        # Beamforming weights (complex)
        beam_weights = (beam_real + 1j * beam_imag)
        power_per_user = np.abs(beam_weights) ** 2
        total_power = np.sum(power_per_user)

        # Power constraint
        if total_power > self.max_power:
            beam_weights = beam_weights * np.sqrt(self.max_power / (total_power + 1e-8))

        # Compute data rates
        rates = []
        for u in range(self.num_users):
            sc = subcarrier_allocation[u]
            h_u = self.channel_gains[u, sc]
            signal_power = (h_u * abs(beam_weights[u])) ** 2
            interference = 0.0
            for u2 in range(self.num_users):
                if u2 != u and subcarrier_allocation[u2] == sc:
                    h_u2 = self.channel_gains[u2, sc]
                    interference += (h_u2 * abs(beam_weights[u2])) ** 2
            sinr = signal_power / (interference + self.noise_power)
            rate = self.bandwidth_per_sc * np.log2(1 + sinr)
            rates.append(rate)

        sum_rate = np.sum(rates)
        fairness = (np.sum(rates) ** 2) / (self.num_users * np.sum(np.square(rates)) + 1e-8)
        used_power = np.sum(np.abs(beam_weights)**2)

        # Reward: sum rate + fairness - power penalty
        reward = 1e-6 * sum_rate + 5.0 * fairness - 0.01 * used_power

        # Move users (simulate dynamics)
        self.user_positions += (np.random.randn(*self.user_positions.shape) * 0.5)
        self.user_positions = np.clip(self.user_positions, 0, 100)
        self._update_state()

        terminated = False
        truncated = False
        info = {
            'sum_rate': sum_rate,
            'fairness': fairness,
            'power': used_power
        }

        return self.state.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        print(f"Users: {self.num_users}, Sum Rate: {self.last_info['sum_rate']:.2f} bps")


# -------------------------------
# 2. Custom Feature Extractor
# -------------------------------
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)


# -------------------------------
# 3. Training Callback for Logging
# -------------------------------
class RewardCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_reward += reward

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            if len(self.episode_rewards) % 10 == 0 and self.verbose >= 1:
                print(f"Episode {len(self.episode_rewards)}: Total Reward = {self.current_reward:.2f}")
            self.current_reward = 0.0

        return True


# -------------------------------
# 4. Training Function (with GPU auto-detection)
# -------------------------------
def train_sac_leo():
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create environment
    env = LEOSatelliteEnv(num_users=4, num_subcarriers=6)
    env = DummyVecEnv([lambda: env])

    # Policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=[256, 256]  # Larger network for better representation
    )

    # Create SAC model with GPU support
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,                    # ✅ GPU enabled here
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        gamma=0.95,
        tau=0.005,
        ent_coef='auto',
        train_freq=64,
        gradient_steps=64,
        target_update_interval=1,
        use_sde=True,
        sde_sample_freq=16,
        tensorboard_log="./sac_leo_tensorboard/"
    )

    # Callback for monitoring
    callback = RewardCallback(check_freq=1000, verbose=1)

    print("🚀 Starting training...")
    model.learn(total_timesteps=50_000, callback=callback, tb_log_name="SAC_LEO")
    model.save("models/sac_leo_beamforming")
    print("✅ Training complete. Model saved to 'models/sac_leo_beamforming'.")

    return model


# -------------------------------
# 5. Evaluation and Plotting
# -------------------------------
def evaluate_model(model, env, episodes=10):
    sum_rates = []
    fairness_vals = []
    powers = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        sum_rates.append(info['sum_rate'])
        fairness_vals.append(info['fairness'])
        powers.append(info['power'])

    print(f"\n📊 Evaluation over {episodes} episodes:")
    print(f"  Avg Sum Rate:  {np.mean(sum_rates):.2f} bps")
    print(f"  Avg Fairness:  {np.mean(fairness_vals):.3f}")
    print(f"  Avg Power:     {np.mean(powers):.3f} W")

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(sum_rates, 'o-', label='Sum Rate')
    axs[0].set_ylabel('Sum Rate (bps)')
    axs[0].grid(True)
    axs[1].plot(fairness_vals, 's-', color='orange', label='Fairness')
    axs[1].set_ylabel('Jain\'s Fairness')
    axs[1].grid(True)
    axs[2].plot(powers, '^-', color='green', label='Power')
    axs[2].set_ylabel('Power (W)')
    axs[2].set_xlabel('Episode')
    axs[2].grid(True)
    plt.tight_layout()
    plt.suptitle('LEO Satellite Resource Allocation Performance', y=0.98)
    plt.show()


# -------------------------------
# 6. Main Execution
# -------------------------------
if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Train model
    model = train_sac_leo()

    # Evaluate
    eval_env = LEOSatelliteEnv(num_users=4, num_subcarriers=6)
    evaluate_model(model, eval_env, episodes=10)
