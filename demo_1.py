import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env, spaces
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# -------------------------------
# 1. LEO Satellite Environment
# -------------------------------
class LEOSatelliteEnv(Env):
    def __init__(self, num_users=5, num_subcarriers=8, num_satellites=1, max_power=10.0):
        super(LEOSatelliteEnv, self).__init__()
        self.num_users = num_users
        self.num_subcarriers = num_subcarriers
        self.num_satellites = num_satellites
        self.max_power = max_power
        self.noise_power = 1e-11  # -100 dBm
        self.bandwidth_per_sc = 1e6  # 1 MHz per subcarrier

        # State: [user_distances (num_users), channel_gains (num_users*num_subcarriers), satellite_positions (2)]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(num_users + num_users * num_subcarriers + 2,), dtype=np.float32
        )

        # Action: subcarrier allocation (one-hot per user) + beamforming weights (complex gains, represented as real/imag)
        # We flatten: [allocation (num_users * num_subcarriers), beam_weights_real (num_users), beam_weights_imag (num_users)]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_users * num_subcarriers + 2 * num_users,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.user_positions = np.random.rand(self.num_users, 2) * 100  # 100km x 100km area
        self.satellite_positions = np.random.rand(self.num_satellites, 2) * 100
        self.distances = np.linalg.norm(self.user_positions - self.satellite_positions[0], axis=1)
        self.channel_gains = self._compute_channel_gains()
        self.state = self._get_state()
        return self.state.astype(np.float32), {}

    def _compute_channel_gains(self):
        # Simplified path loss: L = 1 / d^2 (free space with fading)
        h = 1 / (self.distances[:, None] + 1e-3)  # Avoid division by zero
        fading = (np.random.randn(self.num_users, self.num_subcarriers) +
                  1j * np.random.randn(self.num_users, self.num_subcarriers)) * 0.1
        h = h * (1 + fading.real)  # Real-valued effective gain
        return np.abs(h).astype(np.float32)

    def _get_state(self):
        return np.concatenate([
            self.distances / 100.0,
            self.channel_gains.flatten(),
            self.satellite_positions[0] / 100.0
        ])

    def step(self, action):
        # Parse action
        alloc_logits = action[:self.num_users * self.num_subcarriers]
        beam_real = action[self.num_users * self.num_subcarriers:self.num_users * self.num_subcarriers + self.num_users]
        beam_imag = action[self.num_users * self.num_subcarriers + self.num_users:]

        # Reshape allocation to (users, subcarriers)
        alloc_logits = alloc_logits.reshape((self.num_users, self.num_subcarriers))
        subcarrier_allocation = np.argmax(alloc_logits, axis=1)  # One subcarrier per user

        # One-hot allocation matrix
        A = np.zeros((self.num_users, self.num_subcarriers))
        for u in range(self.num_users):
            A[u, subcarrier_allocation[u]] = 1

        # Beamforming: combine real/imag to complex weights
        beam_weights = (beam_real + 1j * beam_imag)
        power_per_user = np.abs(beam_weights) ** 2
        total_power = np.sum(power_per_user)
        if total_power > self.max_power:
            beam_weights = beam_weights * np.sqrt(self.max_power / (total_power + 1e-8))
            power_per_user = np.abs(beam_weights) ** 2

        # Compute SINR and rate
        rates = []
        for u in range(self.num_users):
            sc = subcarrier_allocation[u]
            h_u = self.channel_gains[u, sc]
            signal = (h_u * beam_weights[u]) ** 2  # |h w|^2
            interference = 0
            for u2 in range(self.num_users):
                if u2 != u and subcarrier_allocation[u2] == sc:
                    h_u2 = self.channel_gains[u2, sc]
                    interference += (h_u2 * beam_weights[u2]) ** 2
            sinr = signal / (interference + self.noise_power)
            rate = self.bandwidth_per_sc * np.log2(1 + sinr)
            rates.append(rate)

        sum_rate = np.sum(rates)
        fairness = (np.sum(rates) ** 2) / (self.num_users * np.sum(np.square(rates)) + 1e-8)  # Jain's fairness
        power_cost = total_power

        # Reward: weighted sum rate and fairness, penalize high power
        reward = sum_rate * 1e-6 + 5 * fairness - 0.01 * power_cost

        # Simulate environment change (e.g., user/satellite movement)
        self.user_positions += (np.random.randn(*self.user_positions.shape) * 0.5)
        self.user_positions = np.clip(self.user_positions, 0, 100)
        self.distances = np.linalg.norm(self.user_positions - self.satellite_positions[0], axis=1)
        self.channel_gains = self._compute_channel_gains()
        self.state = self._get_state()

        terminated = False
        truncated = False
        info = {
            'sum_rate': sum_rate,
            'fairness': fairness,
            'power': power_cost
        }

        return self.state.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        print(f"User positions: {self.user_positions}")
        print(f"Sum Rate: {self.last_info['sum_rate']:.2f} bps")


# -------------------------------
# 2. Custom Feature Extractor (Optional)
# -------------------------------
class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=64):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)


# -------------------------------
# 3. Training the SAC Agent
# -------------------------------
def train_sac_leo():
    env = LEOSatelliteEnv(num_users=4, num_subcarriers=6)
    env = DummyVecEnv([lambda: env])  # Required for Stable-Baselines3

    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=[128, 128]  # Two hidden layers
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        gamma=0.95,
        tau=0.005,
        ent_coef='auto',
        train_freq=64,
        gradient_steps=64,
        target_update_interval=1,
        use_sde=True,  # Stochastic Policy (DSPG-style exploration)
        sde_sample_freq=16,
        device='cuda' 
    )

    print("Starting training...")
    model.learn(total_timesteps=10000, log_interval=10)

    model.save("sac_leo_beamforming")
    print("Training complete. Model saved.")

    return model


# -------------------------------
# 4. Evaluation and Plotting
# -------------------------------
def evaluate_model(model, env, episodes=5):
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

    print(f"Avg Sum Rate: {np.mean(sum_rates):.2f} bps")
    print(f"Avg Fairness: {np.mean(fairness_vals):.3f}")
    print(f"Avg Power: {np.mean(powers):.3f} W")

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    axs[0].plot(sum_rates, label='Sum Rate')
    axs[0].set_title('Sum Rate per Episode')
    axs[1].plot(fairness_vals, color='orange', label='Fairness')
    axs[1].set_title('Jain\'s Fairness')
    axs[2].plot(powers, color='green', label='Total Power')
    axs[2].set_title('Transmit Power')
    plt.tight_layout()
    plt.show()


# -------------------------------
# 5. Run Training & Evaluation
# -------------------------------
if __name__ == "__main__":
    # Train the model
    model = train_sac_leo()

    # Evaluate
    eval_env = LEOSatelliteEnv(num_users=4, num_subcarriers=6)
    print("Environment evaluation finished...")
    evaluate_model(model, eval_env, episodes=2)