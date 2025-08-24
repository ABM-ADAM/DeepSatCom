import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import random
import os
from skyfield.api import load, EarthSatellite
import warnings
warnings.filterwarnings("ignore")

# ================================
# 1. SGP4: Realistic LEO Orbits
# ================================
def get_satellite_positions(tles, times):
    """
    Simulate satellite positions using SGP4 (TLEs)
    :param tles: List of (name, line1, line2)
    :param times: Skyfield time array
    :return: Dict of (lat, lon) positions over time
    """
    ts = load.timescale()
    satellites = []
    for name, l1, l2 in tles:
        sat = EarthSatellite(l1, l2, name, ts)
        satellites.append(sat)

    positions = {sat.name: [] for sat in satellites}
    for t in times:
        for sat in satellites:
            geocentric = sat.at(t)
            subpoint = geocentric.subpoint()
            positions[sat.name].append((subpoint.latitude.degrees, subpoint.longitude.degrees))
    return positions

# Example TLEs (simplified for demo)
TLES = [
    (
        "SAT-1",
        "1 25544U 98067A   24053.50000000  .00001500  00000+0  31174-4 0  9991",
        "2 25544  51.6400  25.2400 0005820  34.2500  65.8500 15.48999999 10000"
    ),
    (
        "SAT-2",
        "1 25545U 98067B   24053.50000000  .00001500  00000+0  31174-4 0  9998",
        "2 25545  51.6400  35.2400 0005820  44.2500  75.8500 15.48999999 10007"
    )
]


# ================================
# 2. 3GPP NTN Channel Model
# ================================
def compute_3gpp_ntn_pathloss(distance_km, frequency=20e9, scenario='rural'):
    """3GPP TR 38.811 path loss model"""
    c = 3e8
    fsl = 20 * np.log10(distance_km * 1000) + 20 * np.log10(frequency) - 20 * np.log10(c) + 20 * np.log10(4 * np.pi)
    
    excess = {
        'rural': np.random.normal(10, 3),
        'urban': np.random.normal(20, 5),
        'dense-urban': np.random.normal(30, 6)
    }[scenario]
    
    total_loss_dB = fsl + excess
    return 10 ** (-total_loss_dB / 10)  # Linear gain

def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute distance between two Earth points"""
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# ================================
# 3. Multi-Agent LEO Environment
# ================================
class MultiAgentLEOEnv:
    def __init__(self, num_users=4, num_subcarriers=6, num_sats=2, max_power=10.0):
        self.num_users = num_users
        self.num_subcarriers = num_subcarriers
        self.num_sats = num_sats
        self.max_power = max_power
        self.noise_power = 1e-11
        self.bandwidth_per_sc = 1e6
        self.scenario = 'rural'

        # User positions (static for now)
        self.user_positions = np.random.rand(num_users, 2) * 100  # (lat, lon) dummy

        # SGP4 setup
        self.ts = load.timescale()
        self.sim_times = self.ts.utc(2024, 3, 1, 12, range(0, 60, 10))  # 6 time steps
        self.sat_positions = get_satellite_positions(TLES, self.sim_times)
        self.sat_names = [name for name, _, _ in TLES]
        self.current_timestep = 0

        # State: [user_pos, sat_pos, dist, ch_gain] per sat
        self.state_dim = num_users * 2 + 2 + num_users + num_users * num_subcarriers
        self.action_dim = num_subcarriers + 2  # alloc_logits + beam_real/imag

    def reset(self):
        self.current_timestep = 0
        self._update_satellite_positions()
        return self._get_states()

    def _update_satellite_positions(self):
        self.sat_latlons = []
        for name in self.sat_names:
            lat, lon = self.sat_positions[name][self.current_timestep]
            self.sat_latlons.append((lat, lon))

    def _get_states(self):
        states = []
        for s in range(self.num_sats):
            sat_lat, sat_lon = self.sat_latlons[s]
            distances = []
            gains = []
            for u in range(self.num_users):
                ulat, ulon = self.user_positions[u]
                d = haversine_distance(sat_lat, sat_lon, ulat, ulon)
                distances.append(d)
                gain = compute_3gpp_ntn_pathloss(d, scenario=self.scenario)
                gains.append(gain)
            state = np.concatenate([
                self.user_positions.flatten(),        # 8
                [sat_lat, sat_lon],                   # 2
                np.array(distances) / 2000.0,         # ~2000km max
                np.array(gains).repeat(self.num_subcarriers)  # Expand to (users * subcarriers)
            ])
            states.append(state.astype(np.float32))
        return states

    def step(self, actions):
        # actions: list of (num_users, action_dim) arrays — one per satellite
        sum_rates = np.zeros(self.num_users)
        power_used = 0.0

        for s in range(self.num_sats):
            act = actions[s]  # (num_users, action_dim)
            alloc_logits = act[:, :self.num_subcarriers]  # (users, subcarriers)
            beam_real = act[:, self.num_subcarriers]
            beam_imag = act[:, self.num_subcarriers + 1]
            beam_weights = beam_real + 1j * beam_imag

            # Normalize power
            pwr = np.sum(np.abs(beam_weights)**2)
            if pwr > self.max_power:
                beam_weights *= np.sqrt(self.max_power / (pwr + 1e-8))
            power_used += np.sum(np.abs(beam_weights)**2)

            # Get channel gains at this timestep
            gains = []
            for u in range(self.num_users):
                d = haversine_distance(*self.sat_latlons[s], *self.user_positions[u])
                g = compute_3gpp_ntn_pathloss(d, scenario=self.scenario)
                gains.append(g)
            gains = np.array(gains)

            # Compute rates
            for u in range(self.num_users):
                sc = np.argmax(alloc_logits[u])
                signal = (gains[u] * abs(beam_weights[u]))**2
                interference = 0.0
                for s2 in range(self.num_sats):
                    if s2 != s:
                        # Interference from other satellites on same subcarrier
                        # Simplified: assume same subcarrier usage
                        interference += 0.1 * signal  # Approximate cross-sat interference
                sinr = signal / (interference + self.noise_power)
                rate = self.bandwidth_per_sc * np.log2(1 + sinr)
                sum_rates[u] += rate

        # Move to next time step
        self.current_timestep = (self.current_timestep + 1) % len(self.sim_times)
        self._update_satellite_positions()
        next_states = self._get_states()

        # Reward: sum rate + fairness - power
        total_rate = np.sum(sum_rates)
        fairness = (total_rate**2) / (self.num_users * np.sum(sum_rates**2) + 1e-8)
        reward_per_sat = (1e-6 * total_rate + 5 * fairness - 0.01 * power_used) / self.num_sats

        rewards = [reward_per_sat] * self.num_sats
        done = self.current_timestep == 0  # One orbit cycle

        info = {
            'sum_rate': total_rate,
            'fairness': fairness,
            'power': power_used
        }
        return next_states, rewards, done, info


# ================================
# 4. MA-SAC Implementation
# ================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.net(x)


class MASAC:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        total_s = num_agents * state_dim
        total_a = num_agents * action_dim

        self.actors = [Actor(state_dim, action_dim).to(device) for _ in range(num_agents)]
        self.actor_optims = [optim.Adam(a.parameters(), lr=3e-4) for a in self.actors]

        self.critics = [Critic(total_s, total_a).to(device) for _ in range(2)]
        self.target_critics = [Critic(total_s, total_a).to(device) for _ in range(2)]
        self.critic_optims = [optim.Adam(c.parameters(), lr=3e-4) for c in self.critics]

        for tc, c in zip(self.target_critics, self.critics):
            tc.load_state_dict(c.state_dict())

        self.buffer = deque(maxlen=50000)
        self.gamma = 0.95
        self.tau = 0.005
        self.ent_coef = 0.2

    def select_action(self, states, evaluate=False):
        actions = []
        with torch.no_grad():
            for i, s in enumerate(states):
                s = torch.FloatTensor(s).to(device)
                mean, std = self.actors[i](s)
                dist = torch.distributions.Normal(mean, std)
                a = dist.sample() if not evaluate else mean
                actions.append(a.cpu().numpy())
        return actions

    def store(self, states, actions, rewards, next_states, done):
        self.buffer.append((states, actions, rewards, next_states, done))

    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).view(batch_size, self.num_agents, -1).to(device)
        actions = torch.FloatTensor(actions).view(batch_size, self.num_agents, -1).to(device)
        rewards = torch.FloatTensor(rewards).view(batch_size, self.num_agents).to(device)
        next_states = torch.FloatTensor(next_states).view(batch_size, self.num_agents, -1).to(device)
        dones = torch.BoolTensor(dones).view(batch_size, 1).to(device)

        total_next_s = next_states.view(batch_size, -1)
        total_a = actions.view(batch_size, -1)
        total_s = states.view(batch_size, -1)

        # Update Critics
        with torch.no_grad():
            next_actions = []
            log_probs = []
            for i in range(self.num_agents):
                mean, std = self.actors[i](next_states[:, i, :])
                dist = torch.distributions.Normal(mean, std)
                sampled_a = dist.sample()
                lp = dist.log_prob(sampled_a).sum(dim=1, keepdim=True)
                next_actions.append(sampled_a)
                log_probs.append(lp)
            next_actions = torch.cat(next_actions, dim=1)
            log_probs = torch.cat(log_probs, dim=1).sum(dim=1, keepdim=True)

            q1 = self.target_critics[0](total_next_s, next_actions)
            q2 = self.target_critics[1](total_next_s, next_actions)
            min_q = torch.min(q1, q2)
            target_q = rewards.sum(dim=1, keepdim=True) + \
                       self.gamma * (1 - dones) * (min_q - self.ent_coef * log_probs)

        curr_q1 = self.critics[0](total_s, total_a)
        curr_q2 = self.critics[1](total_s, total_a)
        q1_loss = nn.MSELoss()(curr_q1, target_q)
        q2_loss = nn.MSELoss()(curr_q2, target_q)

        self.critic_optims[0].zero_grad()
        q1_loss.backward()
        self.critic_optims[0].step()

        self.critic_optims[1].zero_grad()
        q2_loss.backward()
        self.critic_optims[1].step()

        # Update Actors
        actor_actions = []
        log_probs = []
        for i in range(self.num_agents):
            mean, std = self.actors[i](states[:, i, :])
            dist = torch.distributions.Normal(mean, std)
            sampled_a = dist.sample()
            lp = dist.log_prob(sampled_a).sum(dim=1, keepdim=True)
            actor_actions.append(sampled_a)
            log_probs.append(lp)
        actor_actions = torch.cat(actor_actions, dim=1)
        log_probs = torch.cat(log_probs, dim=1).sum(dim=1, keepdim=True)

        q = self.critics[0](total_s, actor_actions)
        actor_loss = (self.ent_coef * log_probs - q).mean()

        for opt in self.actor_optims:
            opt.zero_grad()
        actor_loss.backward()
        for opt in self.actor_optims:
            opt.step()

        # Target update
        for tc, c in zip(self.target_critics, self.critics):
            for t, src in zip(tc.parameters(), c.parameters()):
                t.data.copy_(self.tau * src.data + (1 - self.tau) * t.data)


# ================================
# 5. Training Loop
# ================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    env = MultiAgentLEOEnv(num_users=3, num_subcarriers=4, num_sats=2)
    agent = MASAC(num_agents=env.num_sats, state_dim=env.state_dim, action_dim=env.action_dim)

    total_steps = 0
    episode_rewards = []
    eval_interval = 50

    print("Starting MA-SAC training...")

    for episode in range(500):
        states = env.reset()
        episode_reward = 0
        done = False

        while not done:
            actions = agent.select_action(states)
            next_states, rewards, done, info = env.step(actions)
            agent.store(states, actions, rewards, next_states, done)

            states = next_states
            episode_reward += sum(rewards)
            total_steps += 1

            if total_steps % 50 == 0:
                agent.update(batch_size=128)

        episode_rewards.append(episode_reward)

        if episode % eval_interval == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Sum Rate = {info['sum_rate']:.2f}, Fairness = {info['fairness']:.3f}")

    # ================================
    # 6. Evaluation & Plotting
    # ================================
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('MA-SAC Training: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    print(f"✅ Final Performance: Avg Sum Rate = {info['sum_rate']:.2f} bps")
