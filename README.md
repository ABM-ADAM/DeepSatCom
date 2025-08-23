# DeepSatCom: DSPG + SAC for LEO Satellite Resource Allocation

This repository is part of the **DeepSatCom Project**, which focuses on designing deep learning frameworks for **interference management in satellite communications**.  
The provided script demonstrates a **hybrid Discrete Softmax Policy Gradient (DSPG) + Soft Actor-Critic (SAC)** approach for **resource allocation in LEO satellite networks**.

---

## 📌 Features
- Custom `LEOSatelliteEnv` environment:
  - Models users, subcarriers, and satellite positions.
  - Computes SINR, sum rate, Jain’s fairness index, and power usage.
- DSPG-inspired stochastic exploration with SAC agent.
- Training and evaluation pipeline with result visualization.
- Outputs trained SAC model and plots.

---

## 🛠 Requirements
- Python 3.8+
- NumPy
- PyTorch
- Stable-Baselines3
- Gymnasium
- Matplotlib

Install dependencies:
```bash
pip install numpy torch stable-baselines3 gymnasium matplotlib
