# NEAX: A Self-Expanding Neural Architecture via Entropy-Guided Complexity Growth

**Author:** Gabriele Ottiglio  
**Affiliation:** Independent Researcher

---

## 📝 About

This repository contains the implementation of:

**NEAX: A Self-Expanding Neural Architecture via Entropy-Guided Complexity Growth**

NEAX is a dynamic neural network that expands its capacity during training based on internal uncertainty signals and optimization stagnation.

The method introduces:
- **Entropy-guided expansion** based on MC Dropout-inspired uncertainty.
- **Performance plateau detection** to identify learning stagnation.
- **Progressive freezing and lateral connections** to mitigate catastrophic forgetting.
- Designed for **continual learning** and **reinforcement learning** in **non-stationary environments**.

---

## 📄 Preprint

The full paper is available on **Zenodo**:

🔗 [NEAX Preprint (Zenodo)](https://zenodo.org/records/XXXXXXXX)  
_(replace with actual link if published)_

---

## 💻 Code

The repository includes:
- `model.py`: Full PyTorch implementation of NEAX with entropy-based controller.
- `rl.py`: Integration with Stable-Baselines3 (PPO) and OpenAI Gym environments.
- `final_model.py`: Optional variant or legacy version.

### 🔧 Features:
- Self-expanding `NEAXNetwork` with automatic growth and freezing.
- Integration with PPO for environments like `CartPole-v1`.
- Training utilities for supervised benchmarks with entropy tracking.

---

## 📊 Benchmarks (Highlights)

| Environment    | NEAX Reward | PPO Baseline |
|----------------|--------------|---------------|
| CartPole-v1    | 500          | 500           |
| Acrobot-v1     | -62          | -96           |
| Hopper-v4      | 1415         | 790           |
| Humanoid-v4    | 1126         | 815           |

NEAX achieves **faster adaptation**, **better generalization**, and **robust performance** in high-entropy environments.

---

## 📜 License

- **Paper (preprint)**: Creative Commons Attribution 4.0 International (CC BY 4.0).  
- **Code**: MIT License.

---

## 📬 Contact

For collaborations, questions, or feedback:

**Gabriele Ottiglio**  
Email: [gabrieleottiglio@gmail.com]  
