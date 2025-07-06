
# 🦖 Dino Game AI – Reinforcement Learning with Stable-Baselines3

<img width="822" height="722" alt="Image" src="https://github.com/user-attachments/assets/6d4c4442-8b8d-4df4-b2e5-7f2b097d1c28" />

## 🎯 Project Overview

This project demonstrates how to train a reinforcement learning (RL) agent using Deep Q-Learning (DQN) from the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library to play the **Google Chrome Dino Game** automatically.

It was created to understand the basics of reinforcement learning, including:
- Custom environment creation (`gym.Env`)
- Observation & action spaces
- Model-agent interaction and response
- Training and evaluation using RL

---

## 🎥 Demo Video

https://github.com/user-attachments/assets/56ff5561-2387-451b-9eb2-6772f356bfec

---

## 🧠 Core Concepts

- **Gym Environment**: Custom wrapper using screenshots and `pydirectinput` to interact with the game.
- **Computer Vision**: Captures game state and preprocesses using OpenCV.
- **OCR**: Detects "GAME OVER" using Tesseract for episode termination.
- **DQN Agent**: Trained with a convolutional policy to learn actions based on screen frames.

---

## 🛠 Setup Instructions

### 📦 Prerequisites

Ensure you have:
- Python 3.8+
- NVIDIA GPU with CUDA 12.6 (verified)
- Chrome Dino game accessible (offline or local clone)

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dino-rl-agent.git
cd dino-rl-agent

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install stable-baselines3[extra] protobuf==3.20.*
pip install gym mss pydirectinput pytesseract opencv-python
```

---

## 🚀 How to Run

### 🏋️‍♂️ Train the Agent

```python
from stable_baselines3 import DQN

model = DQN(
    'CnnPolicy',
    env,
    tensorboard_log='./logs/',
    verbose=1,
    buffer_size=600000,
    learning_starts=1000
)

model.learn(total_timesteps=100000, callback=TrainAndLoggingCallback(...))
```

### 🧪 Test the Agent

```python
obs = test_env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
```

---

## 📂 Project Structure

```bash
├── dino_rl_agent.py          # Main RL environment and logic
├── train_model.py            # Training script
├── test_model.py             # Evaluation script
├── utils/                    # Helper functions (if any)
├── train/                    # Saved models (checkpoints)
├── logs/                     # TensorBoard logs
├── README.md
```

---

## 📌 Notes

- Make sure the Dino game is in focus and not covered during training.
- Resolution and OCR sensitivity may vary between systems—tweak bounding boxes if needed.
- For best performance, close all background windows and apps using GPU.

---

## 🙌 Acknowledgements

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [OpenAI Gym](https://gym.openai.com/)
- [MSS](https://github.com/BoboTiG/python-mss)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)


---

> ⭐️ Star the repo if you found it useful or want to follow future improvements!
