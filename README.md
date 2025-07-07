# 🐍 Snake AI with Q-Learning (Pygame)

This project is an implementation of the classic Snake game with a self-learning AI using **Q-learning**, built with **Python** and **Pygame**. The AI learns over time how to play the game better by making decisions that maximize its cumulative reward.

## 📦 Features

- ✅ Q-learning algorithm from scratch (no ML libraries required)
- ✅ Abstract state representation for generalization
- ✅ Relative action space (STRAIGHT, LEFT, RIGHT)
- ✅ Intelligent food spawning and collision handling
- ✅ Reward shaping for faster learning
- ✅ Adjustable training parameters (alpha, gamma, epsilon)

---

## 🎮 Gameplay Overview

The snake starts in a random direction and learns to:
- Find food efficiently
- Avoid walls and self-collisions
- Improve over time through trial and error

The AI improves every time it dies (episode ends), using the Q-table it updates with each move.

---

## 🚀 Getting Started

### Requirements

- Python 3.7+
- pygame (`pip install pygame`)

### Run the Game

```bash
python snake_ai.py
