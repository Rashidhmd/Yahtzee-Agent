# Yahtzee Deep Q-Network Agent

This project implements a Reinforcement Learning agent that learns to play the game of **Yahtzee** using a **Deep Q-Network (DQN)**.

The agent interacts with a custom Yahtzee environment, learns from previous game experiences through replay memory, and progressively improves its decision-making strategy using an epsilon-greedy policy.

## Overview

Yahtzee is a stochastic dice game where the player must decide which dice to reroll and which scorecard category to fill.  
In this project, Yahtzee is modeled as a reinforcement learning problem:

- the **state** represents the current dice configuration, rolls left, and completed scorecard rows;
- the **actions** represent dice rerolls or scorecard category selections;
- the **reward** corresponds to the score obtained by filling a scorecard row;
- the **goal** is to maximize the final score over a complete game.

The agent is trained using Deep Q-Learning to estimate the expected value of each action and improve its policy over time.

## Features

- Custom Python implementation of the Yahtzee game logic
- Gymnasium-compatible reinforcement learning environment
- Deep Q-Network implemented with PyTorch
- Experience replay buffer for more stable learning
- Target network with soft updates
- Epsilon-greedy exploration strategy
- Training loop with score tracking and early stopping
- Model checkpoint saving
- Testing scripts for evaluating the trained agent

## Project Structure

```text
.
├── DQNet.py              # DQN model, replay buffer, and learning logic
├── env.py                # Gymnasium environment wrapper for Yahtzee
├── yahtzee_api.py        # Core Yahtzee game rules and scoring logic
├── train.py              # Main training script
├── Training.py           # Training loop and score plotting
├── Testing.py            # Evaluation utility for a trained agent
├── play.py               # Script to test a saved trained model
└── README.md
```

## Reinforcement Learning Formulation

### State Space

The environment represents the game state using:

- dice value frequencies;
- number of rolls left;
- completed scorecard rows.

The observation is flattened into a numerical vector so that it can be processed by the neural network.

### Action Space

The action space contains **45 discrete actions**:

- actions `0-31`: dice reroll actions;
- actions `32-44`: scorecard category selection actions.

The reroll actions are represented as binary masks indicating which dice should be rerolled.  
The scorecard actions correspond to the 13 Yahtzee scoring categories.

### Reward Function

The reward function is based on the score obtained by the agent during the game:

- reroll actions receive reward `0`;
- valid scorecard actions receive the score obtained in the selected category;
- invalid actions receive a negative penalty.

The objective is to maximize the total score at the end of the game.

## Deep Q-Network

The agent uses a feedforward neural network to approximate the Q-value of each possible action.

The network architecture is:

```text
Input state
   ↓
Linear layer
   ↓
ReLU
   ↓
Linear layer
   ↓
ReLU
   ↓
Output Q-values for all actions
```

The DQN training process includes:

- experience replay;
- Bellman target computation;
- Mean Squared Error loss;
- Adam optimizer;
- target network updates;
- epsilon-greedy action selection.

## Training

The training process is based on an epsilon-greedy policy:

- at the beginning, the agent explores many random actions;
- over time, epsilon decreases;
- the agent gradually exploits the actions with the highest predicted Q-values.

Default training parameters:

```python
BATCH_SIZE = 128
LR = 0.001
EPISODES = 5000
TARGET_SCORE = 250.0
GAMMA = 0.99
MEMORY_SIZE = 10_000
LEARN_STEP = 5
TAU = 0.001

MAX_STEPS = 10_000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
```

To train the agent:

```bash
python train.py
```

After training, the model is saved as:

```text
trained_model.pth
```

## Testing

To evaluate a trained model:

```bash
python play.py
```

The testing script loads the saved model and lets the agent play multiple Yahtzee games without exploration.

During evaluation, the agent selects the action with the highest predicted Q-value. If the selected action is invalid, the script tries the next-best ranked action.

```

## Example Usage

Train the agent:

```bash
python train.py
```

Test the trained model:

```bash
python play.py
```

## Technologies Used

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- tqdm

## Results

During training, the agent learns to improve its Yahtzee score by interacting with the environment and updating its policy through Deep Q-Learning.

The training script records the score obtained in each episode and plots the score history at the end of training.

## License

This project is released under the **MIT License**.

The MIT License is a good choice for this repository because it is simple, permissive, and widely used for educational and open-source software projects. It allows others to use, modify, and distribute the code, while still keeping your name as the original author.
