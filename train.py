import gymnasium as gym
from yahtzee_api import YahtzeeGame
import numpy as np
from env import YahtzeeEnv
from DQNet import DQN, device
from Training import train, plotScore
import torch
from Testing import testPlayer1

BATCH_SIZE = 128  # Number of experiences sampled per learning step
LR = 0.001  # Learning rate for optimizer
EPISODES = 5000  # Maximum number of episodes to train
TARGET_SCORE = 250.0  # Early stop if average score reaches this value
GAMMA = 0.99  # Discount factor for future rewards
MEMORY_SIZE = 10_000  # Maximum capacity of replay memory
LEARN_STEP = 5  # Frequency (in steps) of learning updates
TAU = 0.001  # Soft update rate for the target network
SAVE_CHKPT = True  # Option to save trained model checkpoint

# Exploration parameters
MAX_STEPS = 10_000  # Maximum steps per episode
EPS_START = 1.0  # Initial epsilon for exploration (100% exploration at start)
EPS_END = 0.1  # Minimum epsilon (final exploration rate)
EPS_DECAY = 0.995  # Epsilon decay rate (controls exploration reduction)

# Initialize the Yahtzee environment
env = YahtzeeEnv()

# Define number of vars in states for Tuple of observation space
num_states = gym.spaces.utils.flatdim(env.observation_space)
num_actions = env.action_space.n

# Initialize the DQN agent with customizable parameters
agent = DQN(
    n_states=num_states,
    n_actions=num_actions,
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    gamma=GAMMA,
    mem_size=MEMORY_SIZE,
    learn_step=LEARN_STEP,
    tau=TAU,
)

# Train the agent using the train function and record the score history
score_hist = train(
    env, agent, n_episodes=EPISODES, max_steps=MAX_STEPS,
    eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
    target_score=TARGET_SCORE, do_store_checkpoint=SAVE_CHKPT
)

plotScore(score_hist)
testPlayer1(env, agent, 500, 5000)

# Free up GPU memory if using CUDA
if str(device) == "cuda":
    torch.cuda.empty_cache()
