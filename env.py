from typing import Optional
from yahtzee_api import YahtzeeGame, ACTION_INDEX_LIMIT
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, MultiBinary, Tuple
import numpy as np


class YahtzeeEnv(Env):

    def __init__(self):

        self.game = YahtzeeGame()

        self.action_space = Discrete(45)
        self.observation_space = Tuple(
            (Box(low=0, high=5, shape=(6,), dtype=np.uint8),  # Frequency dices
             Box(low=0, high=2, shape=(1,), dtype=np.uint8),  # Rolls left
             # Scorecard
             MultiBinary(13)))

    def get_observation_space(self):
        game = self.game

        scores = [(0 if v is None else 1) for k, v in game.scorecard.items()]

        # Occurrences of each possible value of the dice
        dice_counts = [game.dice.count(i) for i in range(1, 7)]

        obs_space = (dice_counts, game.rollLeft(), scores)

        # State values formatted to flat array
        format_obs = gym.spaces.utils.flatten(self.observation_space, obs_space)

        return format_obs

    def get_score(self):
        game = self.game

        return game.getTotalReward()

    def reward_function(self, action):
        game = self.game

        # If re-roll action reward is 0
        if action < ACTION_INDEX_LIMIT:
            return 0

        # Get the points of the category filled
        reward = game.getLastReward()

        return reward

    def step(self, action):
        game = self.game

        info_step = {}

        try:
            game.chooseAction(action)
            reward = self.reward_function(action)

            valid_move = True

        except Exception as e:
            valid_move = False
            info_step['Exception'] = e
            reward = -80

        info_step = self.get_info()

        info_step['Action valid'] = valid_move
        info_step['Reward'] = reward
        finished = game.hasFinished()

        return self.get_observation_space(), reward, finished, info_step

    def render(self):
        game = self.game

        print("\nDice:", game.getDiceValues())
        print("Rolls:", game.rollLeft())
        print('Last reward:', game.getLastReward())
        print('Total reward:', game.getTotalReward())
        print('Completed rows:', game.getCompletedRows())
        print('Game finished?', game.hasFinished())

    def get_info(self):
        game = self.game

        return {"Dice:": game.getDiceValues(), "Rolls:": game.rollLeft(), "Completed rows": game.getCompletedRows()}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.game = YahtzeeGame()

        observation = self.get_observation_space()
        info = self.get_info()

        return observation, info
