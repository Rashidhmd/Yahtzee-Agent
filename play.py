from yahtzee_api import YahtzeeGame
import torch
import numpy as np
import torch.nn as nn

NUM_STATES = 20
NUM_ACTIONS = 45

NUM_TEST_EPISODES = 500  # Number of episodes to test the agent
MAX_STEPS_TEST = 2000  # Maximum steps per test episode

PATH = 'trained_model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class QNet(nn.Module):
    # Policy Network
    def __init__(self, n_state_vars, n_actions, dim_hidden=128):
        super(QNet, self).__init__()

        # Define a feedforward neural network with hidden layers, ReLU
        #  activations, and an output layer that maps to the number of actions
        self.fc = nn.Sequential(
            nn.Linear(n_state_vars, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, n_actions)
        )

    def forward(self, x):
        # Passes the input through the network layers to output Q-values
        return self.fc(x)


class AgentDQN:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_eval.load_state_dict(torch.load(PATH, weights_only=True))

    def getAction(self, state, index_max):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.net_eval.eval()  # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.net_eval(state)

        # Indexes of descending action values
        index_action = np.argsort(action_values.cpu().data.numpy())[0][::-1]

        # It gets the best action of rank index_max
        action = index_action[index_max]

        return action


def testPlayer(agent, num_tests, max_steps):
    all_score = 0

    # Run the agent in the environment for the specified number of test episodes
    for i in range(num_tests):
        game = YahtzeeGame()

        for idx_step in range(max_steps):

            obs_space = []

            scores = [(0 if v is None else 1) for k, v in game.scorecard.items()]
            dice_counts = [game.dice.count(i) for i in range(1, 7)]

            obs_space.extend(dice_counts)
            obs_space.append(game.rollLeft())
            obs_space.extend(scores)

            # State values formatted to ndarray
            state = np.array(obs_space)

            # Rank of maximum element to get
            ind_max = 0

            # If the action with maximum value it is not valid, it considers the next best action
            while True:
                try:
                    action = agent.getAction(state, ind_max)
                    game.chooseAction(action)
                    break
                except Exception:
                    ind_max += 1

            # End episode if the game is finished
            if game.hasFinished():
                score = game.getTotalReward()
                all_score += score
                print("Game " + str(i + 1) + " - Score : " + str(score))
                if game.bonus:
                    print("Bonus ottenuto")
                else:
                    print("Bonus non ottenuto")
                break

    print("Avg score:", all_score / num_tests)


if __name__ == '__main__':

    # Initialize the pre-trained DQN agent
    agent = AgentDQN(
        n_states=NUM_STATES,
        n_actions=NUM_ACTIONS,
    )

    # Run the trained agent for a series of test episodes
    testPlayer(agent, num_tests=NUM_TEST_EPISODES, max_steps=MAX_STEPS_TEST)
