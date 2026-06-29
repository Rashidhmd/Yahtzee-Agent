from tqdm import trange
import torch
import numpy as np
import matplotlib.pyplot as plt

CHECKPOINT_NAME = 'trained_model.pth'
RECENT_EPISODES = 100  # Number of episodes for average score in early stopping
MIN_EPISODES_FOR_STOP = 100  # Ensures enough episodes before evaluating target


def train(
        env, agent, n_episodes, max_steps,
        eps_start, eps_end, eps_decay,
        target_score, do_store_checkpoint
):
    # Initialize score history and epsilon (exploration rate)
    score_hist = []
    epsilon = eps_start

    # Progress bar format for tracking training progress
    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt}' \
                 ' [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    for idx_epi in pbar:
        # Reset the environment for a new episode
        state, _ = env.reset()
        score = 0

        for idx_step in range(max_steps):

            # Select an action based on the current policy (epsilon-greedy)
            action = agent.getAction(state, epsilon)

            next_state, reward, done, info_step = env.step(action)

            # Store experience in memory and update the agent
            agent.save2Memory(state, action, reward, next_state, done)
            state = next_state  # Move to the next state

            if reward > 0:
                score += reward

            # Check if the episode is finished
            if done:
                assert score == env.get_score()
                break

        # Track scores and decay epsilon for less exploration over time
        score_hist.append(score)
        score_avg = np.mean(score_hist[-RECENT_EPISODES:])
        epsilon = max(eps_end, epsilon * eps_decay)  # Decay epsilon

        # Update the progress bar with the current score and average
        pbar.set_postfix_str(
            f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}"
        )
        pbar.update(0)

        # Early stopping condition if target score is achieved
        if len(score_hist) >= MIN_EPISODES_FOR_STOP and score_avg >= target_score:
            print("\nTarget Reached!")
            break

    # Print completion message based on early stopping or max episodes
    if (idx_epi + 1) < n_episodes:
        print("\nTraining complete - target reached!")
    else:
        print("\nTraining complete - maximum episodes reached.")

    # Save the trained model if specified
    if do_store_checkpoint:
        torch.save(agent.net_eval.state_dict(), CHECKPOINT_NAME)

    return score_hist


def plotScore(scores):
    # Plot the agent's score history to visualize learning progress
    plt.figure()
    plt.plot(scores)
    plt.title("Score History")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()
