def testPlayer1(env, agent, num_tests, max_steps):
    avg_score = 0
    count = 0
    # Run the agent in the environment for the specified number of test episodes
    for i in range(num_tests):
        state, _ = env.reset()

        for idx_step in range(max_steps):
            # Select action without exploration for deterministic behavior
            action = agent.getAction(state, 0)

            # Render the environment to visualize the agent’s performance
            # env.render()

            # Perform the action and observe the next state and reward
            state, reward, done, _ = env.step(action)

            # End episode if the agent reaches a terminal state
            if done:
                score = env.get_score()
                avg_score += score
                count += 1
                print("Score partita ", i + 1, score)
                break

    print("Avg score:", avg_score / num_tests)
    print("Count", count)

    env.close()  # Close the environment after testing to free resources
