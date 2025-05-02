# SARSA Learning Algorithm


## AIM
To implement SARSA Learning Algorithm.

## PROBLEM STATEMENT
The problem might involve teaching an agent to interact optimally with an environment (e.g., gym-walk), where the agent must learn to choose actions that maximize cumulative rewards using RL algorithms like SARSA and Value Iteration.

## SARSA LEARNING ALGORITHM
1. Initialize the Q-table, learning rate α, discount factor γ, exploration rate ϵ, and the number of episodes.<br>
2. For each episode, start in an initial state s, and choose an action a using the ε-greedy policy.<br>
3. Take action a, observe the reward r and the next state s′ , and choose the next action a′ using the ε-greedy policy.<br>
4. Update the Q-value for the state-action pair (s,a) using the SARSA update rule.<br>
5. Update the current state to s′ and the current action to a′.<br>
6. Repeat steps 3-5 until the episode reaches a terminal state.<br>
7. After each episode, decay the exploration rate 𝜖 and learning rate α, if using decay schedules.<br>
8. Return the Q-table and the learned policy after completing all episodes.<br>

## SARSA LEARNING FUNCTION
### Name: KULASEKARAPANDIAN K
### Register Number: 212222240052

```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilon[e])
      while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = select_action(next_state, Q, epsilon[e])
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state, action = next_state, next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/c380b58a-acd4-4566-b8dd-b2d024b9a8e0)
![image](https://github.com/user-attachments/assets/c370d280-f924-4c07-9194-433f7f58c8ad)
![image](https://github.com/user-attachments/assets/7d53e901-cf00-44c3-9490-29763d9af106)

![image](https://github.com/user-attachments/assets/caaf7b55-2ef1-4372-9e17-0f1179050f3b)
![image](https://github.com/user-attachments/assets/65d4dff3-4477-4b4f-981b-397e6c871d69)


## RESULT:
Thus, to implement SARSA learning algorithm is executed successfully.
