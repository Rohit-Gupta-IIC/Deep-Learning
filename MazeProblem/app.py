from maze_env import Maze
from RL_agents import QLearning
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

episode_count = 50
episodes = range(episode_count)
rewards = []
movements = []


def run_experiment():
    for episode in episodes:
        print("Episode {0}/{1}".format(episode, episodes))
        observations = env.reset()
        moves = 0

        while True:
            env.render()
            action = q_learning_agent.choosing_actions(str(observations))
            observation_, reward, done = env.get_reward_state(action=action)
            moves += 1

            q_learning_agent.learn(str(observations), action, reward, str(observation_))
            observations = observation_

            if done:
                movements.append(moves)
                rewards.append(reward)
                print("Reward : {0}, Moves: {1}".format(reward, moves))
                break

    print("Game is Over")
    plot_reward_movements()


def plot_reward_movements():
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(episodes, movements)
    plt.xlabel("Episodes")
    plt.ylabel("Movements")

    plt.subplot(2, 1, 2)
    plt.step(episodes, rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("plot_movement_qlearning.png")
    plt.show()


if __name__ == "__main__":
    env = Maze()
    q_learning_agent = QLearning(actions=list(range(env.n_actions)))
    env.window.after(10, run_experiment())
    env.window.mainloop()
