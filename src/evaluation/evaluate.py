import torch
from src.training.env import Game2048Env
from src.training.dqn import DQNAgent

def evaluate(model_path, episodes=20):
    env = Game2048Env()
    agent = DQNAgent(...)
    
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()

    scores = []
    max_tiles = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, epsilon=0)
            state, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(total_reward)
        max_tiles.append(info.get("max_tile", 0))

    print("Avg Score:", sum(scores)/len(scores))
    print("Avg Max Tile:", sum(max_tiles)/len(max_tiles))
    