import torch
import random
import numpy as np
from snake_game_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE, SPEED
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot_enhanced, q_tracker


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # control exploration
        self.gamma = 0.9  # discount rate // smaller gamma means more discounting of future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3) # Neural network with 11 inputs, 256 hidden neurons, and 3 outputs
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_r,
            dir_l,
            dir_u,
            dir_d,
            # Food location
            game.food.x < head.x,  # Food is left
            game.food.x > head.x,  # Food is right
            game.food.y < head.y,  # Food is above
            game.food.y > head.y   # Food is below
            #(game.food.x - head.x) / game.w,  # Normalized food position relative to head
            #(game.food.y - head.y) / game.h,  # Normalized food position
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        # Always get Q-values for tracking, regardless of exploration
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = self.model(state0)
        
        if random.randint(0, 200) < self.epsilon:
            # Exploration: random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: best Q-value action
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        # Track Q-values and action selection for analysis
        # This enables monitoring of learning progress and action preferences
        q_tracker.update_q_values(state, prediction, move)
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state 
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        # get new state
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                # agent.model.save("model.pth")
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Use enhanced plotting with Q-value analysis
            # Pass current state for real-time Q-value monitoring
            current_state = agent.get_state(game)
            plot_enhanced(plot_scores, plot_mean_scores, agent, current_state)
            
            # Print additional Q-value statistics every 10 games
            if agent.n_games % 10 == 0:
                recent_stats = q_tracker.get_recent_stats(window=100)
                print(f"\nQ-Value Analysis (Last 100 steps):")
                for action in range(3):
                    action_name = ['Straight', 'Right', 'Left'][action]
                    if f'action_{action}_mean' in recent_stats:
                        mean_q = recent_stats[f'action_{action}_mean']
                        std_q = recent_stats[f'action_{action}_std']
                        print(f"  {action_name}: Mean Q = {mean_q:.3f}, Std = {std_q:.3f}")
                
                # Print action distribution
                total_actions = sum(q_tracker.action_distribution.values())
                if total_actions > 0:
                    print(f"Action Distribution:")
                    for action in range(3):
                        action_name = ['Straight', 'Right', 'Left'][action]
                        pct = q_tracker.action_distribution[action] / total_actions * 100
                        print(f"  {action_name}: {pct:.1f}%")
                print()
    

if __name__ == "__main__":
    train()