import datetime as dt
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from game import Direction, Point, SnakeGame
from model import Linear_QNet, QTrainer, float_tensor
from utils import plot_result
from utils.constant import BATCH_SIZE, GAMMA, LR, MAX_MEMORY


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGame) -> np.ndarray:
        head = game.snake[0]
        pt_l = Point(head.x - 20, head.y)
        pt_r = Point(head.x + 20, head.y)
        pt_u = Point(head.x, head.y - 20)
        pt_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # * Danger straight
            (dir_r and game.is_collision(pt_r))
            or (dir_l and game.is_collision(pt_l))
            or (dir_u and game.is_collision(pt_u))
            or (dir_d and game.is_collision(pt_d)),
            # * Danger right
            # Check collision when snake turns right in current location
            (dir_u and game.is_collision(pt_r))
            or (dir_d and game.is_collision(pt_l))
            or (dir_l and game.is_collision(pt_u))
            or (dir_r and game.is_collision(pt_d)),
            # * Danger left
            # Check collision when snake turns left in current location
            (dir_d and game.is_collision(pt_r))
            or (dir_u and game.is_collision(pt_l))
            or (dir_r and game.is_collision(pt_u))
            or (dir_l and game.is_collision(pt_d)),
            # * Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # * Food location
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y,  # Food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        self.trainer.train_step(*zip(*mini_sample))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            final_move[random.randint(0, 2)] = 1
        else:
            state = float_tensor(state)
            pred = self.model(state)
            final_move[torch.argmax(pred).item()] = 1

        return final_move


def train() -> None:
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(
                    weight_name=f"score_{score}.pt",
                    save_dir=Path(
                        f"weights/{dt.datetime.now().strftime(format='%d-%m-%Y')}"
                    ),
                )

            print("Game", agent.n_games, "| Score", score, "| Record:", record)

            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            plot_result(plot_scores, plot_mean_scores)
